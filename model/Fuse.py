import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, trunc_normal_
from torchvision import ops

class DeformAlign(nn.Module):

    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        kernel_num = self.kernel_size[0] * self.kernel_size[1]
        offset_channels = 2 * kernel_num
        self.offset_conv = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.ReLU(inplace=True), nn.Conv2d(dim, dim, 3, padding=1), nn.Conv2d(dim, offset_channels, 1))
        self.mask_conv = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.ReLU(inplace=True), nn.Conv2d(dim, dim, 3, padding=1), nn.Conv2d(dim, kernel_num, 1), nn.Sigmoid())
        self.deform_conv = ops.DeformConv2d(dim, dim, kernel_size=self.kernel_size, stride=stride, padding=padding)
        self.weight = nn.Sequential(nn.Conv2d(dim * 2, dim, 1), nn.Sigmoid())
        self.init_weights()

    def init_weights(self):
        for m in self.offset_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        for m in self.mask_conv:
            if isinstance(m, nn.Conv2d):
                nn.init.constant_(m.weight, 0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        cat_feat = torch.cat([x1, x2], dim=1)
        offset = self.offset_conv(cat_feat)
        mask = self.mask_conv(cat_feat)
        x2_aligned = self.deform_conv(x2, offset, mask)
        weight = self.weight(torch.cat([x1, x2_aligned], dim=1))
        fused = weight * x1 + (1 - weight) * x2_aligned
        return (fused, x1, x2_aligned)

class RoPE(nn.Module):
    """Rotary Positional Embedding."""

    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()
        channel_dims, feature_dim = (shape[:-1], shape[-1])
        k_max = feature_dim // (2 * len(channel_dims))
        assert feature_dim % k_max == 0
        theta_ks = 1 / base ** (torch.arange(k_max) / k_max)
        angles = torch.cat([t.unsqueeze(-1) * theta_ks for t in torch.meshgrid([torch.arange(d) for d in channel_dims], indexing='ij')], dim=-1)
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer('rotations', rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossLinearAttention(nn.Module):
    """Cross Modal Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

    def forward(self, x_q, x_k, x_v):
        """
        Args:
            x_q: query features with shape of (B, N, C)
            x_kv: key/value features with shape of (B, N, C)
        """
        B, N, C = x_q.shape
        H, W = self.input_resolution
        num_heads = self.num_heads
        head_dim = self.head_dim
        q = self.q(x_q)
        k = self.k(x_k)
        v = x_v
        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = self.rope(q.reshape(B, H, W, C)).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k_rope = self.rope(k.reshape(B, H, W, C)).reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-06)
        kv = k_rope.transpose(-2, -1) * N ** (-0.5) @ (v * N ** (-0.5))
        x = q_rope @ kv * z
        x = x.transpose(1, 2).reshape(B, N, C)
        v_spatial = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.lepe(v_spatial).permute(0, 2, 3, 1).reshape(B, N, C)
        return x

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class FusionMLLABlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, mlp_ratio=4.0, qkv_bias=True, drop=0.0, drop_path=0.0, act_layer=nn.GELU, norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.cpe_x1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.cpe_x2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.cpe_fused = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm_x1 = norm_layer(dim)
        self.norm_x2 = norm_layer(dim)
        self.norm_fuse = norm_layer(dim)
        self.norm_mlp = norm_layer(dim)
        self.in_proj_x1 = nn.Linear(dim, dim)
        self.in_proj_x2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.cross_attn1 = CrossLinearAttention(dim, input_resolution, num_heads, qkv_bias)
        self.cross_attn2 = CrossLinearAttention(dim, input_resolution, num_heads, qkv_bias)
        self.attn_proj = nn.Linear(dim * 2, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.fuse = DeformAlign(dim)
        self.proj_fused = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.dwc_x1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dwc_x2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.fusion_mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x1, x2):
        """
        Args:
            x1: features of modality 1 with shape (B, N, C)
            x2: features of modality 2 with shape (B, N, C)
        Returns:
            Fused features with shape (B, N, C)
        """
        H, W = self.input_resolution
        B, _, C = x1.shape
        x1 = x1.permute(0, 2, 1).view(B, C, H, W)
        x2 = x2.permute(0, 2, 1).view(B, C, H, W)
        fused, x1, x2 = self.fuse(x1, x2)
        x1 = x1 + self.cpe_x1(x1)
        x2 = x2 + self.cpe_x2(x2)
        x1 = x1.flatten(2).permute(0, 2, 1)
        x2 = x2.flatten(2).permute(0, 2, 1)
        fused = fused.flatten(2).permute(0, 2, 1)
        shortcut_X1 = x1
        shortcut_X2 = x2
        shortcut_fused = fused
        x1_norm = self.norm_x1(x1)
        x2_norm = self.norm_x2(x2)
        fused_norm = self.norm_fuse(fused)
        x1_proj = self.in_proj_x1(x1_norm)
        x2_proj = self.in_proj_x2(x2_norm)
        fused_proj = self.proj_fused(fused_norm)
        x1 = self.act(self.dwc_x1(x1_proj.permute(0, 2, 1).view(B, C, H, W))).flatten(2).permute(0, 2, 1)
        x2 = self.act(self.dwc_x2(x2_proj.permute(0, 2, 1).view(B, C, H, W))).flatten(2).permute(0, 2, 1)
        x1_attn = shortcut_X1 + self.drop_path(self.cross_attn1(x1, x2, x2))
        x2_attn = shortcut_X2 + self.drop_path(self.cross_attn2(x2, x1, x1))
        fused_attn = shortcut_fused + self.drop_path(self.out_proj(self.attn_proj(torch.cat([x1_attn, x2_attn], dim=2)) * self.act(fused_proj)))
        fused_attn = fused_attn + self.cpe_fused(fused_attn.permute(0, 2, 1).view(B, C, H, W)).flatten(2).permute(0, 2, 1)
        fused = fused_attn + self.drop_path(self.fusion_mlp(self.norm_mlp(fused_attn)))
        return fused
