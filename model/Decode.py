import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import ops
from timm.layers import DropPath, trunc_normal_

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

class CrossLinearAttention(nn.Module):

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

class DecodeBlock(nn.Module):

    def __init__(self, dim, num_heads, input_resolution, drop_ratio=0.0, drop_path_ratio=0.2):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.cpe_x = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.cpe_mlp = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.cpe_skip = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act_proj = nn.Linear(dim, dim)
        self.act = nn.SiLU()
        self.cross_attn = CrossLinearAttention(dim=dim, num_heads=num_heads, input_resolution=input_resolution)
        self.norm_x = nn.LayerNorm(dim)
        self.norm_skip = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim)
        self.x_proj = nn.Linear(dim, dim)
        self.skip_x_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dwc_x = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.dwc_skip = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.mlp = Mlp(dim, dim * 4, dim, drop=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()

    def forward(self, x, skip_x):
        B, C, H, W = x.shape
        x = x + self.cpe_x(x)
        skip_x = skip_x + self.cpe_skip(skip_x)
        x_seq = x.flatten(2).permute(0, 2, 1)
        skip_seq = skip_x.flatten(2).permute(0, 2, 1)
        norm_x = self.norm_x(x_seq)
        norm_skip = self.norm_skip(skip_seq)
        act_res = self.act(self.act_proj(norm_x))
        q = self.x_proj(norm_x)
        kv = self.skip_x_proj(norm_skip)
        q = self.act(self.dwc_x(q.permute(0, 2, 1).view(B, C, H, W))).flatten(2).permute(0, 2, 1)
        kv = self.act(self.dwc_skip(kv.permute(0, 2, 1).view(B, C, H, W))).flatten(2).permute(0, 2, 1)
        cross_attn = self.cross_attn(q, kv, kv)
        x_seq = x_seq + self.drop_path(self.out_proj(cross_attn * act_res))
        x_seq = x_seq + self.cpe_mlp(x_seq.permute(0, 2, 1).view(B, C, H, W)).flatten(2).permute(0, 2, 1)
        mlp_out = self.mlp(self.norm_mlp(x_seq))
        x_seq = x_seq + self.drop_path(mlp_out)
        out = x_seq.permute(0, 2, 1).view(B, C, H, W)
        return out

class SoftTopKSelector(nn.Module):

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        spatial_score = x_flat.mean(dim=1)
        N = spatial_score.size(1)
        k = int(0.5 * N)
        if k == 0:
            k = 1
        topk_vals = spatial_score.topk(k, dim=1).values
        threshold = topk_vals[:, -1].unsqueeze(1)
        mask = (spatial_score >= threshold).float().unsqueeze(1)
        topk_feat = x_flat * mask - 0.0001 * (1 - mask)
        remain_feat = x_flat * (1 - mask) - 0.0001 * mask
        return (topk_feat.view(B, C, H, W), remain_feat.view(B, C, H, W))

class Attention(nn.Module):

    def __init__(self, dim, num_heads=4):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f'dim {dim} should be divisible by num_heads {num_heads}'
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.block1 = nn.Sequential(nn.Linear(self.head_dim, self.head_dim // 2), nn.ReLU(), nn.Linear(self.head_dim // 2, self.head_dim), nn.Sigmoid())
        self.block2 = nn.Sequential(nn.Linear(self.head_dim, self.head_dim // 2), nn.ReLU(), nn.Linear(self.head_dim // 2, self.head_dim))
        self.topk = SoftTopKSelector()

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.permute(0, 2, 3, 1).view(B, -1, self.dim)
        x_heads = x_flat.view(B, -1, self.num_heads, self.head_dim)
        x_proc = x_heads.permute(0, 2, 3, 1).reshape(B * self.num_heads, self.head_dim, -1)
        x_proc_img = x_proc.view(B * self.num_heads, self.head_dim, H, W)
        topk_map, remain_map = self.topk(x_proc_img)
        curr_B, curr_C, _, _ = topk_map.shape
        topk_in = topk_map.view(curr_B, curr_C, -1).permute(0, 2, 1)
        remain_in = remain_map.view(curr_B, curr_C, -1).permute(0, 2, 1)
        topk_attn = self.block1(topk_in)
        remain_logits = self.block2(remain_in)
        remain_attn = F.softmax(remain_logits, dim=2)
        x_proc_flat = x_proc.permute(0, 2, 1)
        out_proc = x_proc_flat * (topk_attn + remain_attn)
        out = out_proc.view(B, self.num_heads, H * W, self.head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(B, H * W, self.dim)
        out = out.permute(0, 2, 1).view(B, self.dim, H, W)
        return out

class Decoder(nn.Module):

    def __init__(self, num_classes=8, channels=[64, 128, 256, 512], resolutions=[(128, 128), (64, 64), (32, 32), (16, 16)], drop_rate=0.0, drop_path_rate=0.2, model_use='all', fuselist=None):
        super().__init__()
        assert fuselist is not None and len(fuselist) >= 2
        C0, C1, C2, C3 = channels
        R0, R1, R2, R3 = resolutions
        self.model_use = model_use
        self.resolutions = resolutions
        self.block1 = DecodeBlock(C3, input_resolution=R3, num_heads=16, drop_ratio=drop_rate, drop_path_ratio=drop_path_rate)
        self.upsample_stage_1 = nn.Sequential(nn.Conv2d(C3, C2, 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.skip_block2 = nn.Sequential(Attention(dim=C3, num_heads=8), nn.BatchNorm2d(C3), nn.ReLU(), nn.Conv2d(C3, C2, 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.skip_conv2 = nn.Sequential(nn.Conv2d(2 * C2, C2, 1))
        self.block2 = DecodeBlock(C2, input_resolution=R2, num_heads=8, drop_ratio=drop_rate, drop_path_ratio=drop_path_rate)
        self.upsample_stage_2 = nn.Sequential(nn.Conv2d(C2, C1, 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.fuse_stage_3 = fuselist[0]
        self.skip_block3 = nn.Sequential(Attention(dim=C2, num_heads=4), nn.BatchNorm2d(C2), nn.ReLU(), nn.Conv2d(C2, C1, 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.skip_conv3 = nn.Sequential(nn.Conv2d(2 * C1, C1, 1))
        self.block3 = DecodeBlock(C1, input_resolution=R1, num_heads=4, drop_ratio=drop_rate, drop_path_ratio=drop_path_rate)
        self.upsample_stage_3 = nn.Sequential(nn.Conv2d(C1, C0, 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.fuse_stage_4 = fuselist[1]
        self.skip_block4 = nn.Sequential(Attention(dim=C1, num_heads=2), nn.BatchNorm2d(C1), nn.ReLU(), nn.Conv2d(C1, C0, 1), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        self.skip_conv4 = nn.Sequential(nn.Conv2d(2 * C0, C0, 1))
        self.block4 = DecodeBlock(C0, input_resolution=R0, num_heads=2, drop_ratio=drop_rate, drop_path_ratio=drop_path_rate)
        self.compress_stage2 = nn.Conv2d(C2, C0, 1)
        self.compress_stage3 = nn.Conv2d(C1, C0, 1)
        self.compress_stage4 = nn.Conv2d(C0, C0, 1)
        self.align_stage2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.align_stage3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.align_stage4 = nn.Identity()
        self.connect_conv = nn.Sequential(Attention(dim=C0 * 4, num_heads=8), nn.BatchNorm2d(C0 * 4), nn.ReLU(), nn.Conv2d(C0 * 4, C0, 1))
        self.classification_head = nn.Sequential(nn.Conv2d(C0, num_classes, 1))
        self.final_upsample = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
        self.drop_path = DropPath(drop_path_rate)
        self.decoder_feature = []
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

    def _clear_decoder_feature(self):
        self.decoder_feature = []

    def get_decoder_feature(self):
        return self.decoder_feature

    def forward(self, rgb_feat, x_feat, fuse_feat):
        self._clear_decoder_feature()
        RGB_features = rgb_feat
        X_features = x_feat
        FUSE_features = fuse_feat
        B = FUSE_features[0].shape[0]
        layer_1_out = self.block1(FUSE_features[1], FUSE_features[1])
        self.decoder_feature.append(layer_1_out)
        stage_2_input = self.upsample_stage_1(layer_1_out)
        stage_2_skip = self.skip_conv2(torch.cat([self.skip_block2(FUSE_features[1]), FUSE_features[0]], dim=1))
        layer_2_out = self.block2(stage_2_input, stage_2_skip)
        self.decoder_feature.append(layer_2_out)
        stage_3_input = self.upsample_stage_2(layer_2_out)
        if self.model_use == 'f1':
            stage_3_fuse = RGB_features[1]
        elif self.model_use == 'f2':
            stage_3_fuse = X_features[1]
        else:
            f1 = RGB_features[1].flatten(2).permute(0, 2, 1)
            f2 = X_features[1].flatten(2).permute(0, 2, 1)
            stage_3_fuse = self.fuse_stage_3(f1, f2).permute(0, 2, 1).view(B, -1, self.resolutions[1][0], self.resolutions[1][1])
        stage_3_skip = self.skip_conv3(torch.cat([self.skip_block3(FUSE_features[0]), stage_3_fuse], dim=1))
        layer_3_out = self.block3(stage_3_input, stage_3_skip)
        self.decoder_feature.append(layer_3_out)
        stage_4_input = self.upsample_stage_3(layer_3_out)
        if self.model_use == 'f1':
            stage_4_fuse = RGB_features[0]
        elif self.model_use == 'f2':
            stage_4_fuse = X_features[0]
        else:
            f1 = RGB_features[0].flatten(2).permute(0, 2, 1)
            f2 = X_features[0].flatten(2).permute(0, 2, 1)
            stage_4_fuse = self.fuse_stage_4(f1, f2).permute(0, 2, 1).view(B, -1, self.resolutions[0][0], self.resolutions[0][1])
        stage_4_skip = self.skip_conv4(torch.cat([self.skip_block4(stage_3_fuse), stage_4_fuse], dim=1))
        layer_4_out = self.block4(stage_4_input, stage_4_skip)
        self.decoder_feature.append(layer_4_out)
        stage_2_skip_c = self.compress_stage2(stage_2_skip)
        stage_3_skip_c = self.compress_stage3(stage_3_skip)
        stage_4_skip_c = self.compress_stage4(stage_4_skip)
        stage_2_skip_aligned = self.align_stage2(stage_2_skip_c)
        stage_3_skip_aligned = self.align_stage3(stage_3_skip_c)
        stage_4_skip_aligned = self.align_stage4(stage_4_skip_c)
        fuse = layer_4_out + self.drop_path(self.connect_conv(torch.cat([stage_2_skip_aligned, stage_3_skip_aligned, stage_4_skip_aligned, layer_4_out], dim=1)))
        out = self.classification_head(fuse)
        out = self.final_upsample(out)
        return out
