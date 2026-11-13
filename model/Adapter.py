import math
from numbers import Real

import torch
import torch.nn as nn
from timm.layers import trunc_normal_,DropPath


class RoPE(nn.Module):
    r"""Rotary Positional Embedding."""

    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()
        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # 角度计算
        theta_ks = 1 / (base ** (torch.arange(k_max) / k_max))
        angles = torch.cat(
            [
                t.unsqueeze(-1) * theta_ks
                for t in torch.meshgrid(
                    [torch.arange(d) for d in channel_dims], indexing="ij"
                )
            ],
            dim=-1,
        )

        # 旋转矩阵
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer("rotations", rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.view(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class CrossLinearAttention(nn.Module):
    r"""Cross Modal Linear Attention with LePE and RoPE.

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

        # Separate projections for query, key and value
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)

        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))
        self.log_temp = nn.Parameter(torch.zeros(1))

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

        # Projections
        q = self.q(x_q)
        k = self.k(x_k)
        v = x_v

        temp = self.log_temp.exp()

        # Activation
        q = self.elu(q/temp) + 1.0
        k = self.elu(k) + 1.0

        # Apply RoPE
        q_rope = (
            self.rope(q.reshape(B, H, W, C))
            .reshape(B, N, num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        k_rope = (
            self.rope(k.reshape(B, H, W, C))
            .reshape(B, N, num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )

        # Reshape
        q = q.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, N, num_heads, head_dim).permute(0, 2, 1, 3)

        # Linear attention computation
        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (N**-0.5)) @ (v * (N**-0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(B, N, C)

        # LePE from key/value features
        v_spatial = v.transpose(1, 2).reshape(B, H, W, C).permute(0, 3, 1, 2)
        x = x + self.lepe(v_spatial).permute(0, 2, 3, 1).reshape(B, N, C)

        return x




class FeatureAdapter(nn.Module):
    def __init__(
        self,
        channels=[256, 512, 512, 256, 128, 64],
        num_head=[8, 16, 16, 8, 4, 2],
        input_resolution=[(32, 32), (16, 16), (16, 16), (32, 32), (64, 64), (128, 128)],
        init_values=1e-4, # [NEW] LayerScale 的初始值
        drop_path_rate=0.2,
    ):
        super().__init__()
        if (isinstance(drop_path_rate, bool) or not isinstance(drop_path_rate, Real)
                or not math.isfinite(drop_path_rate) or not 0 <= drop_path_rate < 1):
            raise ValueError("drop_path_rate must be a finite number in [0, 1)")
        self.adapters = nn.ModuleList(
            [
                CrossLinearAttention(
                    dim=channels[i],
                    input_resolution=input_resolution[i],
                    num_heads=num_head[i],
                )
                for i in range(len(channels))
            ]
        )
        self.bottneck = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(channels[i], channels[i] // 4),
                    nn.GELU(),
                    nn.Linear(channels[i] // 4, channels[i])
                )
                for i in range(len(channels))
            ]
        )
        self.norm1 = nn.ModuleList([nn.LayerNorm(channels[i]) for i in range(len(channels))])
        self.norm2 = nn.ModuleList([nn.LayerNorm(channels[i]) for i in range(len(channels))])
        self.norm3 = nn.ModuleList([nn.LayerNorm(channels[i]) for i in range(len(channels))])
        
        self.x1_proj = nn.ModuleList([nn.Linear(channels[i], channels[i]) for i in range(len(channels))])
        self.x2_proj = nn.ModuleList([nn.Linear(channels[i], channels[i]) for i in range(len(channels))])
        
        # [NEW] LayerScale Parameters
        # 移除原先的 x2_gate Linear 层，改为直接可学习的参数 gamma
        # 针对每个通道学习一个权重，形状为 (C,)
        self.layer_scale = nn.ParameterList([
            nn.Parameter(init_values * torch.ones(channels[i]), requires_grad=True) 
            for i in range(len(channels))
        ])

        self.droppath = DropPath(float(drop_path_rate))
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
            x1: list of feature maps (B, C, H, W) - Main Modality
            x2: list of feature maps (B, C, H, W) - Aux Modality
        """
        output_features = []
        for i, adapter in enumerate(self.adapters):
            # 1. Prepare Inputs
            # 这里的 detach 取决于训练策略，如果需要端到端微调请移除 .detach()
            x1_flat = x1[i].detach().flatten(2).transpose(1, 2) # (B, N, C)
            x2_flat = x2[i].detach().flatten(2).transpose(1, 2) # (B, N, C)
            
            # Pre-Norm
            x1_norm = self.norm1[i](x1_flat)
            x2_norm = self.norm2[i](x2_flat)
            
            # Projections
            x1_proj = self.x1_proj[i](x1_norm)
            x2_proj = self.x2_proj[i](x2_norm)
            
            # 2. Cross Attention Block with LayerScale
            # 计算 Cross Attention 更新量
            attn_update = adapter(x1_proj, x2_proj, x2_proj)
            
            # [NEW] 应用 LayerScale
            # gamma: (C) -> (1, 1, C) 用于广播乘法
            gamma = self.layer_scale[i].unsqueeze(0).unsqueeze(0)
            
            # Residual Connection: Y = X + DropPath(gamma * Attention(X))
            # 初始状态下 gamma 很小，attn 约等于 x1_flat (Identity)，训练极其稳定
            attn = x1_flat + self.droppath(gamma * attn_update)

            # 3. Bottleneck Block (FFN)
            # FFN 部分通常不加 LayerScale，或者如果加也是加在 FFN 内部，这里保持原样即可
            output = attn + self.droppath(self.bottneck[i](self.norm3[i](attn)))
            
            # Reshape back to (B, C, H, W)
            H, W = x1[i].shape[-2:]
            output = output.transpose(1, 2).reshape(x1[i].shape[0], x1[i].shape[1], H, W)
            output_features.append(output)
            
        return output_features


class TargetAdapter(nn.Module):
    def __init__(
        self, num_classes, channels=[512, 256, 128, 64], target_resolutions=(512, 512)
    ):
        super().__init__()
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels[i], num_classes, kernel_size=1),
                    nn.Upsample(
                        size=target_resolutions, mode="bilinear", align_corners=False
                    ),
                )
                for i in range(len(channels))
            ]
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x_main = x[0]
        x_aux = x[1:]
        for i, feature in enumerate(x_aux):
            x_aux[i] = self.adapters[i](feature)
        x = [x_main] + x_aux
        return x
    

class CBAM(nn.Module):
    """简单的通道和空间注意力模块"""
    def __init__(self, channels, reduction=16):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        x = x * self.sigmoid(out)
        
        # Spatial Attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_spatial = torch.cat([avg_out, max_out], dim=1)
        x_spatial = self.conv_spatial(x_spatial)
        x = x * self.sigmoid(x_spatial)
        return x

class BlockFusionAdapter(nn.Module):
    """
    用于 Phase 4: 
    接收 Teacher 3 某一 Stage 所有 Block 的输出 (List[Tensor])
    1. 堆叠 (Concat)
    2. 空间通道注意力 (CBAM)
    3. 降维 (Conv 1x1) 到 student block 的维度
    """
    def __init__(self, dim, depth, reduction=16):
        super().__init__()
        self.in_channels = dim * depth
        self.out_channels = dim
        
        # 1. 注意力处理 (在堆叠后的高维特征上做)
        self.attention = CBAM(self.in_channels, reduction=reduction)
        
        # 2. 降维映射
        self.project = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, 1, bias=False),
            nn.BatchNorm2d(self.out_channels), # 保持分布一致性
            nn.ReLU(inplace=True),
            nn.Conv2d(self.out_channels,self.out_channels,1)
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def forward(self, features_list):
        # features_list: [B, N, C] or [B, C, H, W]
        # MLLA 输出通常是 [B, N, C]，我们需要 reshape 回 [B, C, H, W] 进行卷积处理
        
        reshaped_feats = []
        for feat in features_list:
            if feat.dim() == 3:
                B, N, C = feat.shape
                H = int(N**0.5)
                W = N // H
                feat = feat.permute(0, 2, 1).reshape(B, C, H, W)
            reshaped_feats.append(feat)
            
        # 1. 堆叠
        x = torch.cat(reshaped_feats, dim=1) # [B, C*Depth, H, W]
        
        # 2. 注意力
        x = self.attention(x)
        
        # 3. 降维
        x = self.project(x)
        
        # 恢复到 Transformer 常用的 [B, N, C] 格式以便与 Student 对齐计算 MSE
        x = x.flatten(2).permute(0, 2, 1)
        return x
