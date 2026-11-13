import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_
from collections import OrderedDict


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
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


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        dropout=0,
        norm=nn.BatchNorm2d,
        act_func=nn.ReLU,
    ):
        super(ConvLayer, self).__init__()
        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=(padding, padding),
            dilation=(dilation, dilation),
            groups=groups,
            bias=bias,
        )
        self.norm = norm(num_features=out_channels) if norm else None
        self.act = act_func() if act_func else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class RoPE(torch.nn.Module):
    r"""Rotary Positional Embedding."""

    def __init__(self, shape, base=10000):
        super(RoPE, self).__init__()

        channel_dims, feature_dim = shape[:-1], shape[-1]
        k_max = feature_dim // (2 * len(channel_dims))

        assert feature_dim % k_max == 0

        # angles
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

        # rotation
        rotations_re = torch.cos(angles).unsqueeze(dim=-1)
        rotations_im = torch.sin(angles).unsqueeze(dim=-1)
        rotations = torch.cat([rotations_re, rotations_im], dim=-1)
        self.register_buffer("rotations", rotations)

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        x = torch.view_as_complex(x.reshape(*x.shape[:-1], -1, 2))
        pe_x = torch.view_as_complex(self.rotations) * x
        return torch.view_as_real(pe_x).flatten(-2)


class LinearAttention(nn.Module):
    r"""Linear Attention with LePE and RoPE.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_heads, qkv_bias=True, **kwargs):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.qk = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.elu = nn.ELU()
        self.lepe = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.rope = RoPE(shape=(input_resolution[0], input_resolution[1], dim))

    def forward(self, x):
        """
        Args:
            x: input features with shape of (B, N, C)
        """
        b, n, c = x.shape
        h = int(n**0.5)
        w = int(n**0.5)
        num_heads = self.num_heads
        head_dim = c // num_heads

        qk = self.qk(x).reshape(b, n, 2, c).permute(2, 0, 1, 3)
        q, k, v = qk[0], qk[1], x
        # q, k, v: b, n, c

        q = self.elu(q) + 1.0
        k = self.elu(k) + 1.0
        q_rope = (
            self.rope(q.reshape(b, h, w, c))
            .reshape(b, n, num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        k_rope = (
            self.rope(k.reshape(b, h, w, c))
            .reshape(b, n, num_heads, head_dim)
            .permute(0, 2, 1, 3)
        )
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k_rope.transpose(-2, -1) * (n**-0.5)) @ (v * (n**-0.5))
        x = q_rope @ kv * z

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, h, w, c).permute(0, 3, 1, 2)
        x = x + self.lepe(v).permute(0, 2, 3, 1).reshape(b, n, c)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, num_heads={self.num_heads}"


class MLLABlock(nn.Module):
    r"""MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.in_proj = nn.Linear(dim, dim)
        self.act_proj = nn.Linear(dim, dim)
        self.dwc = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.attn = LinearAttention(
            dim=dim,
            input_resolution=input_resolution,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.out_proj = nn.Linear(dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x + self.cpe1(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(
            0, 2, 1
        )
        shortcut = x

        x = self.norm1(x)
        act_res = self.act(self.act_proj(x))
        x = self.in_proj(x).view(B, H, W, C)
        x = self.act(self.dwc(x.permute(0, 3, 1, 2))).permute(0, 2, 3, 1).view(B, L, C)

        # Linear Attention
        x = self.attn(x)

        x = self.out_proj(x * act_res)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(
            0, 2, 1
        )

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"mlp_ratio={self.mlp_ratio}"
        )


class PatchMerging(nn.Module):
    r"""Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
    """

    def __init__(self, input_resolution, dim, ratio=4.0):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        in_channels = dim
        out_channels = 2 * dim
        self.conv = nn.Sequential(
            ConvLayer(in_channels, int(out_channels * ratio), kernel_size=1, norm=None),
            ConvLayer(
                int(out_channels * ratio),
                int(out_channels * ratio),
                kernel_size=3,
                stride=2,
                padding=1,
                groups=int(out_channels * ratio),
                norm=None,
            ),
            ConvLayer(
                int(out_channels * ratio), out_channels, kernel_size=1, act_func=None
            ),
        )

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."
        x = (
            self.conv(x.reshape(B, H, W, C).permute(0, 3, 1, 2))
            .flatten(2)
            .permute(0, 2, 1)
        )
        return x


class BasicLayer(nn.Module):
    """A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                MLLABlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:

            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        feature = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, feature

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class Stem(nn.Module):
    r"""Stem

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.conv1 = ConvLayer(
            in_chans, embed_dim // 2, kernel_size=3, stride=2, padding=1, bias=False
        )
        self.conv2 = nn.Sequential(
            ConvLayer(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            ConvLayer(
                embed_dim // 2,
                embed_dim // 2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                act_func=None,
            ),
        )
        self.conv3 = nn.Sequential(
            ConvLayer(
                embed_dim // 2,
                embed_dim * 4,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            ConvLayer(
                embed_dim * 4, embed_dim, kernel_size=1, bias=False, act_func=None
            ),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.conv1(x)
        x = self.conv2(x) + x
        x = self.conv3(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Encoder(nn.Module):
    r"""MLLA
        A PyTorch impl of : `Demystify Mamba in Vision: A Linear Attention Perspective`
    """

    def __init__(
        self,
        img_size=224,
        patch_size=4,
        in_chans=3,
        embed_dim=64,
        depths=[2, 4, 8, 4],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        use_checkpoint=False,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        self.patch_embed = Stem(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Standalone encoder schedule; Model replaces it with the global depth schedule.
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    def _reshape_to_4d(self, x):
        B, N, C = x.shape

        H = int(N**0.5)
        while N % H != 0:
            H -= 1
        W = N // H

        return x.contiguous().view(B, H, W, C).permute(0, 3, 1, 2)

    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        stage_outputs = []
        for layer in self.layers:
            x, feature = layer(x)
            stage_outputs.append(self._reshape_to_4d(feature))
        return stage_outputs

    def forward(self, x):
        stage_outputs = self.forward_features(x)
        return stage_outputs


class DualStreamNetwork(nn.Module):
    def __init__(
        self,
        pretrained_path,
        img_size=224,
        patch_size=4,
        full_in_chans=3,  # 固定完整分支为3通道
        light_in_chans=1,  # 轻量分支输入通道数
        embed_dim=64,
        drop_rate=0.2,
        drop_path_rate=0.1,
        ape=False,
        model_use="all",
        fuselayer=None,
        depths=(2, 4, 8, 4),
        num_heads=(2, 4, 8, 16),
    ):
        super().__init__()
        self.model_use = model_use
        self.full_in_chans = full_in_chans
        self.light_in_chans = light_in_chans
        self.patch_size = patch_size

        # 加载预训练权重
        # 初始化完整分支
        self.full_stream = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=full_in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            ape=ape,
            use_checkpoint=False,
        )
        if pretrained_path is not None:
            checkpoint_dict = torch.load(pretrained_path, map_location="cpu", weights_only=False)
            pretrained_dict = checkpoint_dict
            for key in ("state_dict_ema", "model", "state_dict"):
                if key in checkpoint_dict and checkpoint_dict[key] is not None:
                    pretrained_dict = checkpoint_dict[key]
                    break
            pretrained_dict = OrderedDict(
                (key.removeprefix("module."), value) for key, value in pretrained_dict.items()
            )
            self.full_stream = self._load_pretrained_weights(self.full_stream, pretrained_dict, num_stages=4)
            print(f"Loaded pretrained backbone: {pretrained_path}")
            del checkpoint_dict, pretrained_dict

        # 初始化轻量分支
        self.light_stream = Encoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=light_in_chans,
            embed_dim=embed_dim,
            depths=depths[:2],
            num_heads=num_heads[:2],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=nn.LayerNorm,
            ape=ape,
            use_checkpoint=False,
        )

        self._add_light_downsample()
        self.fuse = fuselayer

    def _add_light_downsample(self):
        """构建轻量分支的下采样层"""
        self.light_downsample = PatchMerging(
            input_resolution=self.light_stream.layers[1].input_resolution,
            dim=self.light_stream.layers[1].dim,
        )

    @staticmethod
    def _load_pretrained_weights(model, pretrained_dict, num_stages=4):
        """静态方法：加载预训练权重"""
        model_dict = model.state_dict()
        new_dict = OrderedDict()
        
        max_layer_idx = num_stages - 1

        for k, v in pretrained_dict.items():
            if k.startswith("head"): continue
            
            # 处理 RoPE
            if "rope.rotations" in k:
                parts = k.split(".")
                try:
                    layer_idx = int(parts[1])
                except (ValueError, IndexError): continue
                if layer_idx > max_layer_idx: continue
                
                H = model.patches_resolution[0] // (2**layer_idx)
                W = model.patches_resolution[1] // (2**layer_idx)
                if v.shape[0] == H and v.shape[1] == W:
                    new_dict[k] = v
                continue

            # 处理 Layers
            if k.startswith("layers."):
                parts = k.split(".")
                try:
                    layer_idx = int(parts[1])
                except (ValueError, IndexError): continue
                if layer_idx > max_layer_idx: continue

            if k in model_dict and model_dict[k].shape == v.shape:
                new_dict[k] = v

        missing = [name for name, _ in model.named_parameters() if name not in new_dict]
        if missing:
            raise ValueError(
                "Pretrained checkpoint does not match the configured backbone; "
                f"{len(missing)} parameters missing or shape-mismatched: {missing[:8]}"
            )
        # RoPE buffers are rebuilt for the segmentation resolution.
        model.load_state_dict(new_dict, strict=False)
        return model

    def forward(self, x1, x2):
        full_front_features = []
        full_downsampled = None
        
        # --- Full Stream Forward ---
        if self.model_use in ["all", "f1", "fuse"]:
            x_full = self.full_stream.patch_embed(x1)
            if self.full_stream.ape:
                x_full = x_full + self.full_stream.absolute_pos_embed
            x_full = self.full_stream.pos_drop(x_full)

            x_full, full_feat0 = self.full_stream.layers[0](x_full)
            full_front_features.append(self.full_stream._reshape_to_4d(full_feat0))

            x_full, full_feat1 = self.full_stream.layers[1](x_full)
            full_front_features.append(self.full_stream._reshape_to_4d(full_feat1))

            full_downsampled = x_full 

        # --- Light Stream Forward ---
        light_features = []
        light_downsampled = None
        
        if self.model_use in ["all", "f2", "fuse"]:
            x_light = self.light_stream.patch_embed(x2)
            if self.light_stream.ape:
                x_light = x_light + self.light_stream.absolute_pos_embed
            x_light = self.light_stream.pos_drop(x_light)

            x_light, light_feat0 = self.light_stream.layers[0](x_light)
            light_features.append(self.light_stream._reshape_to_4d(light_feat0))

            x_light, light_feat1 = self.light_stream.layers[1](x_light)
            light_features.append(self.light_stream._reshape_to_4d(light_feat1))

            light_downsampled = self.light_downsample(x_light)

        # --- Fusion / Selection ---
        if self.model_use == "f1":
            fused_feat = full_downsampled
        elif self.model_use == "f2":
            fused_feat = light_downsampled
        else:
            fused_feat = self.fuse(full_downsampled, light_downsampled)

        # --- Deep Layers Forward ---
        full_back_features = []
        
        x_full, full_feat2 = self.full_stream.layers[2](fused_feat)
        full_back_features.append(self.full_stream._reshape_to_4d(full_feat2))

        x_full, full_feat3 = self.full_stream.layers[3](x_full)
        full_back_features.append(self.full_stream._reshape_to_4d(full_feat3))

        return full_front_features, light_features, full_back_features
