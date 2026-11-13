import math
from numbers import Real

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
# 注意：以下引用需要确保当前目录存在对应的 Encode.py, Decode.py, Fuse.py 文件
from .Encode import DualStreamNetwork
from .Decode import Decoder
from .Fuse import FusionMLLABlock

class Model(nn.Module):
    BACKBONES = {
        "MLLA-T": (64, (2, 4, 8, 4), (2, 4, 8, 16)),
        "MLLA-S": (64, (3, 6, 21, 6), (2, 4, 8, 16)),
        "MLLA-B": (96, (3, 6, 21, 6), (3, 6, 12, 24)),
    }

    def __init__(
        self,
        num_classes=6,
        full_in_chans=3,
        light_in_chans=1,
        embed_dim=None,
        model_use="all",
        backbone="MLLA-T",
        pretrained_path=None,
        load_pretrained=True,
        drop_rate=0.0,
        drop_path_rate=0.2,
    ):
        super().__init__()
        for name, value in (("drop_rate", drop_rate), ("drop_path_rate", drop_path_rate)):
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not 0 <= value < 1):
                raise ValueError(f"model.{name} must be a finite number in [0, 1)")
        self.drop_rate = float(drop_rate)
        self.drop_path_rate = float(drop_path_rate)
        backbone = backbone.upper().replace("MILA-", "MLLA-")
        if backbone not in self.BACKBONES:
            raise ValueError(f"Unknown backbone {backbone}; choose from {tuple(self.BACKBONES)}")
        preset_dim, depths, num_heads = self.BACKBONES[backbone]
        if embed_dim is not None and embed_dim != preset_dim:
            raise ValueError(f"{backbone} requires embed_dim={preset_dim}, got {embed_dim}")
        embed_dim = preset_dim
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.feature_channels = [embed_dim * factor for factor in (4, 8, 8, 4, 2, 1)]
        self.feature_heads = [num_heads[index] for index in (2, 3, 3, 2, 1, 0)]
        self._phase3_freeze_shallow = False
        self.model_use = model_use
        self._cache = {
            "encoder_back": None,  # 编码器深层特征
            "decoder_all": None    # 解码器所有层级输出
        }   
        self.fuse = nn.ModuleList(
            [
                # 第一层级：1/16分辨率
                FusionMLLABlock(
                    dim=embed_dim * 4,
                    input_resolution=(512 // 16, 512 // 16),
                    num_heads=num_heads[2],
                    drop=self.drop_rate,
                    drop_path=self.drop_path_rate,
                ),
                # 第二层级：1/8分辨率
                FusionMLLABlock(
                    dim=embed_dim * 2,
                    input_resolution=(512 // 8, 512 // 8),
                    num_heads=num_heads[1],
                    drop=self.drop_rate,
                    drop_path=self.drop_path_rate,
                ),
                # 第三层级：1/4分辨率
                FusionMLLABlock(
                    dim=embed_dim,
                    input_resolution=(512 // 4, 512 // 4),
                    num_heads=num_heads[0],
                    drop=self.drop_rate,
                    drop_path=self.drop_path_rate,
                ),
            ]
        )
        # 初始化双流编码器
        self.encode = DualStreamNetwork(
            pretrained_path=(pretrained_path or f"./pretrain/{backbone}.pth") if load_pretrained else None,
            img_size=512,
            full_in_chans=full_in_chans,
            light_in_chans=light_in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            model_use=model_use,
            fuselayer=self.fuse[0],
        )

        # 初始化解码器
        self.decode = Decoder(
            num_classes=num_classes,
            channels=[embed_dim * (2 ** index) for index in range(4)],
            drop_rate=self.drop_rate,
            drop_path_rate=self.drop_path_rate,
            model_use=model_use,
            fuselist=self.fuse[1:],
        )
        self._configure_global_drop_path()

    def _configure_global_drop_path(self):
        """Assign one fixed depth schedule across encoder, SAMT and decoder.

        Parallel front streams share depth positions. All three SAMT blocks
        retain their positions even when bypassed in f1/f2 mode. The final
        decoder fusion residual occupies the last position.
        """
        positions = []
        full_layers = self.encode.full_stream.layers
        light_layers = self.encode.light_stream.layers
        for stage, (full_layer, light_layer) in enumerate(zip(full_layers[:2], light_layers)):
            if len(full_layer.blocks) != len(light_layer.blocks):
                raise ValueError("Parallel encoder stages must have matching depths")
            for block, (full, light) in enumerate(zip(full_layer.blocks, light_layer.blocks)):
                positions.append((
                    (f"encode.full_stream.layers.{stage}.blocks.{block}", full),
                    (f"encode.light_stream.layers.{stage}.blocks.{block}", light),
                ))
        positions.append((("fuse.0", self.fuse[0]),))
        for stage, layer in enumerate(full_layers[2:], start=2):
            for block, module in enumerate(layer.blocks):
                positions.append(((f"encode.full_stream.layers.{stage}.blocks.{block}", module),))
        positions.extend([
            (("decode.block1", self.decode.block1),),
            (("decode.block2", self.decode.block2),),
            (("fuse.1", self.fuse[1]),),
            (("decode.block3", self.decode.block3),),
            (("fuse.2", self.fuse[2]),),
            (("decode.block4", self.decode.block4),),
            (("decode", self.decode),),
        ])
        rates = tuple(self.drop_path_rate * (index / (len(positions) - 1))
                      for index in range(len(positions)))
        schedule = []
        for depth, (rate, modules) in enumerate(zip(rates, positions)):
            for name, module in modules:
                module.drop_path = DropPath(rate) if rate > 0.0 else nn.Identity()
            schedule.append({"depth": depth, "probability": rate,
                             "modules": tuple(name for name, _ in modules)})
        self.drop_path_schedule = tuple(schedule)
        self.decode.drop_path_rates = tuple(
            getattr(getattr(self.decode, f"block{index}").drop_path, "drop_prob", 0.0)
            for index in range(1, 5)
        )

    @classmethod
    def from_config(cls, config, model_use="all", load_pretrained=True):
        return cls(
            num_classes=config.num_classes,
            light_in_chans=config.x_num,
            model_use=model_use,
            backbone=getattr(config, "backbone", "MLLA-T"),
            pretrained_path=getattr(config, "pretrained_path", None),
            load_pretrained=load_pretrained,
            drop_rate=getattr(config, "drop_rate", 0.0),
            drop_path_rate=getattr(config, "drop_path_rate", 0.2),
        )

    def train(self, mode=True):
        super().train(mode)
        if mode and self._phase3_freeze_shallow:
            self.bn_eval_phase3()
        return self

    # ----------------------------------------------------------------
    #  基础工具方法
    # ----------------------------------------------------------------

    def _clear_cache(self):
        """清空缓存以释放显存"""
        self._cache["encoder_back"] = None
        self._cache["decoder_all"] = None

    def _clone_norm_attributes(self, src_norm, dst_norm):
        """复制 BN/LN 的权重和统计量"""
        if src_norm.weight is not None and dst_norm.weight is not None:
             dst_norm.weight.data = src_norm.weight.data.clone()
        if src_norm.bias is not None and dst_norm.bias is not None:
             dst_norm.bias.data = src_norm.bias.data.clone()
        
        if hasattr(src_norm, 'running_mean') and hasattr(dst_norm, 'running_mean'):
            dst_norm.running_mean.data = src_norm.running_mean.data.clone()
        if hasattr(src_norm, 'running_var') and hasattr(dst_norm, 'running_var'):
            dst_norm.running_var.data = src_norm.running_var.data.clone()
        if hasattr(src_norm, 'num_batches_tracked') and hasattr(dst_norm, 'num_batches_tracked'):
             dst_norm.num_batches_tracked.data = src_norm.num_batches_tracked.data.clone()

    def _clone_full_weights_to_light_internal(self):
        """执行具体的权重克隆逻辑"""
        full_stream = self.encode.full_stream
        light_stream = self.encode.light_stream
        
        # 1. Clone Stem
        full_conv1 = full_stream.patch_embed.conv1.conv
        light_conv1 = light_stream.patch_embed.conv1.conv
        
        if self.encode.light_in_chans == 3:
            light_conv1.weight.data = full_conv1.weight.data.clone()
        elif self.encode.light_in_chans == 1:
            light_conv1.weight.data = full_conv1.weight.data.mean(dim=1, keepdim=True)
        else:
            # 自定义通道处理
            avg_weight = full_conv1.weight.data.mean(dim=1, keepdim=True)
            new_weight = avg_weight.repeat(1, self.encode.light_in_chans, 1, 1)
            light_conv1.weight.data = new_weight * (3 / self.encode.light_in_chans)

        # Clone Stem Biases and Norms
        if full_conv1.bias is not None and light_conv1.bias is not None:
            light_conv1.bias.data = full_conv1.bias.data.clone()
        
        # Stem Conv2/3 Clone Loop
        full_stem = full_stream.patch_embed
        light_stem = light_stream.patch_embed
        
        for full_c, light_c in zip(full_stem.conv2, light_stem.conv2):
            light_c.conv.weight.data = full_c.conv.weight.data.clone()
            if full_c.conv.bias is not None: light_c.conv.bias.data = full_c.conv.bias.data.clone()
            if full_c.norm: self._clone_norm_attributes(full_c.norm, light_c.norm)

        for full_c, light_c in zip(full_stem.conv3, light_stem.conv3):
            light_c.conv.weight.data = full_c.conv.weight.data.clone()
            if full_c.conv.bias is not None: light_c.conv.bias.data = full_c.conv.bias.data.clone()
            if full_c.norm: self._clone_norm_attributes(full_c.norm, light_c.norm)

        # 2. Clone Layers (0 and 1)
        for light_layer, full_layer in zip(light_stream.layers, full_stream.layers[:2]):
            for light_blk, full_blk in zip(light_layer.blocks, full_layer.blocks):
                # Copy MLLA block params
                light_blk.cpe1.weight.data = full_blk.cpe1.weight.data.clone()
                light_blk.cpe2.weight.data = full_blk.cpe2.weight.data.clone()
                if full_blk.cpe1.bias is not None: light_blk.cpe1.bias.data = full_blk.cpe1.bias.data.clone()
                if full_blk.cpe2.bias is not None: light_blk.cpe2.bias.data = full_blk.cpe2.bias.data.clone()
                
                self._clone_norm_attributes(full_blk.norm1, light_blk.norm1)
                self._clone_norm_attributes(full_blk.norm2, light_blk.norm2)
                
                light_blk.in_proj.weight.data = full_blk.in_proj.weight.data.clone()
                light_blk.act_proj.weight.data = full_blk.act_proj.weight.data.clone()
                light_blk.dwc.weight.data = full_blk.dwc.weight.data.clone()
                light_blk.out_proj.weight.data = full_blk.out_proj.weight.data.clone()
                
                if full_blk.in_proj.bias is not None: light_blk.in_proj.bias.data = full_blk.in_proj.bias.data.clone()
                if full_blk.act_proj.bias is not None: light_blk.act_proj.bias.data = full_blk.act_proj.bias.data.clone()
                if full_blk.dwc.bias is not None: light_blk.dwc.bias.data = full_blk.dwc.bias.data.clone()
                if full_blk.out_proj.bias is not None: light_blk.out_proj.bias.data = full_blk.out_proj.bias.data.clone()

                light_blk.attn.qk.weight.data = full_blk.attn.qk.weight.data.clone()
                if full_blk.attn.qk.bias is not None: light_blk.attn.qk.bias.data = full_blk.attn.qk.bias.data.clone()
                light_blk.attn.lepe.weight.data = full_blk.attn.lepe.weight.data.clone()
                if full_blk.attn.lepe.bias is not None: light_blk.attn.lepe.bias.data = full_blk.attn.lepe.bias.data.clone()

                light_blk.mlp.fc1.weight.data = full_blk.mlp.fc1.weight.data.clone()
                light_blk.mlp.fc2.weight.data = full_blk.mlp.fc2.weight.data.clone()
                if full_blk.mlp.fc1.bias is not None: light_blk.mlp.fc1.bias.data = full_blk.mlp.fc1.bias.data.clone()
                if full_blk.mlp.fc2.bias is not None: light_blk.mlp.fc2.bias.data = full_blk.mlp.fc2.bias.data.clone()

        # 3. Clone Downsample
        full_down = full_stream.layers[1].downsample
        light_down = self.encode.light_downsample
        if full_down is not None and light_down is not None:
             for l_conv, f_conv in zip(light_down.conv, full_down.conv):
                l_conv.conv.weight.data = f_conv.conv.weight.data.clone()
                if f_conv.conv.bias is not None: l_conv.conv.bias.data = f_conv.conv.bias.data.clone()
                if f_conv.norm: self._clone_norm_attributes(f_conv.norm, l_conv.norm)
        
        print("成功将主分支权重克隆到轻量分支（包括下采样模块）")

    # ----------------------------------------------------------------
    #  Phase 1: 切换至 f1 模式
    #  目标：使用Full分支训练。冻结Light分支、Light下采样、Fuse。
    # ----------------------------------------------------------------

    def switch_to_f1(self):
        """[Phase 1-1] 切换模式并配置参数冻结状态"""
        self.model_use = "f1"
        self.encode.model_use = "f1"
        self.decode.model_use = "f1"
        self._set_phase1_freeze()

    def _set_phase1_freeze(self):
        """[Phase 1-2] 冻结 Light, Downsample, Fuse; 解冻 Full (All)"""
        # 1. 解冻所有 Full 参数
        for param in self.encode.full_stream.parameters():
            param.requires_grad = True
        
        # 2. 冻结 Light 分支
        for param in self.encode.light_stream.parameters():
            param.requires_grad = False
        
        # 3. 冻结 Light Downsample
        for param in self.encode.light_downsample.parameters():
            param.requires_grad = False
            
        # 4. [调整顺序] 先解冻 Decoder
        # 注意：这步操作会副作用解冻 self.decode 内部持有的 fuse 模块
        for param in self.decode.parameters():
            param.requires_grad = True

        # 5. [CRITICAL FIX] 最后再次强制冻结 Fusion
        # 必须放在 Decode 解冻之后，因为 Decode 包含了 Fuse 引用
        for param in self.fuse.parameters():
            param.requires_grad = False
            
        # 双重保险：检查 Encode 中的 Fuse 引用
        if hasattr(self.encode, 'fuse') and self.encode.fuse is not None:
            for param in self.encode.fuse.parameters():
                param.requires_grad = False

    # ----------------------------------------------------------------
    #  Phase 2: 切换至 f2 模式
    # ----------------------------------------------------------------

    def switch_to_f2(self):
        """[Phase 2-1] 切换到 f2 模式"""
        self.model_use = "f2"
        self.encode.model_use = "f2"
        self.decode.model_use = "f2"
        

    def init_light_weights(self):
        """[Phase 2-2] 执行权重克隆 (Full -> Light)"""
        self._clone_full_weights_to_light_internal()

    def phase2_warmup(self):
        """[Phase 2-3] 预热"""
        # 1. 全局冻结
        for param in self.parameters():
            param.requires_grad = False

        # 2. 判断 Conv1 维度
        if self.encode.light_in_chans != 3:
            for param in self.encode.light_stream.patch_embed.parameters():
                param.requires_grad = True

        # 3. 解冻 BN
        for m in self.encode.light_stream.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                for p in m.parameters(): p.requires_grad = True
        
        for m in self.encode.light_downsample.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                for p in m.parameters(): p.requires_grad = True

        for layer in self.encode.full_stream.layers[2:]:
            for m in layer.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    for p in m.parameters(): p.requires_grad = True

        for param in self.fuse.parameters():
            param.requires_grad = False
    
    def phase2_main(self):
        """[Phase 2-4] 主训练"""
        # 1. 冻结 Full Front
        self.encode.full_stream.patch_embed.requires_grad_(False)
        self.encode.full_stream.layers[0].requires_grad_(False)
        self.encode.full_stream.layers[1].requires_grad_(False) 

        # 2. 解冻 Light Stream
        for param in self.encode.light_stream.parameters():
            param.requires_grad = True
        
        # 3. 解冻 Light Downsample
        for param in self.encode.light_downsample.parameters():
            param.requires_grad = True
            
        # 4. 解冻 Full Back
        for layer in self.encode.full_stream.layers[2:]:
             for param in layer.parameters():
                param.requires_grad = True
        
        for param in self.fuse.parameters():
            param.requires_grad = False

    def bn_eval_phase2(self):
        """[Phase 2-Extra]"""
        self.encode.full_stream.patch_embed.eval()
        self.encode.full_stream.layers[0].eval()
        self.encode.full_stream.layers[1].eval()

    # ----------------------------------------------------------------
    #  Phase 3: 切换至 All 模式
    # ----------------------------------------------------------------

    def switch_to_all(self):
        """[Phase 3-1] 切换到 all 模式"""
        self.model_use = "all"
        self.encode.model_use = "all"
        self.decode.model_use = "all"

    def phase3_warmup(self):
        """[Phase 3-2] 预热"""
        # 1. 全局冻结
        for param in self.parameters():
            param.requires_grad = False
            
        # 2. 解冻 Fuse
        for param in self.fuse.parameters():
            param.requires_grad = True
        if hasattr(self.encode, 'fuse') and self.encode.fuse is not None:
             for param in self.encode.fuse.parameters():
                param.requires_grad = True
            
        # 3. 解冻 Deep Stages BN
        for layer in self.encode.full_stream.layers[2:]:
             for m in layer.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    for p in m.parameters(): p.requires_grad = True

    def phase3_main(self, freeze_shallow=True):
        """[Phase 3-3] 主训练"""
        # 1. 冻结 Full Front
        self._phase3_freeze_shallow = bool(freeze_shallow)
        self.encode.full_stream.patch_embed.requires_grad_(not freeze_shallow)
        self.encode.full_stream.layers[0].requires_grad_(not freeze_shallow)
        self.encode.full_stream.layers[1].requires_grad_(not freeze_shallow)

        # 2. 冻结 Light Stream & Downsample
        self.encode.light_stream.requires_grad_(not freeze_shallow)
        self.encode.light_downsample.requires_grad_(not freeze_shallow)

        # 3. 解冻 Fuse
        for param in self.fuse.parameters():
            param.requires_grad = True
        if hasattr(self.encode, 'fuse') and self.encode.fuse is not None:
             for param in self.encode.fuse.parameters():
                param.requires_grad = True

        # 4. 解冻 Full Back
        for layer in self.encode.full_stream.layers[2:]:
            for param in layer.parameters():
                param.requires_grad = True

        # 5. 解冻 Decoder
        for param in self.decode.parameters():
            param.requires_grad = True

        if self.training and self._phase3_freeze_shallow:
            self.bn_eval_phase3()

    def bn_eval_phase3(self):
        """[Phase 3-Extra]"""
        self.encode.full_stream.patch_embed.eval()
        self.encode.full_stream.layers[0].eval()
        self.encode.full_stream.layers[1].eval()
        
        self.encode.light_stream.eval()
        self.encode.light_downsample.eval()

    # ----------------------------------------------------------------
    #  Forward & Interfaces
    # ----------------------------------------------------------------

    def forward(self, rgb, dsm):
        self._clear_cache()
        rgb_features, dsm_features, last_features = self.encode(rgb, dsm)
        output = self.decode(rgb_features, dsm_features, last_features)
        
        self._cache["encoder_back"] = last_features
        self._cache["decoder_all"] = self.decode.get_decoder_feature()
        return output

    def get_distill_features(self):
        return self._cache["decoder_all"]

    def get_align_features(self):
        return self._cache["encoder_back"] + self._cache["decoder_all"]

if __name__ == "__main__":
    print("Initializing Model...")
    try:
        model = Model(
            num_classes=6,
            full_in_chans=3,
            light_in_chans=1,
            embed_dim=64,
            model_use="all"
        )
        
        # --- 新增：计算并打印模型总参数量 (单位: M) ---
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model Total Parameters: {total_params / 1e6:.2f} M")
        # ---------------------------------------------

        rgb_input = torch.randn(2, 3, 512, 512)
        dsm_input = torch.randn(2, 1, 512, 512)
        
        if torch.cuda.is_available():
            model = model.cuda()
            rgb_input = rgb_input.cuda()
            dsm_input = dsm_input.cuda()
            print("Running on CUDA")
        
        model.train()
        outputs = model(rgb_input, dsm_input)
        print("Forward pass successful.")

        print("\nTesting method calls...")
        model.switch_to_f1()
        print("Switched to f1.")
        # 检查参数状态
        print(f"Fuse requires_grad: {next(model.fuse.parameters()).requires_grad} (Should be False)")
        
        model.switch_to_f2()
        print("Switched to f2.")
        
        model.switch_to_all()
        print("Switched back to all.")

    except ImportError as e:
        print(f"\n[Error] Import failed: {e}")
    except Exception as e:
        print(f"\n[Error] Runtime error: {e}")
        import traceback
        traceback.print_exc()
