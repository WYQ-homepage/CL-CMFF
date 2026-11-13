# Support both package imports and direct script execution.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reproducibility import make_data_generator, seed_augmentations, seed_worker
import os
import cv2
import torch
import random
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================= 1. 配置类 (Config) =================

@dataclass
class WHUAugmentConfig:
    """WHU 数据增强配置"""
    enable: bool = True
    
    # 1. 几何变换 (由 CustomAffine 和 Unified Transform 共同完成)
    p_geo: float = 0.5
    rotate_limit: tuple = (-45, 45)
    scale_limit: tuple = (0.8, 1.2)
    shift_limit: tuple = (-0.0625, 0.0625)
    
    # 2. RGB 专属增强
    p_rgb: float = 0.5
    p_hsv: float = 0.3
    h_shift: int = 20
    s_shift: int = 30
    v_shift: int = 20
    
    # 通用强度
    p_intensity: float = 0.3
    
    # 3. IR 专属增强
    p_ir: float = 0.5
    
    # 4. SAR 专属增强
    p_sar: float = 0.5
    sar_speckle_limit: tuple = (0.9, 1.1)

    # 填充配置 (逻辑移交 CustomAffine，此处保留作为参考或备用)
    mask_pad_value: int = 255 

@dataclass
class DatasetStats:
    """统计信息: RGB(3ch) + IR(1ch) + SAR(1ch)"""
    # 默认值
    rgb_mean: list = field(default_factory=lambda: [0.4, 0.4, 0.4])
    rgb_std: list = field(default_factory=lambda: [0.2, 0.2, 0.2])
    ir_mean: float = 0.4
    ir_std: float = 0.2
    sar_mean: float = 0.4
    sar_std: float = 0.2

# ================= 2. 统计计算工具 (Calculator) =================

def _worker_stat(file_info):
    """Worker: 分别计算 RGB, IR, SAR 的统计量"""
    rgb_path, ir_path, sar_path, mask_path = file_info

    # RGB
    img_bgr = cv2.imread(rgb_path)
    if img_bgr is None: return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # IR
    ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    if ir is None: return None
    # 假设数据范围兼容 (如 uint8 0-255)
    ir = ir.astype(np.float32) / 255.0 if ir.dtype == np.uint8 else ir.astype(np.float32)

    # SAR
    sar = cv2.imread(sar_path, cv2.IMREAD_UNCHANGED)
    if sar is None: return None
    sar = sar.astype(np.float32) / 255.0 if sar.dtype == np.uint8 else sar.astype(np.float32)
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    valid_mask = mask != 255 
    if not np.any(valid_mask): return None

    valid_rgb = rgb[valid_mask]
    valid_ir = ir[valid_mask]
    valid_sar = sar[valid_mask]

    return (
        np.sum(valid_rgb, axis=0),      # rgb sum
        np.sum(valid_rgb**2, axis=0),   # rgb sq_sum
        np.sum(valid_ir),               # ir sum
        np.sum(valid_ir**2),            # ir sq_sum
        np.sum(valid_sar),              # sar sum
        np.sum(valid_sar**2),           # sar sq_sum
        np.sum(valid_mask),             # count
    )

class StatisticsCalculator:
    def __init__(self, root_dir, split="train"):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.ir_dir = os.path.join(root_dir, split, "ir")
        self.sar_dir = os.path.join(root_dir, split, "sar")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.files = (sorted([f for f in os.listdir(self.img_dir) if f.endswith(".tif")])
                      if os.path.exists(self.img_dir) else [])

    def compute(self) -> DatasetStats:
        if not self.files: return DatasetStats()

        file_list = [
            (os.path.join(self.img_dir, f), os.path.join(self.ir_dir, f),
             os.path.join(self.sar_dir, f), os.path.join(self.mask_dir, f))
            for f in self.files
        ]

        num_workers = min(cpu_count(), 16, max(1, len(file_list)))
        print(f"Calculating stats with {num_workers} workers...")

        rgb_sum, rgb_sq_sum = np.zeros(3), np.zeros(3)
        ir_sum, ir_sq_sum = 0.0, 0.0
        sar_sum, sar_sq_sum = 0.0, 0.0
        count = 0

        with Pool(num_workers) as p:
            for res in tqdm(p.imap(_worker_stat, file_list), total=len(file_list)):
                if res:
                    r_s, r_sq, i_s, i_sq, s_s, s_sq, c = res
                    rgb_sum += r_s
                    rgb_sq_sum += r_sq
                    ir_sum += i_s
                    ir_sq_sum += i_sq
                    sar_sum += s_s
                    sar_sq_sum += s_sq
                    count += c

        if count == 0: raise ValueError("No valid pixels.")

        rgb_mean = rgb_sum / count
        rgb_std = np.sqrt((rgb_sq_sum / count) - (rgb_mean**2))
        ir_mean = ir_sum / count
        ir_std = np.sqrt((ir_sq_sum / count) - (ir_mean**2))
        sar_mean = sar_sum / count
        sar_std = np.sqrt((sar_sq_sum / count) - (sar_mean**2))

        print(f"Stats Calculated:\n RGB:{rgb_mean}\n IR:{ir_mean:.4f}\n SAR:{sar_mean:.4f}")

        return DatasetStats(
            rgb_mean=rgb_mean.tolist(),
            rgb_std=rgb_std.tolist(),
            ir_mean=float(ir_mean),
            ir_std=float(ir_std),
            sar_mean=float(sar_mean),
            sar_std=float(sar_std),
        )

# ================= 3. 自定义仿射变换 (Custom Affine) [核心新增] =================

class CustomAffine:
    """
    针对 WHU (RGB, IR, SAR) 的自定义仿射变换。
    完全替代 Albumentations 的 Affine，提供基于模态均值的填充策略。
    """
    def __init__(self, config: WHUAugmentConfig, stats: DatasetStats):
        self.config = config
        self.stats = stats

    def _get_affine_matrix(self, height, width):
        """生成共享的仿射矩阵"""
        center = (width / 2, height / 2)
        angle = random.uniform(self.config.rotate_limit[0], self.config.rotate_limit[1])
        scale = random.uniform(self.config.scale_limit[0], self.config.scale_limit[1])
        shift_x = random.uniform(self.config.shift_limit[0], self.config.shift_limit[1]) * width
        shift_y = random.uniform(self.config.shift_limit[0], self.config.shift_limit[1]) * height

        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += shift_x
        M[1, 2] += shift_y
        return M

    def __call__(self, rgb, ir, sar, mask):
        """
        Input:
            rgb: (H, W, 3) uint8
            ir:  (H, W) uint8 or compatible type
            sar: (H, W) uint8 or compatible type
            mask: (H, W) uint8
        """
        if not self.config.enable or random.random() > self.config.p_geo:
            return rgb, ir, sar, mask

        h, w = rgb.shape[:2]
        M = self._get_affine_matrix(h, w)
        dsize = (w, h)

        # 1. RGB (均值填充)
        # 统计量是 [0,1]，数据是 uint8，所以 * 255
        rgb_pad = tuple(np.array(self.stats.rgb_mean) * 255.0)
        rgb_aug = cv2.warpAffine(rgb, M, dsize, flags=cv2.INTER_LINEAR, 
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=rgb_pad)

        # 2. IR (均值填充)
        # 假设 IR 输入是 uint8 (0-255)
        ir_pad = self.stats.ir_mean * 255.0
        ir_aug = cv2.warpAffine(ir, M, dsize, flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=ir_pad)

        # 3. SAR (均值填充)
        # SAR 图像背景噪声较大，填 0 极不自然，均值填充是最佳选择
        sar_pad = self.stats.sar_mean * 255.0
        sar_aug = cv2.warpAffine(sar, M, dsize, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=sar_pad)

        # 4. Mask (Ignore Value 填充)
        # 必须使用最近邻插值 (INTER_NEAREST)
        mask_aug = cv2.warpAffine(mask, M, dsize, flags=cv2.INTER_NEAREST,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        return rgb_aug, ir_aug, sar_aug, mask_aug

# ================= 4. 数据集核心 (Dataset) =================

class WHUDataset(Dataset):
    def __init__(self, root_dir, split="train", aug_config=None, stats=None):
        self.split = split
        self.img_dir = os.path.join(root_dir, split, "images")
        self.ir_dir = os.path.join(root_dir, split, "ir")
        self.sar_dir = os.path.join(root_dir, split, "sar")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        
        self.aug_config = aug_config if aug_config else WHUAugmentConfig(enable=False)
        self.stats = stats if stats else DatasetStats()
        
        self.files = (sorted([f for f in os.listdir(self.img_dir) if f.endswith(".tif")])
                      if os.path.exists(self.img_dir) else [])
        
        # 构建增强
        self.custom_affine = CustomAffine(self.aug_config, self.stats) # [NEW]
        self.unified_transform = self._build_unified_transform()
        self.rgb_transform = self._build_rgb_transform()
        self.ir_transform = self._build_ir_transform()
        self.sar_transform = self._build_sar_transform()

    def _build_unified_transform(self):
        """只保留离散几何变换，连续仿射变换移交给 CustomAffine"""
        if not self.aug_config.enable: return None
        c = self.aug_config
        return A.Compose([
            A.OneOf([A.HorizontalFlip(p=1), A.VerticalFlip(p=1), A.RandomRotate90(p=1)], p=c.p_geo),
        ], additional_targets={"ir": "image", "sar": "image", "mask": "mask"})

    def _build_rgb_transform(self):
        if not self.aug_config.enable: return None
        c = self.aug_config
        return A.Compose([
            A.HueSaturationValue(hue_shift_limit=c.h_shift, sat_shift_limit=c.s_shift, val_shift_limit=c.v_shift, p=c.p_hsv),
            A.OneOf([A.RandomBrightnessContrast(p=1), A.RandomGamma(p=1)], p=c.p_intensity),
            A.GaussNoise(std_range=(0.0001, 0.005), p=0.1), # Albumentations 新版参数名 var_limit
        ], p=c.p_rgb)

    def _build_ir_transform(self):
        if not self.aug_config.enable: return None
        c = self.aug_config
        return A.Compose([
            A.OneOf([A.RandomBrightnessContrast(p=1), A.RandomGamma(p=1)], p=c.p_intensity),
            A.GaussNoise(std_range=(0.0001, 0.005), p=0.1),
        ], p=c.p_ir)

    def _build_sar_transform(self):
        if not self.aug_config.enable: return None
        c = self.aug_config
        return A.Compose([
            A.OneOf([A.RandomBrightnessContrast(p=1), A.RandomGamma(p=1)], p=c.p_intensity),
            A.MultiplicativeNoise(multiplier=c.sar_speckle_limit, elementwise=True, p=0.5),
            A.GaussNoise(std_range=(0.0001, 0.005), p=0.1),
        ], p=c.p_sar)

    def _load_raw(self, idx):
        filename = self.files[idx]
        img_bgr = cv2.imread(os.path.join(self.img_dir, filename))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # uint8
        
        ir = cv2.imread(os.path.join(self.ir_dir, filename), cv2.IMREAD_UNCHANGED) # uint8
        sar = cv2.imread(os.path.join(self.sar_dir, filename), cv2.IMREAD_UNCHANGED) # uint8
        mask = cv2.imread(os.path.join(self.mask_dir, filename), cv2.IMREAD_GRAYSCALE) # uint8
        return rgb, ir, sar, mask

    def _standardize(self, rgb, ir, sar):
        rgb = rgb.astype(np.float32) / 255.0
        rgb_mean = np.array(self.stats.rgb_mean, dtype=np.float32).reshape(1, 1, 3)
        rgb_std = np.array(self.stats.rgb_std, dtype=np.float32).reshape(1, 1, 3)
        rgb = (rgb - rgb_mean) / (rgb_std + 1e-6)
        
        ir = ir.astype(np.float32) / 255.0
        ir = (ir - self.stats.ir_mean) / (self.stats.ir_std + 1e-6)
        
        sar = sar.astype(np.float32) / 255.0
        sar = (sar - self.stats.sar_mean) / (self.stats.sar_std + 1e-6)
        return rgb, ir, sar

    def __getitem__(self, idx):
        # 1. Load (uint8)
        rgb, ir, sar, mask = self._load_raw(idx)
        
        # 2. Discrete Transform (Flip/Rotate90)
        if self.unified_transform:
            aug = self.unified_transform(image=rgb, ir=ir, sar=sar, mask=mask)
            rgb, ir, sar, mask = aug["image"], aug["ir"], aug["sar"], aug["mask"]
            
        # 3. [NEW] Custom Affine Transform (Continuous)
        # 支持对 IR/SAR 填充均值，对 Mask 填充 255
        rgb, ir, sar, mask = self.custom_affine(rgb, ir, sar, mask)
        
        # 4. Intensity Transform (Color/Noise)
        if self.aug_config.enable:
            if self.rgb_transform: rgb = self.rgb_transform(image=rgb)["image"]
            if self.ir_transform: ir = self.ir_transform(image=ir)["image"]
            if self.sar_transform: sar = self.sar_transform(image=sar)["image"]
        
        # 5. Mask Cleaning (Strict)
        # 确保仿射变换后的 mask 只有有效类别和 255
        valid_mask = (mask >= 0) & (mask < 8) # 假设 8 类
        mask[~valid_mask] = 255
        
        # 6. Standardize (Normalize)
        rgb, ir, sar = self._standardize(rgb, ir, sar)
        
        # 7. Stack & Tensor
        irsar = np.dstack([ir, sar])
        
        return (torch.from_numpy(rgb.transpose(2, 0, 1)).float(),
                torch.from_numpy(irsar.transpose(2, 0, 1)).float(),
                torch.from_numpy(mask).long())

    def __len__(self):
        return len(self.files)

# ================= 5. Loader 管理 (PipLine) =================

def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

class PipLine:
    def __init__(self, root_dir, batch_size=8, num_workers=16, calculate_stats=True, seed=42, pin_memory=True):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if not isinstance(pin_memory, bool):
            raise ValueError("data.pin_memory must be a YAML boolean (true or false)")
        self.pin_memory = pin_memory
        self.seed = int(seed)

        if calculate_stats:
            self.stats = self._get_stats_distributed()
        else:
            if get_rank() == 0:
                print("Using hardcoded statistics.")
            self.stats = DatasetStats()

    def _get_stats_distributed(self):
        rank = get_rank()
        stats_obj = None

        if rank == 0:
            print(f"[Rank {rank}] Calculating stats...")
            calc = StatisticsCalculator(self.root_dir, split="train")
            stats_obj = calc.compute()
        
        if dist.is_available() and dist.is_initialized():
            object_list = [stats_obj]
            dist.broadcast_object_list(object_list, src=0)
            stats_obj = object_list[0]
            if rank != 0:
                print(f"[Rank {rank}] Received stats via broadcast.")

        return stats_obj

    def get_loader(self, split, augment=False, shuffle=False):
        cfg = WHUAugmentConfig(enable=augment)
        ds = WHUDataset(self.root_dir, split, aug_config=cfg, stats=self.stats)
        
        generator = make_data_generator(self.seed, rank=get_rank(), split=split)
        seed_augmentations(ds, generator.initial_seed(), seed_globals=augment and self.num_workers == 0)

        return DataLoader(
            ds,
            generator=generator,
            worker_init_fn=seed_worker,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=(split == "train"),
        )

# ================= 6. 可视化验证 =================

def visualize_check(loader):
    if get_rank() != 0: return
    
    ds = loader.dataset
    ds.aug_config.enable = True
    # 重置组件以生效
    ds.custom_affine = CustomAffine(ds.aug_config, ds.stats)
    ds.unified_transform = ds._build_unified_transform()
    ds.rgb_transform = ds._build_rgb_transform()
    ds.ir_transform = ds._build_ir_transform()
    ds.sar_transform = ds._build_sar_transform()

    idx = random.randint(0, len(ds) - 1)
    rgb_t, irsar_t, mask_t = ds[idx]
    
    print(f"\n[Check] Sample {idx}")
    print(f"Stats -> RGB Mean:{ds.stats.rgb_mean} | IR Mean:{ds.stats.ir_mean:.3f} | SAR Mean:{ds.stats.sar_mean:.3f}")
    
    # 反标准化
    rgb_vis = rgb_t.numpy().transpose(1, 2, 0) * np.array(ds.stats.rgb_std) + np.array(ds.stats.rgb_mean)
    
    ir_t = irsar_t[0].numpy()
    sar_t = irsar_t[1].numpy()
    
    ir_vis = ir_t * ds.stats.ir_std + ds.stats.ir_mean
    sar_vis = sar_t * ds.stats.sar_std + ds.stats.sar_mean
    
    # 调色板
    palette = {
        0: (102, 204, 102), 1: (204, 102, 102), 2: (204, 204, 102),
        3: (102, 102, 204), 4: (0, 153, 0),     5: (153, 153, 153),
        6: (204, 102, 204), 7: (0, 0, 0)
    }
    
    def colorize_mask(mask, pal):
        h, w = mask.shape
        col_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for c, color in pal.items():
            col_mask[mask == c] = color
        return col_mask

    mask_vis = colorize_mask(mask_t.numpy(), palette)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(np.clip(rgb_vis, 0, 1))
    ax[0].set_title("RGB (Mean Pad)")
    
    ax[1].imshow(ir_vis, cmap="inferno")
    ax[1].set_title(f"IR (Pad ~{ds.stats.ir_mean:.2f})")
    
    ax[2].imshow(sar_vis, cmap="gray")
    ax[2].set_title(f"SAR (Pad ~{ds.stats.sar_mean:.2f})")
    
    ax[3].imshow(mask_vis)
    ax[3].set_title("Mask (Pad 255)")
    
    plt.tight_layout()
    plt.savefig("check_whu_custom_affine.png")
    print("Saved check_whu_custom_affine.png")

if __name__ == "__main__":
    root = "./dataset/WHU_split"
    
    loader_mgr = PipLine(root, batch_size=4, calculate_stats=True)
    train_loader = loader_mgr.get_loader("train", augment=True, shuffle=True)
    if len(train_loader.dataset) > 0:
            visualize_check(train_loader)