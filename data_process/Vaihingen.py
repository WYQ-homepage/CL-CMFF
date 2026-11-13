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
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass, field
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

# ================= 1. 配置类 (Config) =================

@dataclass
class VaihingenAugmentConfig:
    """数据增强配置"""
    enable: bool = True
    # 1. [统一增强] 几何变换
    p_geo: float = 0.5
    rotate_limit: tuple = (-45, 45)
    scale_limit: tuple = (0.8, 1.2)
    shift_limit: tuple = (-0.0625, 0.0625)
    # 2. [RGB专属] 颜色与强度
    p_rgb: float = 0.5
    p_hsv: float = 0.3
    h_shift: int = 20
    s_shift: int = 30
    v_shift: int = 20
    p_intensity: float = 0.3 
    # 3. [DSM专属] 物理参数
    p_dsm: float = 0.5
    dsm_offset_range: tuple = (-0.1, 0.1)  
    dsm_noise_std: float = 0.05            
    # 4. 填充策略 (Mask 填充值)
    mask_pad_value: int = 255   # Ignore Label

@dataclass
class DatasetStats:
    """统计信息"""
    # 默认值
    channels_mean: list = field(default_factory=lambda: [0.47, 0.32, 0.32])
    channels_std: list = field(default_factory=lambda: [0.2, 0.2, 0.2])
    dsm_mean: float = 280.0
    dsm_std: float = 20.0

    def to_dict(self):
        return {
            "channels_mean": self.channels_mean,
            "channels_std": self.channels_std,
            "dsm_mean": self.dsm_mean,
            "dsm_std": self.dsm_std
        }

    @staticmethod
    def from_dict(data):
        stats = DatasetStats()
        stats.channels_mean = data["channels_mean"]
        stats.channels_std = data["channels_std"]
        stats.dsm_mean = data["dsm_mean"]
        stats.dsm_std = data["dsm_std"]
        return stats

# ================= 2. 统计计算工具 (Worker) =================

def _worker_stat(args):
    """
    单个文件的统计计算工作函数
    """
    img_path, dsm_path = args
    
    # 1. 读取 RGB (转换到 0-1 float 便于累加防止溢出)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None # Skip invalid files
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    
    # 2. 读取 DSM
    dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
    if dsm is None:
        return None
    dsm = dsm.astype(np.float32)
        
    # 计算统计量
    # RGB
    # sum: (3,) sum_sq: (3,)
    rgb_sum = np.sum(img, axis=(0, 1))
    rgb_sum_sq = np.sum(img ** 2, axis=(0, 1))
    
    # DSM
    dsm_sum = np.sum(dsm)
    dsm_sum_sq = np.sum(dsm ** 2)
    
    pixel_count = img.shape[0] * img.shape[1]
    
    return rgb_sum, rgb_sum_sq, dsm_sum, dsm_sum_sq, pixel_count

class StatisticsCalculator:
    def __init__(self, root_dir, split="train"):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.dsm_dir = os.path.join(root_dir, split, "dsm")
        
        self.files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".tif")])

    def compute(self):
        print("Calculating dataset statistics (Recalculating every run)...")
        
        # 准备任务列表
        tasks = []
        for f in self.files:
            img_path = os.path.join(self.img_dir, f)
            dsm_path = os.path.join(self.dsm_dir, f)
            tasks.append((img_path, dsm_path))

        # 并行计算
        total_rgb_sum = np.zeros(3, dtype=np.float64)
        total_rgb_sum_sq = np.zeros(3, dtype=np.float64)
        total_dsm_sum = 0.0
        total_dsm_sum_sq = 0.0
        total_pixels = 0

        # 根据 CPU 核心数设定进程数，保留一些余量
        num_processes = min(16, max(1, cpu_count() - 2), max(1, len(tasks)))
        
        with Pool(processes=num_processes) as pool:
            # 使用 tqdm 显示进度
            results = list(tqdm(pool.imap(_worker_stat, tasks), total=len(tasks), desc="Computing Stats"))
        
        # 聚合结果
        for res in results:
            if res is None: continue
            r_sum, r_sq, d_sum, d_sq, count = res
            total_rgb_sum += r_sum
            total_rgb_sum_sq += r_sq
            total_dsm_sum += d_sum
            total_dsm_sum_sq += d_sq
            total_pixels += count

        if total_pixels == 0:
            print("Warning: No valid pixels found during statistics calculation. Using defaults.")
            return DatasetStats()

        # 计算最终 Mean 和 Std
        # E[X]
        rgb_mean = total_rgb_sum / total_pixels
        dsm_mean = total_dsm_sum / total_pixels
        
        # Std = sqrt( E[X^2] - (E[X])^2 )
        rgb_std = np.sqrt((total_rgb_sum_sq / total_pixels) - (rgb_mean ** 2))
        dsm_std = np.sqrt((total_dsm_sum_sq / total_pixels) - (dsm_mean ** 2))

        # 构建对象
        stats = DatasetStats(
            channels_mean=rgb_mean.tolist(),
            channels_std=rgb_std.tolist(),
            dsm_mean=float(dsm_mean),
            dsm_std=float(dsm_std)
        )
        
        # 不再保存到文件
        return stats

# ================= 3. 数据集核心 (Dataset) =================
# (这部分代码与之前保持一致)

class VaihingenDataset(Dataset):
    def __init__(self, root_dir, split="train", aug_config=None, stats=None):
        self.split = split
        self.img_dir = os.path.join(root_dir, split, "images")
        self.dsm_dir = os.path.join(root_dir, split, "dsm")
        self.mask_dir = os.path.join(root_dir, split, "masks")

        if not os.path.exists(self.img_dir):
             raise FileNotFoundError(f"Image dir not found: {self.img_dir}")

        self.aug_config = aug_config if aug_config else VaihingenAugmentConfig(enable=False)
        self.stats = stats if stats else DatasetStats()

        self.files = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".tif")])

        # 预计算 Normalization 参数
        self.rgb_mean = np.array(self.stats.channels_mean, dtype=np.float32).reshape(1, 1, 3)
        self.rgb_std = np.array(self.stats.channels_std, dtype=np.float32).reshape(1, 1, 3)
        self.dsm_mean = self.stats.dsm_mean
        self.dsm_std = self.stats.dsm_std

        # 计算 Raw Domain 下的填充均值 (用于 Augmentation 阶段)
        self.fill_values_rgb = [m * 255.0 for m in self.stats.channels_mean] 
        self.fill_value_dsm = self.stats.dsm_mean

        # 构建增强流水线
        self.unified_transform = self._build_unified_transform()
        self.rgb_transform = self._build_rgb_transform()

    def _build_unified_transform(self):
        if not self.aug_config.enable: return None
        c = self.aug_config
        combined_fill_values = self.fill_values_rgb + [self.fill_value_dsm]

        return A.Compose(
            [
                A.OneOf([A.HorizontalFlip(p=1), A.VerticalFlip(p=1), A.RandomRotate90(p=1)], p=c.p_geo),
                A.Affine(
                    scale=c.scale_limit, 
                    rotate=c.rotate_limit, 
                    translate_percent=c.shift_limit,
                    border_mode=cv2.BORDER_CONSTANT, 
                    fill=combined_fill_values, 
                    fill_mask=c.mask_pad_value,   
                    p=c.p_geo
                ),
            ]
        )

    def _build_rgb_transform(self):
        if not self.aug_config.enable: return None
        c = self.aug_config
        return A.Compose(
            [
                A.HueSaturationValue(hue_shift_limit=c.h_shift, sat_shift_limit=c.s_shift, val_shift_limit=c.v_shift, p=c.p_hsv),
                A.OneOf([A.RandomBrightnessContrast(p=1), A.RandomGamma(p=1)], p=c.p_intensity),
                A.GaussNoise(std_range=(0.0001, 0.005), p=0.5), 
            ],
            p=c.p_rgb,
        )

    def _augment_dsm_physics(self, dsm, mask):
        if self.aug_config.enable and np.random.random() < self.aug_config.p_dsm:
            c = self.aug_config
            offset = np.random.uniform(c.dsm_offset_range[0], c.dsm_offset_range[1])
            noise = np.random.normal(0, c.dsm_noise_std, dsm.shape)
            
            valid_region = (mask != 255)
            dsm[valid_region] = dsm[valid_region] + offset + noise[valid_region]
        return dsm

    def _load_raw(self, idx):
        filename = self.files[idx]
        img_bgr = cv2.imread(os.path.join(self.img_dir, filename))
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) 
        dsm = cv2.imread(os.path.join(self.dsm_dir, filename), cv2.IMREAD_UNCHANGED).astype(np.float32)
        mask = cv2.imread(os.path.join(self.mask_dir, filename), cv2.IMREAD_GRAYSCALE)
        return img, dsm, mask

    def __getitem__(self, idx):
        img, dsm, mask = self._load_raw(idx)

        # 统一几何增强
        if self.unified_transform:
            img_float = img.astype(np.float32)
            merged_input = np.dstack([img_float, dsm])
            aug = self.unified_transform(image=merged_input, mask=mask)
            merged_aug, mask = aug["image"], aug["mask"]
            
            img = merged_aug[:, :, :3] 
            dsm = merged_aug[:, :, 3] 
            img = np.clip(img, 0, 255).astype(np.uint8)

        # RGB 增强
        if self.aug_config.enable and self.rgb_transform:
            img = self.rgb_transform(image=img)["image"]
        
        # DSM 增强
        dsm = self._augment_dsm_physics(dsm, mask)

        # 标准化
        img = (img.astype(np.float32) / 255.0 - self.rgb_mean) / (self.rgb_std + 1e-6)
        dsm = (dsm - self.dsm_mean) / (self.dsm_std + 1e-6)

        # 绝对归零
        ignore_mask = mask == 255
        if np.any(ignore_mask):
            img[ignore_mask] = 0.0
            dsm[ignore_mask] = 0.0
        
        # 截断
        dsm = np.clip(dsm, -5.0, 5.0)

        return (
            torch.from_numpy(img.transpose(2, 0, 1)).float(), 
            torch.from_numpy(dsm).float().unsqueeze(0),       
            torch.from_numpy(mask).long()                     
        )

    def __len__(self):
        return len(self.files)

# ================= 4. Loader 管理 (PipLine) =================

def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

class PipLine:
    def __init__(self, root_dir, batch_size=8, num_workers=4, calculate_stats=True, seed=42, pin_memory=False):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        if not isinstance(pin_memory, bool):
            raise ValueError("data.pin_memory must be a YAML boolean (true or false)")
        self.pin_memory = pin_memory
        self.seed = int(seed)
        
        if calculate_stats:
            self.stats = StatisticsCalculator(root_dir, split="train").compute() if get_rank() == 0 else None
            if dist.is_available() and dist.is_initialized():
                stats = [self.stats]
                dist.broadcast_object_list(stats, src=0)
                self.stats = stats[0]
        else:
            self.stats = DatasetStats()

    def get_loader(self, split, augment=False, shuffle=False):
        cfg = VaihingenAugmentConfig(enable=augment)
        ds = VaihingenDataset(self.root_dir, split, aug_config=cfg, stats=self.stats)
        
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
            persistent_workers=False
        )

# ================= 5. 主程序 (Main) =================

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    
    root = "./dataset/Vaihingen_split"
    
    if os.path.exists(root):
        print("Running Pipeline Test with Auto-Calculation (No Cache)...")
        
        # 实例化时会自动计算均值方差
        loader_mgr = PipLine(root, batch_size=4, num_workers=4, calculate_stats=True)
        
        print("\n[Calculated Stats]")
        print(f"RGB Mean: {loader_mgr.stats.channels_mean}")
        print(f"RGB Std : {loader_mgr.stats.channels_std}")
        print(f"DSM Mean: {loader_mgr.stats.dsm_mean:.2f}")
        print(f"DSM Std : {loader_mgr.stats.dsm_std:.2f}")

        train_loader = loader_mgr.get_loader("train", augment=True, shuffle=True)
        
        if len(train_loader.dataset) > 0:
            global_min = torch.full((4,), float('inf'))
            global_max = torch.full((4,), float('-inf'))

            print("\nScanning Batches (Sanity Check)...")
            for i, (img, dsm, mask) in enumerate(tqdm(train_loader)):
                batch_data = torch.cat([img, dsm], dim=1)
                channels_flatten = batch_data.transpose(0, 1).reshape(4, -1)
                batch_min = channels_flatten.min(dim=1)[0]
                batch_max = channels_flatten.max(dim=1)[0]

                global_min = torch.minimum(global_min, batch_min)
                global_max = torch.maximum(global_max, batch_max)

            print("\n[Sanity Check Result]")
            print(f"Normalized RGB Min (Should be near -2.x): {global_min[:3].tolist()}") 
            print(f"Normalized DSM Min (Should be > -5.0): {global_min[3].item():.4f}") 
            print(f"Normalized DSM Max (Should be < 5.0): {global_max[3].item():.4f}")
            
        else:
            print("Dataset is empty.")
    else:
        print(f"Root path {root} does not exist.")