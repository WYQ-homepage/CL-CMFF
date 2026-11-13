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
class PotsdamAugmentConfig:
    """数据增强配置"""
    enable: bool = True
    p_geo: float = 1
    rotate_limit: tuple = (-45, 45)
    scale_limit: tuple = (0.8, 1.2)
    shift_limit: tuple = (-0.0625, 0.0625)
    p_rgb: float = 0.5
    p_hsv: float = 0.3
    h_shift: int = 20
    s_shift: int = 30
    v_shift: int = 20
    p_intensity: float = 0.3
    p_ir: float = 0.5
    p_dsm: float = 0.5
    dsm_offset_range: tuple = (-0.1, 0.1)
    dsm_noise_std: float = 0.05
    img_pad_value: int = 0
    irdsm_pad_value: float = 0
    mask_pad_value: int = 255

@dataclass
class DatasetStats:
    """统计信息：适应 RGB + IRDSM 结构"""
    rgb_mean: list = field(default_factory=lambda: [0.4, 0.4, 0.4])
    rgb_std: list = field(default_factory=lambda: [0.2, 0.2, 0.2])
    irdsm_mean: list = field(default_factory=lambda: [0.4, 50.0])
    irdsm_std: list = field(default_factory=lambda: [0.2, 20.0])

# ================= 2. 统计计算工具 (Calculator) =================

def _worker_stat(file_info):
    """Worker: 计算 RGB 和 IRDSM 的统计量"""
    rgb_path, ir_path, dsm_path, mask_path = file_info

    img_bgr = cv2.imread(rgb_path)
    if img_bgr is None: return None
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    ir = cv2.imread(ir_path, cv2.IMREAD_UNCHANGED)
    if ir is None: return None
    ir = ir.astype(np.float32) / 255.0

    dsm = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED)
    if dsm is None: return None
    dsm = dsm.astype(np.float32)

    irdsm = np.dstack([ir, dsm])
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    valid_mask = mask != 255

    if not np.any(valid_mask): return None

    valid_rgb = rgb[valid_mask]
    valid_irdsm = irdsm[valid_mask]

    return (
        np.sum(valid_rgb, axis=0),
        np.sum(valid_rgb**2, axis=0),
        np.sum(valid_irdsm, axis=0),
        np.sum(valid_irdsm**2, axis=0),
        np.sum(valid_mask),
    )

class StatisticsCalculator:
    def __init__(self, root_dir, split="train"):
        self.img_dir = os.path.join(root_dir, split, "images")
        self.ir_dir = os.path.join(root_dir, split, "ir")
        self.dsm_dir = os.path.join(root_dir, split, "dsm")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.files = (sorted([f for f in os.listdir(self.img_dir) if f.endswith(".tif")])
                      if os.path.exists(self.img_dir) else [])

    def compute(self) -> DatasetStats:
        if not self.files: return DatasetStats()

        file_list = [
            (os.path.join(self.img_dir, f), os.path.join(self.ir_dir, f),
             os.path.join(self.dsm_dir, f), os.path.join(self.mask_dir, f))
            for f in self.files
        ]

        num_workers = min(cpu_count(), 16, max(1, len(file_list)))
        print(f"Calculating stats with {num_workers} workers...")

        rgb_sum = np.zeros(3)
        rgb_sq_sum = np.zeros(3)
        irdsm_sum = np.zeros(2)
        irdsm_sq_sum = np.zeros(2)
        count = 0

        with Pool(num_workers) as p:
            for res in tqdm(p.imap(_worker_stat, file_list), total=len(file_list)):
                if res:
                    r_s, r_sq, id_s, id_sq, c = res
                    rgb_sum += r_s
                    rgb_sq_sum += r_sq
                    irdsm_sum += id_s
                    irdsm_sq_sum += id_sq
                    count += c

        if count == 0: raise ValueError("No valid pixels.")

        rgb_mean = rgb_sum / count
        rgb_std = np.sqrt((rgb_sq_sum / count) - (rgb_mean**2))
        irdsm_mean = irdsm_sum / count
        irdsm_std = np.sqrt((irdsm_sq_sum / count) - (irdsm_mean**2))

        print(f"Stats Calculated:\n RGB Mean:{rgb_mean}\n IRDSM Mean:{irdsm_mean}")

        return DatasetStats(
            rgb_mean=rgb_mean.tolist(),
            rgb_std=rgb_std.tolist(),
            irdsm_mean=irdsm_mean.tolist(),
            irdsm_std=irdsm_std.tolist(),
        )

# ================= 3. 自定义仿射变换 (Custom Affine) [核心新增] =================

class CustomAffine:
    """
    高性能自定义仿射变换，支持多模态独立均值填充。
    替代 albumentations.Affine 以解决多通道填充问题。
    """
    def __init__(self, config: PotsdamAugmentConfig, stats: DatasetStats):
        self.config = config
        self.stats = stats

    def _get_affine_matrix(self, height, width):
        """生成随机仿射矩阵"""
        # 1. 旋转中心 (图像中心)
        center = (width / 2, height / 2)
        
        # 2. 随机参数
        angle = random.uniform(self.config.rotate_limit[0], self.config.rotate_limit[1])
        scale = random.uniform(self.config.scale_limit[0], self.config.scale_limit[1])
        shift_x = random.uniform(self.config.shift_limit[0], self.config.shift_limit[1]) * width
        shift_y = random.uniform(self.config.shift_limit[0], self.config.shift_limit[1]) * height

        # 3. 计算旋转+缩放矩阵 (2x3)
        M = cv2.getRotationMatrix2D(center, angle, scale)

        # 4. 添加平移
        M[0, 2] += shift_x
        M[1, 2] += shift_y
        
        return M

    def __call__(self, rgb, irdsm, mask):
        """
        应用仿射变换
        rgb: (H, W, 3) uint8
        irdsm: (H, W, 2) float32 (IR, DSM)
        mask: (H, W) uint8
        """
        if not self.config.enable or random.random() > self.config.p_geo:
            return rgb, irdsm, mask

        h, w = rgb.shape[:2]
        M = self._get_affine_matrix(h, w)
        dsize = (w, h)

        # 1. RGB 变换: 填充值 = RGB 均值 (需转为 0-255)
        # 统计量中的 rgb_mean 是 [0,1]，这里是 uint8 数据，所以 * 255
        rgb_pad = tuple(np.array(self.stats.rgb_mean) * 255.0)
        rgb_aug = cv2.warpAffine(rgb, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=rgb_pad)

        # 2. IRDSM 变换: 分通道处理
        # 拆分通道
        ir = irdsm[:, :, 0]
        dsm = irdsm[:, :, 1]

        # IR 填充值 (0-1 -> 0-255 假设原始IR是uint8读入后未归一化，或者如果是float需注意)
        # 根据 _load_raw, IR 读入是 (H,W,1), 这里是 (H,W)。
        # 注意: 统计量 IR 是归一化后的 mean。
        # _load_raw 中 IR 是 cv2.IMREAD_UNCHANGED。如果是 uint8，则 range 0-255。
        # 假设 IR 原图是 uint8，则 pad 值应为 mean * 255。
        ir_pad = self.stats.irdsm_mean[0] * 255.0
        ir_aug = cv2.warpAffine(ir, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=ir_pad)

        # DSM 填充值 (Raw Float)
        dsm_pad = self.stats.irdsm_mean[1]
        dsm_aug = cv2.warpAffine(dsm, M, dsize, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=dsm_pad)

        # 合并回 IRDSM
        irdsm_aug = np.dstack([ir_aug, dsm_aug])

        # 3. Mask 变换: 填充值 = 255 (Ignore), 插值 = 最近邻
        mask_aug = cv2.warpAffine(mask, M, dsize, flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

        return rgb_aug, irdsm_aug, mask_aug

# ================= 4. 数据集核心 (Dataset) =================

class PotsdamDataset(Dataset):
    def __init__(self, root_dir, split="train", aug_config=None, stats=None):
        self.split = split
        self.img_dir = os.path.join(root_dir, split, "images")
        self.ir_dir = os.path.join(root_dir, split, "ir")
        self.dsm_dir = os.path.join(root_dir, split, "dsm")
        self.mask_dir = os.path.join(root_dir, split, "masks")
        self.aug_config = aug_config if aug_config else PotsdamAugmentConfig(enable=False)
        self.stats = stats if stats else DatasetStats()
        self.files = (sorted([f for f in os.listdir(self.img_dir) if f.endswith(".tif")])
                      if os.path.exists(self.img_dir) else [])
        
        # 构建 Transform
        self.custom_affine = CustomAffine(self.aug_config, self.stats) # 使用自定义仿射
        self.unified_transform = self._build_unified_transform()
        self.rgb_transform = self._build_rgb_transform()
        self.ir_transform = self._build_ir_transform()

    def _build_unified_transform(self):
        """只保留 Flip/Rotate90 等离散几何变换，Affine 已被 CustomAffine 接管"""
        if not self.aug_config.enable: return None
        c = self.aug_config
        return A.Compose([
            A.OneOf([A.HorizontalFlip(p=1), A.VerticalFlip(p=1), A.RandomRotate90(p=1)], p=c.p_geo),
            # A.Affine 已移除，由 CustomAffine 处理
        ], additional_targets={"irdsm": "image", "mask": "mask"})

    def _build_rgb_transform(self):
        if not self.aug_config.enable: return None
        c = self.aug_config
        return A.Compose([
            A.HueSaturationValue(hue_shift_limit=c.h_shift, sat_shift_limit=c.s_shift, val_shift_limit=c.v_shift, p=c.p_hsv),
            A.OneOf([A.RandomBrightnessContrast(p=1), A.RandomGamma(p=1)], p=c.p_intensity),
            A.GaussNoise(std_range=(0.0001, 0.005), p=0.3),
        ], p=c.p_rgb)

    def _build_ir_transform(self):
        if not self.aug_config.enable: return None
        c = self.aug_config
        return A.Compose([
            A.OneOf([A.RandomBrightnessContrast(p=1), A.RandomGamma(p=1)], p=c.p_intensity),
            A.GaussNoise(std_range=(0.0001, 0.005), p=0.3),
        ], p=c.p_ir)

    def _augment_dsm(self, dsm):
        if self.aug_config.enable and np.random.random() < self.aug_config.p_dsm:
            c = self.aug_config
            dsm = dsm + np.random.uniform(c.dsm_offset_range[0], c.dsm_offset_range[1])
            dsm = dsm + np.random.normal(0, c.dsm_noise_std, dsm.shape)
        return dsm

    def _load_raw(self, idx):
        filename = self.files[idx]
        img_bgr = cv2.imread(os.path.join(self.img_dir, filename))
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB) # uint8
        
        ir = cv2.imread(os.path.join(self.ir_dir, filename), cv2.IMREAD_UNCHANGED) # uint8 or uint16
        if len(ir.shape) == 2: ir = ir[:, :, np.newaxis]
        
        dsm = cv2.imread(os.path.join(self.dsm_dir, filename), cv2.IMREAD_UNCHANGED).astype(np.float32) # float32
        if len(dsm.shape) == 2: dsm = dsm[:, :, np.newaxis]
        
        # 暂时合并，CustomAffine 会在内部拆开处理以应用不同的 padding
        irdsm = np.concatenate([ir, dsm], axis=2)
        mask = cv2.imread(os.path.join(self.mask_dir, filename), cv2.IMREAD_GRAYSCALE)
        return rgb, irdsm, mask

    def _standardize(self, rgb, irdsm):
        # RGB 标准化
        rgb = rgb.astype(np.float32) / 255.0
        rgb_mean = np.array(self.stats.rgb_mean, dtype=np.float32).reshape(1, 1, 3)
        rgb_std = np.array(self.stats.rgb_std, dtype=np.float32).reshape(1, 1, 3)
        rgb = (rgb - rgb_mean) / (rgb_std + 1e-6)
        
        # IR 标准化 (假设 IR 原本是 0-255)
        irdsm[:, :, 0] = irdsm[:, :, 0] / 255.0
        
        # IRDSM 统一减均值除方差
        irdsm_mean = np.array(self.stats.irdsm_mean, dtype=np.float32).reshape(1, 1, 2)
        irdsm_std = np.array(self.stats.irdsm_std, dtype=np.float32).reshape(1, 1, 2)
        irdsm = (irdsm - irdsm_mean) / (irdsm_std + 1e-6)
        return rgb, irdsm

    def __getitem__(self, idx):
        rgb, irdsm, mask = self._load_raw(idx)

        # 1. 离散几何变换 (Flip, Rotate90)
        if self.unified_transform:
            aug = self.unified_transform(image=rgb, irdsm=irdsm, mask=mask)
            rgb, irdsm, mask = aug["image"], aug["irdsm"], aug["mask"]
        
        # 2. [NEW] 自定义连续仿射变换 (Scale, Rotate, Shift) - 支持特定均值填充
        rgb, irdsm, mask = self.custom_affine(rgb, irdsm, mask)

        # 3. 颜色/噪声增强
        if self.aug_config.enable:
            if self.rgb_transform: rgb = self.rgb_transform(image=rgb)["image"]
            
            ir, dsm = irdsm[:, :, 0], irdsm[:, :, 1]
            if self.ir_transform: ir = self.ir_transform(image=ir)["image"]
            dsm = self._augment_dsm(dsm)
            irdsm = np.dstack([ir, dsm])

        # 4. 忽略区域填充 (在仿射变换产生的新区域已经被正确填充了，这里主要处理原始Mask中的忽略区域)
        # 逻辑稍微冗余但安全：Mask==255 的区域会被填充为均值
        # 注意：仿射变换后 padding 区域 mask 也是 255，所以这里会再次确保 padding 区域值正确
        ignore_mask = mask == 255
        if np.any(ignore_mask):
            rgb_mean_val = np.array(self.stats.rgb_mean) * 255.0
            rgb[ignore_mask] = rgb_mean_val.astype(np.uint8) # 保持 uint8
            
            ir_mean_val = self.stats.irdsm_mean[0] * 255.0 # 假设 IR 是 0-255
            dsm_mean_val = self.stats.irdsm_mean[1]
            
            # IRDSM 此时是 float32 (因为 dsm 是 float, concat 后整体转 float) 或者是混合类型
            # 实际上 load_raw 中 concat 导致 ir 变成了 float (如果 dsm 是 float)
            # 所以这里可以直接赋值
            irdsm[ignore_mask, 0] = ir_mean_val
            irdsm[ignore_mask, 1] = dsm_mean_val

        # 5. 标准化
        rgb, irdsm = self._standardize(rgb, irdsm)

        return (torch.from_numpy(rgb.transpose(2, 0, 1)).float(),
                torch.from_numpy(irdsm.transpose(2, 0, 1)).float(),
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
            print(f"[Rank {rank}] Calculating stats (Memory Mode)...")
            calc = StatisticsCalculator(self.root_dir, split="train")
            stats_obj = calc.compute()
        else:
            pass

        if dist.is_available() and dist.is_initialized():
            object_list = [stats_obj]
            dist.broadcast_object_list(object_list, src=0)
            stats_obj = object_list[0]
            if rank != 0:
                print(f"[Rank {rank}] Received stats via broadcast.")

        return stats_obj

    def get_loader(self, split, augment=False, shuffle=False):
        cfg = PotsdamAugmentConfig(enable=augment)
        ds = PotsdamDataset(self.root_dir, split, aug_config=cfg, stats=self.stats)
        
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
    # 强制开启增强进行测试
    ds.aug_config.enable = True
    
    # 重新构建 transform 确保参数生效
    ds.custom_affine = CustomAffine(ds.aug_config, ds.stats)
    ds.unified_transform = ds._build_unified_transform()
    ds.rgb_transform = ds._build_rgb_transform()
    ds.ir_transform = ds._build_ir_transform()

    idx = random.randint(0, len(ds) - 1)
    print(f"Checking index: {idx}")
    rgb_t, irdsm_t, mask_t = ds[idx]
    print(f"\n[Check] Sample {idx} | RGB:{rgb_t.shape}, IRDSM:{irdsm_t.shape}, Mask:{mask_t.shape}")
    
    # 反标准化以便可视化
    rgb_vis = rgb_t.numpy().transpose(1, 2, 0) * np.array(ds.stats.rgb_std) + np.array(ds.stats.rgb_mean)
    irdsm_vis = irdsm_t.numpy().transpose(1, 2, 0) * np.array(ds.stats.irdsm_std) + np.array(ds.stats.irdsm_mean)

    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    ax[0].imshow(np.clip(rgb_vis, 0, 1))
    ax[0].set_title("RGB (Affine Mean Pad)")
    ax[1].imshow(irdsm_vis[:, :, 0], cmap="inferno")
    ax[1].set_title(f"IR (Pad Val ~{ds.stats.irdsm_mean[0]:.2f})")
    im = ax[2].imshow(irdsm_vis[:, :, 1], cmap="jet")
    ax[2].set_title(f"DSM (Pad Val ~{ds.stats.irdsm_mean[1]:.1f})")
    plt.colorbar(im, ax=ax[2])
    ax[3].imshow(mask_t, cmap="gray", vmin=0, vmax=255)
    ax[3].set_title("Mask (Pad 255)")
    plt.tight_layout()
    plt.savefig("check_custom_affine.png")
    print("Saved check_custom_affine.png")

if __name__ == "__main__":
    root = "./dataset/Potsdam_split"
    if os.path.exists(root):
        loader_mgr = PipLine(root, batch_size=4, calculate_stats=True)
        train_loader = loader_mgr.get_loader("train", augment=True, shuffle=True)
        if len(train_loader.dataset) > 0:
            visualize_check(train_loader)