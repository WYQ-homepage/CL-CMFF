import os
import cv2
import numpy as np
from tqdm import tqdm
import multiprocessing
from functools import partial

# ================= 1. 配置区域 =================
# 原始数据根目录
RAW_DATA_ROOT = "./dataset/Vaihingen"
# 切分后输出目录
OUTPUT_ROOT = "./dataset/Vaihingen_split"

# 文件名模板
IMG_PATTERN = os.path.join(RAW_DATA_ROOT, "top/top_mosaic_09cm_area{}.tif")
DSM_PATTERN = os.path.join(RAW_DATA_ROOT, "dsm/dsm_09cm_matching_area{}.tif")

# GT 路径模板
GT_PATTERN = os.path.join(RAW_DATA_ROOT, "gts_for_participants/top_mosaic_09cm_area{}.tif")
GT_ERODED_PATTERN = os.path.join(RAW_DATA_ROOT, "gts_eroded_for_participants/top_mosaic_09cm_area{}_noBoundary.tif")

# 数据划分
TRAIN_IDS = ["1", "2", "3", "4", "7", "8", "10", "11", "12", "14", "15", "17", "20", "21", "22", "24", "28", "31", "33", "35", "38"]
VAL_IDS = ["5", "16", "26", "27", "29", "30"]
TEST_IDS = ["6", "13", "23", "32", "34", "37"]

# --- 采样策略配置 ---
# 1. 基础策略 (所有数据集)
CROP_SIZE_STD = 512
STRIDE_STD_TRAIN = 256      # 训练集：重叠采样
STRIDE_STD_VAL = 512        # 验证/测试集：不重叠 (Stride = Crop Size)

# 2. 大窗口策略 (仅训练集)
ENABLE_LARGE_CROP = True    # 开关
CROP_SIZE_LARGE = 1024      # 大窗口尺寸
STRIDE_LARGE = 512          # 大窗口步长
TARGET_SIZE_LARGE = 512     # 下采样后的目标尺寸

# --- Class 4 重采样阈值 ---
CLASS4_THRESHOLD = 0.1  # 10%

# 颜色板定义 (RGB -> Class ID)
PALETTE = {
    (255, 255, 255): 0,  # Impervious surfaces
    (0, 0, 255): 1,      # Buildings
    (0, 255, 255): 2,    # Low vegetation
    (0, 255, 0): 3,      # Trees
    (255, 255, 0): 4,    # Cars
    (255, 0, 0): 5,      # Clutter
    (0, 0, 0): 6,        # Undefined
}

# ================= 2. 核心工具函数 =================

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def rgb_to_mask_fast(rgb_img, palette):
    """
    高速 Mask 转换函数 (位运算优化)
    """
    r, g, b = rgb_img[..., 0].astype(np.int32), rgb_img[..., 1].astype(np.int32), rgb_img[..., 2].astype(np.int32)
    img_int = (r << 16) | (g << 8) | b
    
    mask = np.full(img_int.shape, 255, dtype=np.uint8)
    
    for color, idx in palette.items():
        c_int = (color[0] << 16) | (color[1] << 8) | color[2]
        mask[img_int == c_int] = idx
        
    return mask

def get_robust_grid_points(total_length, crop_size, stride):
    """计算切片坐标"""
    if total_length <= crop_size: return [0]
    points = []
    current = 0
    while current + crop_size <= total_length:
        points.append(current)
        current += stride
    if not points or points[-1] != total_length - crop_size:
        points.append(total_length - crop_size)
    return points

def save_single_patch(img, dsm, mask, filename, save_dirs):
    """
    底层保存函数
    输入 img 为 RGB，保存前转为 BGR 以适配 OpenCV
    """
    if np.all(mask == 255): # Double check inside save to be safe
        return False

    cv2.imwrite(os.path.join(save_dirs['img'], filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dirs['dsm'], filename), dsm)
    cv2.imwrite(os.path.join(save_dirs['mask'], filename), mask)
    return True

def process_single_area(area_id, split_name, save_dirs):
    """
    处理单张大图的函数
    """
    # ================= 修改 1: 标签选择逻辑 =================
    # 验证集 (val) 和 测试集 (test) 都使用腐蚀后的标签
    if split_name in ['val', 'test']:
        gt_p = GT_ERODED_PATTERN.format(area_id)
    else:
        # 仅 train 使用原始标签
        gt_p = GT_PATTERN.format(area_id)
    # ======================================================

    img_p = IMG_PATTERN.format(area_id)
    dsm_p = DSM_PATTERN.format(area_id)

    if not (os.path.exists(img_p) and os.path.exists(dsm_p) and os.path.exists(gt_p)):
        return 0 

    try:
        img_bgr = cv2.imread(img_p)
        if img_bgr is None: raise ValueError(f"Image read failed: {img_p}")
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        dsm = cv2.imread(dsm_p, cv2.IMREAD_UNCHANGED).astype(np.float32)
        if dsm is None: raise ValueError(f"DSM read failed: {dsm_p}")
        
        gt_bgr = cv2.imread(gt_p)
        if gt_bgr is None: raise ValueError(f"GT read failed: {gt_p}")
        gt_rgb = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"Error reading area {area_id}: {e}")
        return 0

    mask = rgb_to_mask_fast(gt_rgb, PALETTE)
    h, w = img.shape[:2]

    total_patches = 0
    is_train = (split_name == 'train')

    # ================= 策略 1: 标准 512x512 切分 =================
    stride_std = STRIDE_STD_TRAIN if is_train else STRIDE_STD_VAL
    y_list = get_robust_grid_points(h, CROP_SIZE_STD, stride_std)
    x_list = get_robust_grid_points(w, CROP_SIZE_STD, stride_std)

    for y in y_list:
        for x in x_list:
            # 1. 基础采样
            img_c = img[y:y+CROP_SIZE_STD, x:x+CROP_SIZE_STD]
            dsm_c = dsm[y:y+CROP_SIZE_STD, x:x+CROP_SIZE_STD]
            mask_c = mask[y:y+CROP_SIZE_STD, x:x+CROP_SIZE_STD]

            if img_c.shape[:2] != (CROP_SIZE_STD, CROP_SIZE_STD): continue
            if np.all(mask_c == 255): continue # 跳过背景

            # 保存当前基础切片
            fname = f"area{area_id}_{y}_{x}.tif"
            if save_single_patch(img_c, dsm_c, mask_c, fname, save_dirs):
                total_patches += 1

            # ================= 修改 2: Class 4 上下左右重采样 =================
            # 仅在训练集进行增强
            if is_train:
                ratio = np.mean(mask_c == 4)
                if ratio >= CLASS4_THRESHOLD:
                    # 定义位移步长：规定步长的一半
                    half_stride = stride_std // 2
                    
                    # 定义 4 个方向的偏移 (dy, dx, suffix)
                    shifts = [
                        (-half_stride, 0, "_shift_u"), # 上
                        (half_stride, 0,  "_shift_d"), # 下
                        (0, -half_stride, "_shift_l"), # 左
                        (0, half_stride,  "_shift_r")  # 右
                    ]

                    for dy, dx, suffix in shifts:
                        new_y, new_x = y + dy, x + dx

                        # 边界检查：确保偏移后的坐标在图像范围内
                        if 0 <= new_y <= h - CROP_SIZE_STD and 0 <= new_x <= w - CROP_SIZE_STD:
                            img_shift = img[new_y:new_y+CROP_SIZE_STD, new_x:new_x+CROP_SIZE_STD]
                            dsm_shift = dsm[new_y:new_y+CROP_SIZE_STD, new_x:new_x+CROP_SIZE_STD]
                            mask_shift = mask[new_y:new_y+CROP_SIZE_STD, new_x:new_x+CROP_SIZE_STD]
                            
                            fname_shift = f"area{area_id}_{y}_{x}{suffix}.tif"
                            if save_single_patch(img_shift, dsm_shift, mask_shift, fname_shift, save_dirs):
                                total_patches += 1

    # ================= 策略 2: [训练集专属] 1024x1024 切分并下采样 =================
    if is_train and ENABLE_LARGE_CROP:
        y_list_lg = get_robust_grid_points(h, CROP_SIZE_LARGE, STRIDE_LARGE)
        x_list_lg = get_robust_grid_points(w, CROP_SIZE_LARGE, STRIDE_LARGE)

        for y in y_list_lg:
            for x in x_list_lg:
                # 1. 基础大窗口采样
                img_lg = img[y:y+CROP_SIZE_LARGE, x:x+CROP_SIZE_LARGE]
                dsm_lg = dsm[y:y+CROP_SIZE_LARGE, x:x+CROP_SIZE_LARGE]
                mask_lg = mask[y:y+CROP_SIZE_LARGE, x:x+CROP_SIZE_LARGE]

                if img_lg.shape[:2] != (CROP_SIZE_LARGE, CROP_SIZE_LARGE): continue
                if np.all(mask_lg == 255): continue

                # 执行缩放保存 (Scale Down)
                img_resized = cv2.resize(img_lg, (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_LINEAR)
                dsm_resized = cv2.resize(dsm_lg, (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(mask_lg, (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_NEAREST)

                fname_lg = f"area{area_id}_{y}_{x}_scaled.tif"
                if save_single_patch(img_resized, dsm_resized, mask_resized, fname_lg, save_dirs):
                    total_patches += 1

                # ================= 修改 2 (应用到大窗口): Class 4 上下左右重采样 =================
                ratio = np.mean(mask_lg == 4) # 在原始 1024 分辨率下计算比例
                if ratio >= CLASS4_THRESHOLD:
                    half_stride = STRIDE_LARGE // 2 # 512 // 2 = 256
                    
                    shifts = [
                        (-half_stride, 0, "_shift_u_scaled"),
                        (half_stride, 0,  "_shift_d_scaled"),
                        (0, -half_stride, "_shift_l_scaled"),
                        (0, half_stride,  "_shift_r_scaled")
                    ]

                    for dy, dx, suffix in shifts:
                        new_y, new_x = y + dy, x + dx

                        # 边界检查 (针对 1024 窗口)
                        if 0 <= new_y <= h - CROP_SIZE_LARGE and 0 <= new_x <= w - CROP_SIZE_LARGE:
                            img_s = img[new_y:new_y+CROP_SIZE_LARGE, new_x:new_x+CROP_SIZE_LARGE]
                            dsm_s = dsm[new_y:new_y+CROP_SIZE_LARGE, new_x:new_x+CROP_SIZE_LARGE]
                            mask_s = mask[new_y:new_y+CROP_SIZE_LARGE, new_x:new_x+CROP_SIZE_LARGE]

                            # 必须同样进行 resize 操作
                            img_s_re = cv2.resize(img_s, (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_LINEAR)
                            dsm_s_re = cv2.resize(dsm_s, (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_LINEAR)
                            mask_s_re = cv2.resize(mask_s, (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_NEAREST)
                            
                            fname_s = f"area{area_id}_{y}_{x}{suffix}.tif"
                            if save_single_patch(img_s_re, dsm_s_re, mask_s_re, fname_s, save_dirs):
                                total_patches += 1

    return total_patches

def worker_wrapper(args):
    return process_single_area(*args)

def process_dataset_multiprocess():
    num_workers = max(1, (os.cpu_count() or 1) - 1)
    print(f"Starting processing with {num_workers} workers...")

    tasks_config = [
        (TRAIN_IDS, 'train'),
        (VAL_IDS,   'val'),
        (TEST_IDS,  'test')
    ]
    
    for ids, split in tasks_config:
        print(f"\nProcessing {split} set...")
        
        sub_dirs = {
            'img': os.path.join(OUTPUT_ROOT, split, 'images'),
            'dsm': os.path.join(OUTPUT_ROOT, split, 'dsm'),
            'mask': os.path.join(OUTPUT_ROOT, split, 'masks')
        }
        for p in sub_dirs.values():
            create_dir(p)
            
        task_list = [(aid, split, sub_dirs) for aid in ids]
        
        valid_tasks = []
        for t in task_list:
            if os.path.exists(IMG_PATTERN.format(t[0])):
                valid_tasks.append(t)
            else:
                print(f"[Warning] File for Area {t[0]} not found, skipping.")
        
        if not valid_tasks:
            print(f"No valid files found for {split}.")
            continue

        total_split = 0
        with multiprocessing.Pool(processes=num_workers) as pool:
            for patches_count in tqdm(pool.imap(worker_wrapper, valid_tasks), total=len(valid_tasks)):
                total_split += patches_count
        
        print(f"Finished {split}: {total_split} patches generated.")

if __name__ == "__main__":
    if not os.path.exists(RAW_DATA_ROOT):
        print(f"Error: RAW_DATA_ROOT '{RAW_DATA_ROOT}' does not exist.")
    else:
        multiprocessing.freeze_support()
        process_dataset_multiprocess()