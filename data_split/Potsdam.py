import os
import cv2
import numpy as np
from skimage import io
from tqdm import tqdm
import multiprocessing
from functools import partial

# ================= 1. Configuration Area =================
# Raw data root directory
RAW_DATA_ROOT = "./dataset/Potsdam"
# Output directory
OUTPUT_ROOT = "./dataset/Potsdam_split"

# File Pattern Templates
# RGBIR is usually Tiff
IMG_PATTERN = os.path.join(RAW_DATA_ROOT, "4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif")

# DSM Pattern is tricky. We will use a helper function to find the file because of '02_10' vs '2_10' padding issues.
# Path to the folder containing DSMs
DSM_FOLDER = os.path.join(RAW_DATA_ROOT, "1_DSM_normalisation")

# GT Patterns (Train uses raw, Val/Test uses eroded)
GT_PATTERN = os.path.join(RAW_DATA_ROOT, "5_Labels_all/top_potsdam_{}_label.tif")
GT_ERODED_PATTERN = os.path.join(RAW_DATA_ROOT, "5_Labels_all_noBoundary/top_potsdam_{}_label_noBoundary.tif")

# Data Splits
TRAIN_IDS = ["6_7", "6_9", "7_7", "7_8", "2_10", "2_13", "3_10", "3_12", "3_14", "4_10", "4_11", "4_12", "5_10", "5_12", "5_14", "5_15", "6_10", "6_13", "6_14", "7_10", "7_12", "7_13"]
VAL_IDS = ["7_9", "3_13", "4_13", "4_14", "5_13", "6_11", "6_12", "7_11"]
TEST_IDS = ["6_8", "2_11", "2_12", "2_14", "3_11", "4_15", "5_11", "6_15"]

# --- Sampling Strategy Config ---
# 1. Basic Strategy
CROP_SIZE_STD = 512
STRIDE_STD_TRAIN = 256      # Train: Overlapping
STRIDE_STD_VAL = 512        # Val/Test: Non-overlapping

# 2. Large Crop Strategy (Train only)
ENABLE_LARGE_CROP = True    # Toggle
CROP_SIZE_LARGE = 1024      # Large crop size
STRIDE_LARGE = 512          # Large crop stride
TARGET_SIZE_LARGE = 512     # Target size after downsampling

# --- Class 4 (Cars) Resampling Threshold ---
CLASS4_THRESHOLD = 0.1      # 10%

# Palette Definition (RGB -> Class ID)
PALETTE = {
    (255, 255, 255): 0, # Impervious surfaces
    (0, 0, 255): 1,     # Buildings
    (0, 255, 255): 2,   # Low vegetation
    (0, 255, 0): 3,     # Trees
    (255, 255, 0): 4,   # Cars
    (255, 0, 0): 5,     # Clutter
    (0, 0, 0): 6,       # Undefined
}

# ================= 2. Core Utility Functions =================

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_dsm_path(area_id, dsm_folder):
    """
    Robustly find the DSM file path.
    Handles '2_10' -> '02_10' zero padding and extension (.jpg/.tif) issues.
    """
    if not os.path.exists(dsm_folder):
        # Fallback to check relative path if user didn't set absolute path correctly
        return None

    # Parse Row and Column
    parts = area_id.split('_')
    if len(parts) == 2:
        row, col = parts
        # Try different naming conventions
        candidates = [
            f"dsm_potsdam_{area_id}_normalized_lastools.jpg",  # Standard
            f"dsm_potsdam_{int(row):02d}_{int(col):02d}_normalized_lastools.jpg", # Zero padded
            f"dsm_potsdam_{area_id}_normalized_lastools.tif",  # Tiff version
            f"dsm_potsdam_{int(row):02d}_{int(col):02d}_normalized_lastools.tif"  # Zero padded Tiff
        ]
        
        for cand in candidates:
            path = os.path.join(dsm_folder, cand)
            if os.path.exists(path):
                return path
    
    # Fallback: direct check
    return None

def rgb_to_mask_fast(rgb_img, palette):
    """
    [Optimization] Fast Mask Conversion (Bitwise)
    """
    r, g, b = rgb_img[..., 0].astype(np.int32), rgb_img[..., 1].astype(np.int32), rgb_img[..., 2].astype(np.int32)
    img_int = (r << 16) | (g << 8) | b
    
    mask = np.full(img_int.shape, 255, dtype=np.uint8)
    
    for color, idx in palette.items():
        c_int = (color[0] << 16) | (color[1] << 8) | color[2]
        mask[img_int == c_int] = idx
        
    return mask

def get_robust_grid_points(total_length, crop_size, stride):
    """Calculate crop coordinates"""
    if total_length <= crop_size: return [0]
    points = []
    current = 0
    while current + crop_size <= total_length:
        points.append(current)
        current += stride
    if not points or points[-1] != total_length - crop_size:
        points.append(total_length - crop_size)
    return points

def save_single_patch(img_rgb, img_ir, dsm, mask, filename, save_dirs):
    """
    Base Save Function (Potsdam: RGB, IR, DSM, Mask)
    """
    # Skip pure background
    if np.all(mask == 255):
        return False

    # 1. Save RGB (Convert to BGR for OpenCV)
    cv2.imwrite(os.path.join(save_dirs['img'], filename), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    # 2. Save IR (Single channel)
    cv2.imwrite(os.path.join(save_dirs['ir'], filename), img_ir)
    # 3. Save DSM
    cv2.imwrite(os.path.join(save_dirs['dsm'], filename), dsm)
    # 4. Save Mask
    cv2.imwrite(os.path.join(save_dirs['mask'], filename), mask)
    
    return True

def process_single_area(area_id, split_name, save_dirs):
    """
    Process a single Potsdam large image
    """
    # ================= 1. Path & Label Strategy =================
    img_p = IMG_PATTERN.format(area_id)
    
    # GT Strategy: Raw for Train, Eroded for Val/Test
    if split_name in ['val', 'test']:
        gt_p = GT_ERODED_PATTERN.format(area_id)
    else:
        gt_p = GT_PATTERN.format(area_id)

    # Robust DSM Path Finding
    dsm_p = get_dsm_path(area_id, DSM_FOLDER)

    # Check existence
    if not (os.path.exists(img_p) and dsm_p and os.path.exists(dsm_p) and os.path.exists(gt_p)):
        # If DSM is missing, print a specific warning
        if not dsm_p or not os.path.exists(dsm_p):
            print(f"[Skipping] DSM not found for Area {area_id} in {DSM_FOLDER}")
        return 0

    try:
        # Read 4-channel Tiff (RGBIR) - skimage is reliable for multi-channel
        img_4ch = io.imread(img_p) 
        if len(img_4ch.shape) < 3 or img_4ch.shape[2] < 4:
            raise ValueError(f"Image {area_id} channels error: {img_4ch.shape}")
        
        # Split RGB and IR
        img_rgb = img_4ch[:, :, :3] # 0,1,2
        img_ir = img_4ch[:, :, 3]   # 3
        
        # Read DSM (Handle potential float vs uint8)
        dsm = cv2.imread(dsm_p, cv2.IMREAD_UNCHANGED)
        
        # Read GT
        gt_rgb_raw = io.imread(gt_p)
        
    except Exception as e:
        print(f"Error reading area {area_id}: {e}")
        return 0

    # Size Alignment (Potsdam images sometimes have minor size mismatches)
    h, w = img_rgb.shape[:2]
    if dsm.shape[:2] != (h, w):
        dsm = cv2.resize(dsm, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Convert Mask
    mask = rgb_to_mask_fast(gt_rgb_raw, PALETTE)

    total_patches = 0
    is_train = (split_name == 'train')

    # ================= Strategy 1: Standard 512x512 Crop =================
    stride_std = STRIDE_STD_TRAIN if is_train else STRIDE_STD_VAL
    y_list = get_robust_grid_points(h, CROP_SIZE_STD, stride_std)
    x_list = get_robust_grid_points(w, CROP_SIZE_STD, stride_std)

    for y in y_list:
        for x in x_list:
            # Basic crop
            rgb_c = img_rgb[y:y+CROP_SIZE_STD, x:x+CROP_SIZE_STD]
            ir_c = img_ir[y:y+CROP_SIZE_STD, x:x+CROP_SIZE_STD]
            dsm_c = dsm[y:y+CROP_SIZE_STD, x:x+CROP_SIZE_STD]
            mask_c = mask[y:y+CROP_SIZE_STD, x:x+CROP_SIZE_STD]

            if rgb_c.shape[:2] != (CROP_SIZE_STD, CROP_SIZE_STD): continue

            # Save
            fname = f"potsdam_{area_id}_{y}_{x}.tif"
            if save_single_patch(rgb_c, ir_c, dsm_c, mask_c, fname, save_dirs):
                total_patches += 1

            # --- Class 4 Resampling (Train Only) ---
            if is_train:
                ratio = np.mean(mask_c == 4)
                if ratio >= CLASS4_THRESHOLD:
                    half_stride = stride_std // 2
                    # 4-Direction Resampling
                    shifts = [
                        (-half_stride, 0, "_shift_u"),
                        (half_stride, 0,  "_shift_d"),
                        (0, -half_stride, "_shift_l"),
                        (0, half_stride,  "_shift_r")
                    ]
                    for dy, dx, suffix in shifts:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny <= h - CROP_SIZE_STD and 0 <= nx <= w - CROP_SIZE_STD:
                            if save_single_patch(
                                img_rgb[ny:ny+CROP_SIZE_STD, nx:nx+CROP_SIZE_STD],
                                img_ir[ny:ny+CROP_SIZE_STD, nx:nx+CROP_SIZE_STD],
                                dsm[ny:ny+CROP_SIZE_STD, nx:nx+CROP_SIZE_STD],
                                mask[ny:ny+CROP_SIZE_STD, nx:nx+CROP_SIZE_STD],
                                f"potsdam_{area_id}_{y}_{x}{suffix}.tif",
                                save_dirs
                            ):
                                total_patches += 1

    # ================= Strategy 2: Large Crop 1024x1024 -> 512x512 (Train Only) =================
    if is_train and ENABLE_LARGE_CROP:
        y_list_lg = get_robust_grid_points(h, CROP_SIZE_LARGE, STRIDE_LARGE)
        x_list_lg = get_robust_grid_points(w, CROP_SIZE_LARGE, STRIDE_LARGE)

        for y in y_list_lg:
            for x in x_list_lg:
                rgb_lg = img_rgb[y:y+CROP_SIZE_LARGE, x:x+CROP_SIZE_LARGE]
                ir_lg = img_ir[y:y+CROP_SIZE_LARGE, x:x+CROP_SIZE_LARGE]
                dsm_lg = dsm[y:y+CROP_SIZE_LARGE, x:x+CROP_SIZE_LARGE]
                mask_lg = mask[y:y+CROP_SIZE_LARGE, x:x+CROP_SIZE_LARGE]

                if rgb_lg.shape[:2] != (CROP_SIZE_LARGE, CROP_SIZE_LARGE): continue

                # Scale Down (Note Interpolation methods)
                rgb_re = cv2.resize(rgb_lg, (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_LINEAR)
                ir_re = cv2.resize(ir_lg, (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_LINEAR)
                dsm_re = cv2.resize(dsm_lg, (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_LINEAR)
                mask_re = cv2.resize(mask_lg, (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_NEAREST)

                fname_lg = f"potsdam_{area_id}_{y}_{x}_scaled.tif"
                if save_single_patch(rgb_re, ir_re, dsm_re, mask_re, fname_lg, save_dirs):
                    total_patches += 1

                # --- Large Crop Class 4 Resampling ---
                ratio = np.mean(mask_lg == 4) # Calculate on 1024 scale
                if ratio >= CLASS4_THRESHOLD:
                    half_stride = STRIDE_LARGE // 2
                    shifts = [
                        (-half_stride, 0, "_shift_u_scaled"),
                        (half_stride, 0,  "_shift_d_scaled"),
                        (0, -half_stride, "_shift_l_scaled"),
                        (0, half_stride,  "_shift_r_scaled")
                    ]
                    for dy, dx, suffix in shifts:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny <= h - CROP_SIZE_LARGE and 0 <= nx <= w - CROP_SIZE_LARGE:
                            # Extract -> Resize -> Save
                            rgb_s = cv2.resize(img_rgb[ny:ny+CROP_SIZE_LARGE, nx:nx+CROP_SIZE_LARGE], (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_LINEAR)
                            ir_s = cv2.resize(img_ir[ny:ny+CROP_SIZE_LARGE, nx:nx+CROP_SIZE_LARGE], (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_LINEAR)
                            dsm_s = cv2.resize(dsm[ny:ny+CROP_SIZE_LARGE, nx:nx+CROP_SIZE_LARGE], (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_LINEAR)
                            mask_s = cv2.resize(mask[ny:ny+CROP_SIZE_LARGE, nx:nx+CROP_SIZE_LARGE], (TARGET_SIZE_LARGE, TARGET_SIZE_LARGE), interpolation=cv2.INTER_NEAREST)
                            
                            if save_single_patch(rgb_s, ir_s, dsm_s, mask_s, f"potsdam_{area_id}_{y}_{x}{suffix}.tif", save_dirs):
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
        
        # Create sub-directories: img(rgb), ir, dsm, mask
        sub_dirs = {
            'img': os.path.join(OUTPUT_ROOT, split, 'images'),
            'ir':  os.path.join(OUTPUT_ROOT, split, 'ir'),
            'dsm': os.path.join(OUTPUT_ROOT, split, 'dsm'),
            'mask': os.path.join(OUTPUT_ROOT, split, 'masks')
        }
        for p in sub_dirs.values():
            create_dir(p)
            
        task_list = [(aid, split, sub_dirs) for aid in ids]
        
        # Pre-check file existence
        valid_tasks = []
        for t in task_list:
            if os.path.exists(IMG_PATTERN.format(t[0])):
                valid_tasks.append(t)
            else:
                print(f"[Warning] Image file for Area {t[0]} not found, skipping.")
        
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