import os
import cv2
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ================= 1. 配置区域 =================
RAW_DATA_ROOT = "./dataset/whu-opt-sar"
OUTPUT_ROOT = "./dataset/WHU_split" # 修改输出路径

# 原始文件路径模板
OPTICAL_PATTERN = os.path.join(RAW_DATA_ROOT, "optical/{}.tif")
SAR_PATTERN = os.path.join(RAW_DATA_ROOT, "sar/{}.tif")
LABEL_PATTERN = os.path.join(RAW_DATA_ROOT, "lbl/{}.tif")

# 数据集划分 IDs
TRAIN_IDS = [
    "NI49E021011", "NH49E008024", "NH49E014007", "NI49E022015", "NH49E006019", "NH49E010014", "NH50E011002", "NH49E006014",
    "NH50E010004", "NH49E001017", "NH50E009006", "NI49E022014", "NI49E021012", "NH49E013007", "NH50E012005", "NH49E005019",
    "NH49E013021", "NH50E012001", "NH49E007014", "NH49E012018", "NH50E006007", "NH49E013010", "NH49E006017", "NH49E013023",
    "NH49E003024", "NH49E002014", "NI49E022013", "NH49E005011", "NH50E013005", "NI49E023017", "NH49E007023", "NH49E001013",
    "NH49E005023", "NH50E007003", "NH49E010015", "NH49E012009", "NH49E004013", "NH49E009024", "NH50E007002", "NH49E001014",
    "NH49E011024", "NH49E009017", "NH49E012020", "NI49E024020", "NH49E008017", "NI49E021010", "NI49E024019", "NH50E011004",
    "NH49E011011", "NH50E012008", "NH49E007024", "NH50E014003", "NH49E006024", "NH50E011005", "NH49E013018", "NH49E006022",
    "NH50E010003", "NH49E005024", "NH50E009003", "NH49E002012",
]
VAL_IDS = [
    "NH49E008014", "NH49E014021", "NH50E012004", "NH49E006011", "NH49E014024", "NH49E007017", "NH50E003001", "NH50E008003",
    "NH49E010020", "NH50E009001", "NH49E005014", "NH49E010022", "NH50E013002", "NI49E022010", "NI49E024017", "NH49E003014",
    "NH49E008013", "NH49E012022", "NH49E014017", "NI49E024018",
]
TEST_IDS = [
    "NH49E007013", "NH49E009023", "NH49E004021", "NH49E014022", "NH49E011017", "NH50E006001", "NH49E009012", "NH49E009014",
    "NH50E009004", "NI49E022011", "NH50E016002", "NI49E024013", "NH50E007001", "NH50E010001", "NH49E012017", "NH49E004011",
    "NH50E007004", "NH49E010016", "NH49E002016", "NH50E013001",
]

# 标签映射表
LABEL_MAPPING = {
    0: 7,   # Background
    10: 0,  # Farmland
    20: 1,  # City
    30: 2,  # Village
    40: 3,  # Water
    50: 4,  # Forest
    60: 5,  # Road (TARGET FOR RESAMPLING)
    70: 6,  # Others
}

# 切片参数
CROP_SIZE = 512
STRIDE_TRAIN = 256
STRIDE_VAL_TEST = 512

# [修改] 重采样配置
RESAMPLE_CLASS = 5       # 需要重采样的类别ID (Road=5)
RESAMPLE_SHIFT = 128     # 重采样时的偏移量
RESAMPLE_MIN_RATIO = 0.2 # [新增] 只有当目标类别像素占比 > 20% 时才进行重采样 (0.2)

# ================= 2. 工具函数 =================

def create_dir(path):
    if not os.path.exists(path): os.makedirs(path)

def remap_label(label_img):
    new_label = np.full_like(label_img, 7, dtype=np.uint8)
    for orig, target in LABEL_MAPPING.items():
        if orig == 0: continue 
        new_label[label_img == orig] = target
    return new_label

def get_robust_grid_points(total_length, crop_size, stride):
    if total_length <= crop_size: return [0]
    points = []
    current = 0
    while current + crop_size <= total_length:
        points.append(current)
        current += stride
    if not points or points[-1] != total_length - crop_size:
        points.append(total_length - crop_size)
    return points

def save_patches(rgb_crop, ir_crop, sar_crop, mask_crop, filename, save_dirs):
    """保存切片的通用函数"""
    h, w = rgb_crop.shape[:2]
    if h != CROP_SIZE or w != CROP_SIZE: return

    cv2.imwrite(os.path.join(save_dirs['img'], filename), cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(save_dirs['ir'], filename), ir_crop)
    cv2.imwrite(os.path.join(save_dirs['sar'], filename), sar_crop)
    cv2.imwrite(os.path.join(save_dirs['mask'], filename), mask_crop)

def process_single_area(area_id, split_name, stride, save_dirs):
    opt_p = OPTICAL_PATTERN.format(area_id)
    sar_p = SAR_PATTERN.format(area_id)
    lbl_p = LABEL_PATTERN.format(area_id)

    if not (os.path.exists(opt_p) and os.path.exists(sar_p) and os.path.exists(lbl_p)):
        return 0

    # 1. 读取数据
    opt_img = cv2.imread(opt_p, cv2.IMREAD_UNCHANGED)
    if opt_img is None: raise ValueError(f"Read Error: {opt_p}")

    if len(opt_img.shape) == 3 and opt_img.shape[2] == 4:
        rgb = cv2.cvtColor(opt_img[:, :, :3], cv2.COLOR_BGR2RGB)
        ir = opt_img[:, :, 3]
    else:
        raise ValueError(f"Channel Error {area_id}: Expected 4 channels")

    sar = cv2.imread(sar_p, cv2.IMREAD_UNCHANGED)
    label = cv2.imread(lbl_p, cv2.IMREAD_UNCHANGED)
    
    # 2. 统一尺寸
    h, w = rgb.shape[:2]
    if sar.shape != (h, w): sar = cv2.resize(sar, (w, h), interpolation=cv2.INTER_LINEAR)
    if label.shape != (h, w): label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

    # 3. 标签映射
    mask = remap_label(label)

    # 4. 生成切片坐标
    y_list = get_robust_grid_points(h, CROP_SIZE, stride)
    x_list = get_robust_grid_points(w, CROP_SIZE, stride)

    count = 0

    for y in y_list:
        for x in x_list:
            # ============= 基础切片逻辑 =============
            mask_crop = mask[y:y+CROP_SIZE, x:x+CROP_SIZE]
            
            # [修改] 删除了背景过滤逻辑，所有切片都会被保存
            # 原有的 FILTER_BACKGROUND 代码块已移除

            # 准备基础数据
            rgb_crop = rgb[y:y+CROP_SIZE, x:x+CROP_SIZE]
            ir_crop = ir[y:y+CROP_SIZE, x:x+CROP_SIZE]
            sar_crop = sar[y:y+CROP_SIZE, x:x+CROP_SIZE]

            # 保存标准切片
            save_patches(rgb_crop, ir_crop, sar_crop, mask_crop, 
                         f"whu_{area_id}_{y}_{x}.tif", save_dirs)
            count += 1

            # ============= [修改] 条件重采样逻辑 =============
            # 仅在训练集启用
            if split_name == 'train':
                # 1. 计算目标类别像素数量
                target_pixels = np.count_nonzero(mask_crop == RESAMPLE_CLASS)
                total_pixels = mask_crop.size # CROP_SIZE * CROP_SIZE
                
                # 2. 计算占比
                ratio = target_pixels / total_pixels

                # 3. 判断是否满足重采样条件 (存在目标且占比大于阈值)
                if ratio > RESAMPLE_MIN_RATIO:
                    
                    # 定义额外的偏移坐标
                    extra_offsets = [
                        (x + RESAMPLE_SHIFT, y),
                        (x, y + RESAMPLE_SHIFT),
                        (x + RESAMPLE_SHIFT, y + RESAMPLE_SHIFT)
                    ]

                    for ex, ey in extra_offsets:
                        # 边界检查
                        if ex + CROP_SIZE <= w and ey + CROP_SIZE <= h:
                            
                            # 提取重采样切片
                            e_mask = mask[ey:ey+CROP_SIZE, ex:ex+CROP_SIZE]
                            e_rgb = rgb[ey:ey+CROP_SIZE, ex:ex+CROP_SIZE]
                            e_ir = ir[ey:ey+CROP_SIZE, ex:ex+CROP_SIZE]
                            e_sar = sar[ey:ey+CROP_SIZE, ex:ex+CROP_SIZE]

                            # [可选] 这里也可以加一个判定，确保偏移后的切片依然包含足够的道路
                            # if (np.count_nonzero(e_mask == RESAMPLE_CLASS) / e_mask.size) > RESAMPLE_MIN_RATIO:

                            save_patches(e_rgb, e_ir, e_sar, e_mask,
                                         f"whu_{area_id}_{ey}_{ex}_res.tif", save_dirs)
                            count += 1

    return count

def process_single_area_wrapper(args):
    """多进程包装器"""
    aid, split, stride, sub_dirs = args
    try:
        return process_single_area(aid, split, stride, sub_dirs)
    except Exception as e:
        print(f"\n[ERROR] Area {aid}: {e}")
        return 0

def process_dataset_multiprocessing():
    tasks_info = []
    # 配置任务
    configs = [
        (TRAIN_IDS, 'train', STRIDE_TRAIN),
        (VAL_IDS, 'val', STRIDE_VAL_TEST),
        (TEST_IDS, 'test', STRIDE_VAL_TEST)
    ]

    for ids, split, stride in configs:
        if not ids: continue
        
        # 创建子目录
        sub_dirs = {
            'img': os.path.join(OUTPUT_ROOT, split, 'images'),
            'ir':  os.path.join(OUTPUT_ROOT, split, 'ir'),
            'sar': os.path.join(OUTPUT_ROOT, split, 'sar'),
            'mask': os.path.join(OUTPUT_ROOT, split, 'masks')
        }
        for p in sub_dirs.values(): create_dir(p)
        
        for aid in ids:
            tasks_info.append((aid, split, stride, sub_dirs))

    num_processes = max(1, cpu_count() - 4)
    print(f"🚀 开启多进程切片: 使用 {num_processes} 个核心...")
    print(f"🎯 重采样策略: 类别 {RESAMPLE_CLASS}, 且像素占比 > {RESAMPLE_MIN_RATIO*100}%")

    total_patches = 0
    with Pool(processes=num_processes) as pool:
        for count in tqdm(pool.imap_unordered(process_single_area_wrapper, tasks_info), total=len(tasks_info)):
            total_patches += count
            
    print(f"\n✅ 全部完成！共生成切片: {total_patches}")
    print(f"📂 数据集保存在: {OUTPUT_ROOT}")

if __name__ == "__main__":
    process_dataset_multiprocessing()