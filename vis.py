import os
import torch
import argparse
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

# --- 项目特定引用 ---
from utils import TrainConfig, LogManager
from model.core import Model
from val import SegmentationEvaluator as BaseEvaluator

# ==================== 1. 配色板定义 ====================
PALETTE_WHU = {
    0: (102, 204, 102),
    1: (204, 102, 102),
    2: (204, 204, 102),
    3: (102, 102, 204),
    4: (0, 153, 0),
    5: (153, 153, 153),
    6: (204, 102, 204),
    255: (0, 0, 0)
}

PALETTE_ISPRS = {
    0: (255, 255, 255), # Impervious surfaces
    1: (0, 0, 255),     # Buildings
    2: (0, 255, 255),   # Low vegetation
    3: (0, 255, 0),     # Trees
    4: (255, 255, 0),   # Cars
    5: (255, 0, 0),     # Clutter
    6: (0, 0, 0),       # Undefined
    255: (0, 0, 0)      # Ignore
}

def get_palette(dataset_name):
    name = dataset_name.lower()
    if 'whu' in name:
        return PALETTE_WHU
    elif 'vaihingen' in name or 'potsdam' in name:
        return PALETTE_ISPRS
    else:
        return PALETTE_ISPRS

def colorize_mask(mask, palette):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    unique_labels = np.unique(mask)
    for label in unique_labels:
        if label in palette:
            color_mask[mask == label] = palette[label]
        else:
            color_mask[mask == label] = (0, 0, 0)
    # OpenCV uses BGR
    color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)
    return color_mask

# ==================== 2. 评估核心类 ====================
class SegmentationEvaluator(BaseEvaluator):
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        self.device = device
        self.eps = 1e-6

    def update(self, preds, targets):
        preds = preds.flatten()
        targets = targets.flatten()
        # 忽略无效标签 (如255)
        mask = (targets >= 0) & (targets < self.num_classes)
        valid_preds = preds[mask]
        valid_targets = targets[mask]

        if len(valid_targets) > 0:
            indices = self.num_classes * valid_targets + valid_preds
            counts = torch.bincount(indices, minlength=self.num_classes**2)
            self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def calculate(self, valid_classes=None):
        metrics = super().calculate(valid_classes)
        full = super().calculate()
        return {
            "miou": metrics["miou"],
            "mf1": metrics["mean_f1"],
            "macc": metrics["mean_acc"],
            "oa": metrics["overall_acc"],
            "kappa": metrics["kappa"],
            "fwiou": metrics["fwiou"],
            "iou_per_class": full["iou_per_cls"].cpu().numpy(),
            "f1_per_class": full["f1_per_cls"].cpu().numpy(),
            "precision_per_class": full["prec_per_cls"].cpu().numpy(),
            "recall_per_class": full["recall_per_cls"].cpu().numpy(),
            "included_in_macro": np.arange(self.num_classes) < (self.num_classes if valid_classes is None else valid_classes),
            "confusion_matrix": self.confusion_matrix.cpu().numpy()
        }

# ==================== 3. 辅助函数 ====================
def load_weights_robust(model, weight_path, device, logger):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
    logger.info(f"Loading weights from: {weight_path}")

    checkpoint = torch.load(weight_path, map_location=device)

    # 兼容不同的保存格式
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint

    # 移除 DDP 的 module 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    logger.info("Weights loaded successfully (Strict Mode).")

def save_metrics_to_csv(metrics, output_dir, class_names=None):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 总体指标
    summary_data = {
        "Metric": ["mIoU", "mF1", "mAcc", "OA", "Kappa", "FWIoU"],
        "Value (%)": [metrics['miou'], metrics['mf1'], metrics['macc'],
                      metrics['oa'], metrics['kappa'], metrics['fwiou']]
    }
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(os.path.join(output_dir, "metrics_summary.csv"), index=False)

    # 2. 逐类指标
    num_classes = len(metrics['iou_per_class'])
    if class_names is None:
        class_names = [f"Class {i}" for i in range(num_classes)]
    else:
        # 确保 class_names 长度匹配
        if len(class_names) > num_classes:
            class_names = class_names[:num_classes]
        elif len(class_names) < num_classes:
            class_names += [f"Class {i}" for i in range(len(class_names), num_classes)]

    class_data = {
        "Class Name": class_names,
        "Included in Macro": metrics['included_in_macro'],
        "IoU": metrics['iou_per_class'],
        "F1": metrics['f1_per_class'],
        "Precision": metrics['precision_per_class'],
        "Recall": metrics['recall_per_class']
    }
    df_class = pd.DataFrame(class_data)
    df_class.to_csv(os.path.join(output_dir, "metrics_per_class.csv"), index=False)
    np.save(os.path.join(output_dir, "confusion_matrix.npy"), metrics['confusion_matrix'])
    print(f"[Result] Metrics saved to {output_dir}")

# ==================== 4. 主程序 ====================
def main():
    parser = argparse.ArgumentParser(description="Test and Visualize Script")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--weight", type=str, required=True, help="Path to .pth weight file")
    parser.add_argument("--output_dir", type=str, default="./results", help="Base directory to save results")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID")
    parser.add_argument("--mode", type=str, default="all", choices=["f1", "f2", "all"], help="Model mode for inference")
    # 优化点：使用 --no_vis 使得默认开启，加参数关闭
    parser.add_argument("--no_vis", action="store_true", help="Disable saving visualized images")
    args = parser.parse_args()

    # 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------ 1. 加载配置 ------------------
    config = TrainConfig.from_yaml(args.config)
    dataset_name = config.data.name

    # 输出目录结构
    weight_name = os.path.basename(args.weight).split('.')[0]
    final_output_dir = os.path.join(args.output_dir, dataset_name, f"{args.mode}_{weight_name}")
    vis_dir = os.path.join(final_output_dir, "vis")

    # 可视化开关逻辑
    save_vis = not args.no_vis
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    # 初始化日志
    logger = LogManager(f"Test_{dataset_name}", level="INFO").get_logger()
    logger.info(f"Config: {args.config}")
    logger.info(f"Dataset: {dataset_name} | Mode: {args.mode}")
    logger.info(f"Saving Results to: {final_output_dir}")

    # ------------------ 2. 数据准备 ------------------
    # 动态导入 Pipeline
    if dataset_name == "Vaihingen":
        from data_process.Vaihingen import PipLine
    elif dataset_name == "Potsdam":
        from data_process.Potsdam import PipLine
    elif dataset_name == "WHU":
        from data_process.WHU import PipLine
    elif dataset_name == "Levir_CS":
        from data_process.Levir_CS import PipLine
    else:
        raise ValueError(f"Unknown Dataset Name: {dataset_name}")

    # 初始化 Data Manager
    data_manager = PipLine(
        root_dir=config.data.root_dir,
        batch_size=1, # 推理强制单张
        num_workers=config.data.num_workers,
        pin_memory=getattr(config.data, "pin_memory", True),
        calculate_stats=True
    )
    test_loader = data_manager.get_loader(args.split, augment=False, shuffle=False)

    # 获取配色板
    current_palette = get_palette(dataset_name)

    # ------------------ 3. 模型构建 ------------------
    logger.info("Building Model...")
    model = Model.from_config(config.model, model_use=args.mode, load_pretrained=False).to(device)

    # 加载权重
    load_weights_robust(model, args.weight, device, logger)
    model.eval()

    # ------------------ 4. 评估初始化 ------------------
    num_classes = config.model.num_classes
    valid_classes = config.model.valid_classes
    evaluator = SegmentationEvaluator(num_classes, device)

    # 定义类别名称
    if dataset_name in ["Vaihingen", "Potsdam"]:
        class_names = ["Imp. Surf", "Building", "Low Veg", "Tree", "Car", "Clutter"]
    elif dataset_name == "WHU":
        class_names = ["Farmland", "City", "Village", "Water", "Forest", "Road", "Others", "Background"]
    else:
        class_names = [f"Class {i}" for i in range(num_classes)]

    # ------------------ 5. 推理循环 ------------------
    logger.info("Starting Inference...")
    loop = tqdm(test_loader, desc=f"Testing {dataset_name}", ncols=120)

    global_idx = 0
    with torch.inference_mode():
        for batch in loop:
            # 解析 Batch (PipLine 返回 rgb, aux, mask)
            if len(batch) == 3:
                rgb, x, labels = batch
            else:
                rgb, labels = batch
                x = None

            rgb = rgb.to(device)
            labels = labels.to(device).long()
            if x is not None:
                x = x.to(device)

            # 模型前向
            outputs = model(rgb, x)

            # 预测
            preds = torch.argmax(outputs, dim=1) # [B, H, W]

            # 更新指标
            evaluator.update(preds, labels)

            # 可视化保存
            if save_vis:
                batch_size = preds.shape[0]
                for b in range(batch_size):
                    pred_mask = preds[b].cpu().numpy().astype(np.uint8)
                    gt_mask = labels[b].cpu().numpy().astype(np.uint8)

                    # 映射颜色
                    color_pred = colorize_mask(pred_mask, current_palette)
                    color_gt = colorize_mask(gt_mask, current_palette)

                    # 合成文件名
                    save_name = f"frame_{global_idx:05d}"
                    cv2.imwrite(os.path.join(vis_dir, f"{save_name}_pred.png"), color_pred)
                    cv2.imwrite(os.path.join(vis_dir, f"{save_name}_gt.png"), color_gt)

                    global_idx += 1

    # ------------------ 6. 结果计算与保存 ------------------
    results = evaluator.calculate(valid_classes=valid_classes)

    print("\n" + "=" * 50)
    print(f"Dataset: {dataset_name} | Mode: {args.mode}")
    print(f"Final mIoU: {results['miou']:.2f}%")
    print(f"Final F1  : {results['mf1']:.2f}%")
    print(f"Final OA  : {results['oa']:.2f}%")
    print("=" * 50)

    save_metrics_to_csv(results, final_output_dir, class_names)
    logger.info(f"Done. Results saved to: {final_output_dir}")

if __name__ == "__main__":
    main()
