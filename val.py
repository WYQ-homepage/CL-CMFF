import os
import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- 引入项目依赖 ---
from model.core import Model
from utils import TrainConfig 

# ... [保留 SegmentationEvaluator 类和辅助函数不变] ...
class SegmentationEvaluator:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        self.eps = 1e-6 

    def update(self, preds, targets):
        preds = preds.flatten()
        targets = targets.flatten()
        mask = (targets >= 0) & (targets < self.num_classes)
        valid_preds = preds[mask]
        valid_targets = targets[mask]
        
        if len(valid_targets) > 0:
            indices = self.num_classes * valid_targets + valid_preds
            self.confusion_matrix += torch.bincount(
                indices, 
                minlength=self.num_classes**2
            ).reshape(self.num_classes, self.num_classes)

    def calculate(self, valid_classes=None):
        """
        计算所有分割指标
        """
        cm = self.confusion_matrix.double()
        
        # 如果只评估前 N 个类别 (例如忽略 Clutter)
        curr_num_classes = self.num_classes if valid_classes is None else valid_classes
        if not 1 <= curr_num_classes <= self.num_classes:
            raise ValueError("valid_classes must be between 1 and num_classes")
        
        # 基础统计量
        tp = torch.diag(cm)
        sum_pred = cm.sum(dim=0)      # 预测为该类的总数 (For Precision)
        sum_target = cm.sum(dim=1)    # 真实为该类的总数 (For Recall)
        total_pixels = cm.sum()

        # 1. Per-Class Metrics
        union = sum_pred + sum_target - tp
        iou = tp / (union + self.eps)
        
        precision = tp / (sum_pred + self.eps)
        recall = tp / (sum_target + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)

        # 2. Summary Metrics
        # Exclude classes only from macro averages, retaining cross-class errors.
        miou = iou[:curr_num_classes].mean().item() * 100
        mean_f1 = f1[:curr_num_classes].mean().item() * 100
        mean_acc = recall[:curr_num_classes].mean().item() * 100
        
        # Overall Accuracy (OA)
        overall_acc = (tp.sum() / (total_pixels + self.eps)).item() * 100
        
        # Frequency Weighted IoU (FWIoU)
        valid_support = sum_target[:curr_num_classes]
        freq = valid_support / (valid_support.sum() + self.eps)
        fwiou = (freq * iou[:curr_num_classes]).sum().item() * 100

        # Kappa Coefficient
        po = overall_acc / 100.0
        pe = (sum_target * sum_pred).sum() / (total_pixels**2 + self.eps)
        kappa = (po - pe) / (1 - pe + self.eps) * 100

        # 转换为百分比并返回
        return {
            "iou_per_cls": iou[:curr_num_classes] * 100,
            "f1_per_cls": f1[:curr_num_classes] * 100,
            "prec_per_cls": precision[:curr_num_classes] * 100,
            "recall_per_cls": recall[:curr_num_classes] * 100,
            "miou": miou,
            "mean_f1": mean_f1,
            "mean_acc": mean_acc,
            "overall_acc": overall_acc,
            "kappa": kappa.item(),
            "fwiou": fwiou
        }

# ================= 核心修改：简化验证逻辑 =================

def validate(model, loader, device, num_classes, valid_classes=None, is_main_process=True):
    model.eval()
    evaluator = SegmentationEvaluator(num_classes, device)
    
    val_loop = tqdm(loader, desc="Validating", leave=False, ncols=100) if is_main_process else loader

    with torch.inference_mode():
        for rgb, x, labels in val_loop:
            inputs_1, inputs_2 = rgb.to(device, non_blocking=True), x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            
            # 修改：直接获取输出，无需再判断是否为列表
            outputs = model(inputs_1, inputs_2)
            
            preds = torch.argmax(outputs, dim=1)
            evaluator.update(preds, labels)
    
    metrics = evaluator.calculate(valid_classes=valid_classes)
    return metrics.get("miou"), metrics.get("mean_f1"), metrics.get("iou_per_cls")

# ... [保留 load_weights_robust, print_evaluation_table, main 等逻辑不变] ...

def load_weights_robust(model, weight_path, device):
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weight file not found: {weight_path}")
        
    checkpoint = torch.load(weight_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    # 移除 DDP 的 'module.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict, strict=True)
        
    print(f"Loaded weights from {weight_path}")

def print_evaluation_table(metrics, class_names):
    """
    打印美观的评估结果表格
    """
    iou = metrics["iou_per_cls"]
    f1 = metrics["f1_per_cls"]
    prec = metrics["prec_per_cls"]
    recall = metrics["recall_per_cls"]
    
    print("="*80)
    print(f"{'Class Name':<15} | {'IoU (%)':<10} | {'F1 (%)':<10} | {'Prec (%)':<10} | {'Recall (%)':<10}")
    print("-" * 80)
    
    for i in range(len(iou)):
        c_name = class_names[i] if i < len(class_names) else f"Class {i}"
        print(f"{c_name:<15} | {iou[i]:<10.2f} | {f1[i]:<10.2f} | {prec[i]:<10.2f} | {recall[i]:<10.2f}")
        
    print("-" * 80)
    print(" [ Summary Metrics ]")
    print("-" * 80)
    print(f"{'Mean IoU':<15} : {metrics['miou']:.2f}%  (Avg of valid classes)")
    print(f"{'Mean F1':<15} : {metrics['mean_f1']:.2f}%")
    print(f"{'Mean Acc':<15} : {metrics['mean_acc']:.2f}%")
    print(f"{'Overall Acc':<15} : {metrics['overall_acc']:.2f}%")
    print(f"{'Kappa Coeff':<15} : {metrics['kappa']:.2f}%")
    print(f"{'FWIoU':<15} : {metrics['fwiou']:.2f}%")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description="Standalone Evaluation Script")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--weight", type=str, required=True, help="Path to model weight (.pth)")
    parser.add_argument("--gpu", type=str, default="0", help="GPU ID to use")
    parser.add_argument("--mode", type=str, default="all", choices=["f1", "f2", "all"], help="Model mode")
    args = parser.parse_args()

    # 1. 环境设置
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # 2. 加载配置
    config = TrainConfig.from_yaml(args.config)
    
    # 3. 数据准备
    if config.data.name == "Vaihingen":
        from data_process.Vaihingen import PipLine 
    elif config.data.name == "Potsdam":
        from data_process.Potsdam import PipLine 
    elif config.data.name == "WHU":
        from data_process.WHU import PipLine
    elif config.data.name == "Levir_CS":
        from data_process.Levir_CS import PipLine 
    
    calc_stats = getattr(config.data, 'calculate_stats', False)
    print(f"Data Pipeline: calculate_stats={calc_stats}")
    
    data_manager = PipLine(
        root_dir=config.data.root_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        pin_memory=getattr(config.data, "pin_memory", True),
        calculate_stats=calc_stats 
    )
    
    test_loader = data_manager.get_loader("test", augment=False, shuffle=False)

    # 4. 模型初始化
    print(f"Initializing Model with mode: {args.mode}")
    model = Model.from_config(config.model, model_use=args.mode, load_pretrained=False).to(device)

    # 5. 加载权重
    load_weights_robust(model, args.weight, device)

    # 6. 执行验证
    print("Starting Evaluation...")
    
    # 修正 validate 调用，返回完整的 metrics 字典以供打印
    model.eval()
    evaluator = SegmentationEvaluator(config.model.num_classes, device)
    val_loop = tqdm(test_loader, desc="Validating", leave=False, ncols=100)
    with torch.inference_mode():
        for rgb, x, labels in val_loop:
            inputs_1, inputs_2 = rgb.to(device, non_blocking=True), x.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            outputs = model(inputs_1, inputs_2)
            preds = torch.argmax(outputs, dim=1)
            evaluator.update(preds, labels)
    metrics = evaluator.calculate(valid_classes=config.model.valid_classes)

    # 7. 定义类别名称映射
    dataset_name = config.data.name
    class_map_dict = {
        "Vaihingen": ["Imp. Surf", "Building", "Low Veg", "Tree", "Car", "Clutter"],
        "Potsdam": ["Imp. Surf", "Building", "Low Veg", "Tree", "Car", "Clutter"],
        "WHU": ["Farmland", "City", "Village", "Water", "Forest", "Road", "Others", "Background"],
        "Levir_CS": ["Background", "Building"] 
    }
    
    if dataset_name in class_map_dict:
        class_names = class_map_dict[dataset_name]
    else:
        actual_num = len(metrics["iou_per_cls"])
        class_names = [f"Class {i}" for i in range(actual_num)]

    # 8. 打印结果表格
    print(f"\nEvaluation Results | Dataset: {dataset_name} | Mode: {args.mode}")
    print(f"Weight: {args.weight}\n")
    print_evaluation_table(metrics, class_names)

if __name__ == "__main__":
    main()
