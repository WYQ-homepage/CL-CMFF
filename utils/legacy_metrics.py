"""Exact historical float32 evaluator, retained only as an audit reference."""
import torch

class LegacyEvaluator:

    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64, device=device)
        self.eps = 1e-06

    def calculate(self, valid_classes=None):
        """
        计算所有分割指标
        """
        cm = self.confusion_matrix.float()
        if valid_classes is not None and valid_classes < self.num_classes:
            cm = cm[:valid_classes, :valid_classes]
            curr_num_classes = valid_classes
        else:
            curr_num_classes = self.num_classes
        tp = torch.diag(cm)
        sum_pred = cm.sum(dim=0)
        sum_target = cm.sum(dim=1)
        total_pixels = cm.sum()
        union = sum_pred + sum_target - tp
        iou = tp / (union + self.eps)
        precision = tp / (sum_pred + self.eps)
        recall = tp / (sum_target + self.eps)
        f1 = 2 * precision * recall / (precision + recall + self.eps)
        miou = iou.mean().item() * 100
        mean_f1 = f1.mean().item() * 100
        mean_acc = recall.mean().item() * 100
        overall_acc = (tp.sum() / (total_pixels + self.eps)).item() * 100
        freq = sum_target / (total_pixels + self.eps)
        fwiou = (freq * iou).sum().item() * 100
        po = overall_acc / 100.0
        pe = (sum_target * sum_pred).sum() / (total_pixels ** 2 + self.eps)
        kappa = (po - pe) / (1 - pe + self.eps) * 100
        return {'iou_per_cls': iou * 100, 'f1_per_cls': f1 * 100, 'prec_per_cls': precision * 100, 'recall_per_cls': recall * 100, 'miou': miou, 'mean_f1': mean_f1, 'mean_acc': mean_acc, 'overall_acc': overall_acc, 'kappa': kappa.item(), 'fwiou': fwiou}

def calculate_legacy(cm, valid_classes):
    evaluator = LegacyEvaluator(len(cm), 'cpu')
    evaluator.confusion_matrix = torch.as_tensor(cm.copy(), dtype=torch.int64)
    result = evaluator.calculate(valid_classes)
    return {"miou": result["miou"], "mf1": result["mean_f1"], "oa": result["overall_acc"],
            "mean_recall": result["mean_acc"], "kappa": result["kappa"], "fwiou": result["fwiou"]}
