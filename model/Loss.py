import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    def __init__(self, ignore_index=-100, reduction="mean"):
        super(CrossEntropyLoss, self).__init__()
        # 强制使用 'none'，以便我们在 forward 中拥有完全控制权（如应用 weight_map）
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits, targets, weight_map=None):
        # 1. 获取像素级 Loss (B, H, W)
        # 注意：targets 应为 (B, H, W)
        ce_loss = self.ce(logits, targets)

        # 2. 应用权重映射 (B, H, W)
        if weight_map is not None:
            ce_loss = ce_loss * weight_map

        # 3. 根据 reduction 参数手动处理归约
        if self.reduction == 'none':
            return ce_loss

        elif self.reduction == 'sum':
            return ce_loss.sum()

        elif self.reduction == 'mean':
            if self.ignore_index is not None:
                valid_mask = targets != self.ignore_index
                valid_pixels = valid_mask.sum().float()
                if valid_pixels > 0:
                    return ce_loss.sum() / valid_pixels
                else:
                    return ce_loss.sum()
            else:
                return ce_loss.mean()


class DistillationLoss(nn.Module):
    def __init__(self, temperature=4.0, ignore_index=255):
        super().__init__()
        self.temperature = temperature
        self.ignore_index = ignore_index
        self.kl_loss = nn.KLDivLoss(reduction="none")

    def forward(self, student_logits, teacher_logits, labels, weight_map=None):
        teacher_logits = teacher_logits.detach()

        # (B, C, H, W) -> (B, H, W, C)
        student_logits = student_logits.permute(0, 2, 3, 1)
        teacher_logits = teacher_logits.permute(0, 2, 3, 1)

        # LogSoftmax for Student, Softmax for Teacher
        student_soft = F.log_softmax(student_logits / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=-1)

        # KL Div
        kl_per_class = self.kl_loss(student_soft, teacher_soft) * (self.temperature**2)
        kl_per_pixel = kl_per_class.sum(dim=-1)  # (B, H, W)

        # Mask ignore_index
        mask = labels != self.ignore_index

        # Apply Weight Map
        if weight_map is not None:
            kl_per_pixel = kl_per_pixel * weight_map

        num_valid = mask.sum()
        if num_valid == 0:
            return kl_per_pixel[mask].sum()

        loss = kl_per_pixel[mask].sum() / (num_valid + 1e-8)
        return loss


class ModelLoss(nn.Module):
    def __init__(
        self,
        ignore_index=255,
        soft_temperature=4.0,
        main_weight=0.7,
        aux_weight=0.3,
        ce_weight=1.0,
        kd_weight=1,
        aux_weights=(0.75, 0.5, 0.25, 0.1),
        use_weight=True,
    ):
        super(ModelLoss, self).__init__()
        self.ignore_index = ignore_index
        self.main_weight = main_weight
        self.aux_weight = aux_weight

        self.ce_weight = ce_weight
        self.kd_weight = kd_weight

        self.aux_weights = []
        if aux_weights and len(aux_weights) > 0:
            s = float(sum(aux_weights))
            self.aux_weights = [w / s for w in aux_weights]

        self.use_weight = use_weight

        self.ce = CrossEntropyLoss(ignore_index=ignore_index)
        self.distill_loss = DistillationLoss(
            temperature=soft_temperature, ignore_index=ignore_index
        )
        self.soft_temperature = soft_temperature

    def _compute_weight_maps(self, teachers, target):
        """
        生成两套权重：
        1. main_weight: 用于 CE。在 Teacher 错误的地方变大 (Hard Mining)。
        2. distill_weights: 用于 KD。在 Teacher 错误的地方变小 (Noise Suppression)。
        """
        distill_weights = []
        teacher_confs = []
        teacher_errors = []

        for teacher_main in teachers:
            teacher_probs = torch.softmax(teacher_main / self.soft_temperature, dim=1)
            teacher_confidence, teacher_pred = torch.max(teacher_probs, dim=1)

            # 标记错误区域
            error_mask = teacher_pred != target
            teacher_errors.append(error_mask)
            teacher_confs.append(teacher_confidence)

            # --- KD 权重计算 ---
            # 逻辑：如果 Teacher 预测正确，则 KD 权重高；如果预测错误，则 KD 权重低。
            # 原始权重：正确(conf), 错误(-conf)
            raw_weight = (~error_mask) * teacher_confidence + error_mask * (-teacher_confidence)

            # 映射到 (0, 1) 区间，错误的地方接近 0
            # 使用 tanh 平滑处理
            distill_w = 0.5 + 0.5 * (
                torch.tanh(self.soft_temperature * raw_weight)
                / torch.tanh(self.soft_temperature * torch.tensor(1.0))
            )
            distill_weights.append(distill_w)

        # --- CE 权重计算 ---
        # 逻辑：聚合所有 Teacher 的错误。Teacher 错得越确信，说明该样本越难，CE 权重越大。
        n_teachers = len(teachers)
        avg_error_impact = torch.zeros_like(teacher_confs[0], dtype=torch.float32)

        for i in range(n_teachers):
            # 只有在 teacher 错了的时候加权重
            avg_error_impact += teacher_errors[i].float() * teacher_confs[i]

        avg_error_impact = avg_error_impact / n_teachers

        # 基础权重 1.0 + 平均错误置信度
        main_weight = 1.0 + avg_error_impact

        return main_weight, distill_weights

    def _prediction_loss(self, student, teachers, target):
        ce_weights = None
        kd_weights = [None] * len(teachers)
        if teachers and self.use_weight:
            ce_weights, kd_weights = self._compute_weight_maps(teachers, target)
        loss = self.ce_weight * self.ce(student, target, weight_map=ce_weights)
        if teachers and self.kd_weight != 0:
            kd = sum(self.distill_loss(student, teacher, target, weight)
                     for teacher, weight in zip(teachers, kd_weights)) / len(teachers)
            loss = loss + self.kd_weight * kd
        return loss

    def forward(self, student_outputs, target, teacher_outputs_list=None):
        """No teachers means CE only; use_weight disables both CE and KD maps."""
        teachers = [tuple(t.detach() for t in outputs)
                    for outputs in (teacher_outputs_list or [])]
        num_aux = min(len(student_outputs) - 1, len(self.aux_weights)) if self.training else 0
        if any(len(outputs) < 1 + num_aux for outputs in teachers):
            raise ValueError("Teacher outputs must include the corresponding main/aux branches")
        main = self._prediction_loss(student_outputs[0], [t[0] for t in teachers], target)
        aux = 0.0
        for i in range(num_aux):
            aux = aux + self.aux_weights[i] * self._prediction_loss(
                student_outputs[i + 1], [t[i + 1] for t in teachers], target)
        return self.main_weight * main + self.aux_weight * aux


class FeatureLoss(nn.Module):
    """特征对齐损失 - 直接比较学生和教师特征"""

    def __init__(self):
        super(FeatureLoss, self).__init__()

    def forward(self, student_features, teacher_features_list):
        total_loss = 0.0
        layer_num = 0

        for layer_idx in range(len(student_features)):
            s_feat = student_features[layer_idx]
            batch_size = s_feat.size(0)
            s_flat = s_feat.contiguous().view(batch_size, -1)
            s_flat = F.normalize(s_flat, p=2, dim=1)
            layer_losses = []
            for teacher_idx in range(len(teacher_features_list)):
                t_feat = teacher_features_list[teacher_idx][layer_idx]
                t_flat = t_feat.contiguous().view(batch_size, -1)
                t_flat = F.normalize(t_flat, p=2, dim=1)
                # 特征对齐：最小化特征向量的余弦距离
                cosine_sim = F.cosine_similarity(s_flat, t_flat, dim=1).mean()
                layer_losses.append(1 - cosine_sim)

            # 多教师损失平均
            layer_loss = torch.stack(layer_losses).mean()
            total_loss += layer_loss
            layer_num += 1

        return total_loss / layer_num
