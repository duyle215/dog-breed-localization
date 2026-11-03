import math

import torch
import torch.nn.functional as F


def ciou_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Complete-IoU loss for normalised xyxy boxes: 1 - (IoU - rho^2/c^2 - alpha*v)."""
    px1, py1, px2, py2 = pred.unbind(1)
    tx1, ty1, tx2, ty2 = target.unbind(1)

    inter = (torch.min(px2, tx2) - torch.max(px1, tx1)).clamp(0) * \
            (torch.min(py2, ty2) - torch.max(py1, ty1)).clamp(0)
    pw, ph = (px2 - px1).clamp(min=eps), (py2 - py1).clamp(min=eps)
    tw, th = (tx2 - tx1).clamp(min=eps), (ty2 - ty1).clamp(min=eps)
    iou = inter / (pw * ph + tw * th - inter + eps)

    c2 = (torch.max(px2, tx2) - torch.min(px1, tx1)) ** 2 + \
         (torch.max(py2, ty2) - torch.min(py1, ty1)) ** 2 + eps
    rho2 = ((px1 + px2 - tx1 - tx2) ** 2 + (py1 + py2 - ty1 - ty2) ** 2) / 4

    v = (4 / math.pi ** 2) * (torch.atan(tw / th) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)
    return (1 - (iou - rho2 / c2 - alpha * v)).mean()


class LocalizationLoss:
    def __init__(self, lambda_box: float, lambda_l1: float, label_smoothing: float):
        self.lambda_box = lambda_box
        self.lambda_l1 = lambda_l1
        self.label_smoothing = label_smoothing

    def __call__(self, logits, box, labels, target_box):
        # Computed in fp32: atan and small box differences lose precision under fp16 autocast.
        logits, box = logits.float(), box.float()
        cls = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        reg = ciou_loss(box, target_box) + self.lambda_l1 * F.smooth_l1_loss(box, target_box)
        return cls + self.lambda_box * reg, cls.detach(), reg.detach()
