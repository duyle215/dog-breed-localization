import numpy as np
import torch
from sklearn.metrics import f1_score

IOU_THRESHOLDS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def best_iou(pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """IoU of each predicted box against its best-matching GT box. pred (N,4), gt (N,K,4)."""
    p = pred[:, None]
    inter = (torch.min(p[..., 2], gt[..., 2]) - torch.max(p[..., 0], gt[..., 0])).clamp(0) * \
            (torch.min(p[..., 3], gt[..., 3]) - torch.max(p[..., 1], gt[..., 1])).clamp(0)
    area_p = (p[..., 2] - p[..., 0]).clamp(0) * (p[..., 3] - p[..., 1]).clamp(0)
    area_g = (gt[..., 2] - gt[..., 0]).clamp(0) * (gt[..., 3] - gt[..., 1]).clamp(0)
    iou = inter / (area_p + area_g - inter + 1e-7)
    return iou.masked_fill(~mask, 0).max(1).values


def summarize(labels: np.ndarray, preds: np.ndarray, ious: np.ndarray) -> dict:
    """Localization accuracy follows the ImageNet protocol: correct class AND IoU >= threshold."""
    correct = labels == preds
    loc = {f"loc_acc@{t:.2f}": float((correct & (ious >= t)).mean()) for t in IOU_THRESHOLDS}
    return {
        "top1_acc": float(correct.mean()),
        "macro_f1": float(f1_score(labels, preds, average="macro")),
        "mean_iou": float(ious.mean()),
        "iou@0.50": float((ious >= 0.5).mean()),
        "loc_acc@0.50": loc["loc_acc@0.50"],
        "loc_acc@0.75": loc["loc_acc@0.75"],
        "loc_acc@[.50:.95]": float(np.mean(list(loc.values()))),
    }


def bootstrap_ci(labels, preds, ious, key: str, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = np.arange(len(labels))
    vals = []
    for _ in range(n):
        s = rng.choice(idx, len(idx))
        vals.append(summarize(labels[s], preds[s], ious[s])[key])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
