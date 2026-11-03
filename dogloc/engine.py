import numpy as np
import torch
from tqdm.auto import tqdm

from .metrics import best_iou


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, ema, device,
                    grad_clip: float) -> dict:
    model.train()
    totals = np.zeros(3)
    for images, target, labels, _, _ in tqdm(loader, desc="train", leave=False):
        images, target, labels = images.to(device), target.to(device), labels.to(device)
        with torch.autocast(device.type, enabled=scaler.is_enabled()):
            logits, box = model(images)
        loss, cls, reg = criterion(logits, box, labels, target)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        ema.update_parameters(model)
        totals += [loss.item(), cls.item(), reg.item()]
    totals /= len(loader)
    return {"loss": totals[0], "cls_loss": totals[1], "box_loss": totals[2]}


@torch.no_grad()
def predict(model, loader, device, amp: bool) -> dict:
    model.eval()
    out = {k: [] for k in ("labels", "preds", "probs", "boxes", "ious")}
    for images, _, labels, gt, mask in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        with torch.autocast(device.type, enabled=amp):
            logits, box = model(images)
        box = box.float().cpu()
        out["labels"].append(labels)
        out["probs"].append(logits.float().softmax(1).cpu())
        out["preds"].append(logits.argmax(1).cpu())
        out["boxes"].append(box)
        out["ious"].append(best_iou(box, gt, mask))
    return {k: torch.cat(v).numpy() for k, v in out.items()}
