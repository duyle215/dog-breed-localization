"""Reference points for the test split.

- mean box: always predict the average training box (shows how much localization is learned).
- ImageNet zero-shot: the pretrained EfficientNet-B0 classifier restricted to the 20 breeds.
  Stanford Dogs is a subset of ImageNet, so this shows how much the backbone already knows.

    python -m dogloc.baselines --config configs/default.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torchvision.models as tvm
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .data import DogDataset, class_names
from .metrics import best_iou, summarize
from .utils import load_config, load_splits


def _norm(name: str) -> str:
    return name.lower().replace("_", " ").replace("-", " ").strip()


@torch.no_grad()
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.overrides)
    d = cfg["data"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = load_splits(cfg)
    names = class_names(df)
    train_ds = DogDataset(df[df["split"] == "train"], d["root"], d["img_size"], d["target"], train=False)
    test_ds = DogDataset(df[df["split"] == "test"], d["root"], d["img_size"], d["target"], train=False)

    mean_box = torch.stack([train_ds[i][1] for i in range(len(train_ds))]).mean(0)

    weights = tvm.EfficientNet_B0_Weights.DEFAULT
    imagenet = {_norm(c): i for i, c in enumerate(weights.meta["categories"])}
    idx = [imagenet[_norm(n)] for n in names]
    clf = tvm.efficientnet_b0(weights=weights).eval().to(device)

    labels, preds, ious = [], [], []
    for images, _, y, gt, mask in tqdm(DataLoader(test_ds, batch_size=32), desc="baselines"):
        small = torch.nn.functional.interpolate(images, size=224, mode="bilinear", antialias=True)
        logits = clf(small.to(device))[:, idx]
        labels.append(y)
        preds.append(logits.argmax(1).cpu())
        ious.append(best_iou(mean_box.expand(len(y), 4), gt, mask))
    labels, preds, ious = (torch.cat(x).numpy() for x in (labels, preds, ious))

    zero_shot = summarize(labels, preds, ious)
    mean_box_only = summarize(labels, labels, ious)
    result = {
        "imagenet_zero_shot_top1_acc": zero_shot["top1_acc"],
        "imagenet_zero_shot_macro_f1": zero_shot["macro_f1"],
        "mean_box": [round(float(v), 4) for v in mean_box],
        "mean_box_mean_iou": mean_box_only["mean_iou"],
        "mean_box_iou@0.50": mean_box_only["iou@0.50"],
        "zero_shot_plus_mean_box_loc_acc@0.50": zero_shot["loc_acc@0.50"],
        "n_test_images": int(len(labels)),
    }
    out = Path(cfg["output_dir"]) / f"baselines_{d['target']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
