"""Figures for a finished run: test predictions (random and worst cases) and confusion matrix.

    python -m dogloc.visualize runs/largest_seed42 --out assets
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from .data import build_index


def _grid(rows: pd.DataFrame, gt: dict, names: list[str], root: Path, title: str, path: Path):
    n = len(rows)
    cols = 4
    fig, axes = plt.subplots((n + cols - 1) // cols, cols, figsize=(cols * 3.2, 3.4 * ((n + cols - 1) // cols)))
    for ax in axes.flat:
        ax.axis("off")
    for ax, (_, r) in zip(axes.flat, rows.iterrows()):
        im = Image.open(root / r["image"]).convert("RGB")
        w, h = im.size
        ax.imshow(im)
        for b in gt[r["image"]]:
            ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                           lw=2, ec="#22c55e", fc="none"))
        ax.add_patch(patches.Rectangle((r["x1"] * w, r["y1"] * h), (r["x2"] - r["x1"]) * w,
                                       (r["y2"] - r["y1"]) * h, lw=2, ec="#ef4444", fc="none", ls="--"))
        ok = r["label"] == r["pred"]
        ax.set_title(f"pred: {names[r['pred']]}\ntrue: {names[r['label']]} | IoU {r['iou']:.2f}",
                     fontsize=8, color="black" if ok else "#b91c1c")
    fig.suptitle(f"{title}  (green = ground truth, red dashed = prediction)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default="assets")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    run = Path(args.run_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    names, root = ckpt["classes"], Path(ckpt["config"]["data"]["root"])
    preds = pd.read_csv(run / "test_predictions.csv")
    gt = build_index(root).set_index("image")["boxes"].to_dict()

    _grid(preds.sample(12, random_state=args.seed), gt, names, root, "Random test images",
          out / "test_random.png")
    worst = preds.assign(score=(preds["label"] == preds["pred"]) * preds["iou"]).nsmallest(8, "score")
    _grid(worst, gt, names, root, "Hardest test images", out / "test_worst.png")

    cm = confusion_matrix(preds["label"], preds["pred"], labels=range(len(names)), normalize="true")
    fig, ax = plt.subplots(figsize=(11, 10))
    short = [n.replace("_", " ") for n in names]
    ConfusionMatrixDisplay(cm, display_labels=short).plot(ax=ax, cmap="Blues", values_format=".2f",
                                                          xticks_rotation=60, colorbar=False)
    ax.set_title("Confusion matrix (test, row-normalised)")
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=110)
    plt.close(fig)

    hist = pd.read_csv(run / "history.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    axes[0].plot(hist["epoch"], hist["loss"], label="train total")
    axes[0].plot(hist["epoch"], hist["cls_loss"], label="train cls")
    axes[0].plot(hist["epoch"], hist["box_loss"], label="train box")
    axes[0].set(xlabel="epoch", title="Training loss")
    for key, label in [("val_top1_acc", "top-1 acc"), ("val_mean_iou", "mean IoU"),
                       ("val_loc_acc@0.50", "loc acc@0.5"), ("val_loc_acc@[.50:.95]", "loc acc@[.5:.95]")]:
        axes[1].plot(hist["epoch"], hist[key], label=label)
    axes[1].set(xlabel="epoch", title="Validation (EMA weights)", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=110)
    plt.close(fig)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
