"""Train, select the best epoch on validation, then evaluate once on the test split.

    python -m dogloc.train --config configs/default.yaml [section.key=value ...]
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from .data import class_names
from .engine import predict, train_one_epoch
from .losses import LocalizationLoss
from .metrics import bootstrap_ci, summarize
from .model import DogLocalizer
from .utils import load_config, load_splits, make_loader, seed_everything

SELECT_METRIC = "loc_acc@[.50:.95]"


def evaluate_split(model, df, cfg, split, device, out_dir: Path, names: list[str]) -> dict:
    loader = make_loader(df, cfg, split, train=False)
    res = predict(model, loader, device, cfg["train"]["amp"])
    metrics = summarize(res["labels"], res["preds"], res["ious"])
    for key in ("top1_acc", "mean_iou", "loc_acc@0.50"):
        metrics[f"{key}_ci95"] = bootstrap_ci(res["labels"], res["preds"], res["ious"], key)
    metrics["n_images"] = int(len(res["labels"]))

    per_class = []
    for c, name in enumerate(names):
        m = res["labels"] == c
        per_class.append({"breed": name, "n": int(m.sum()),
                          **summarize(res["labels"][m], res["preds"][m], res["ious"][m])})
    pd.DataFrame(per_class).drop(columns="macro_f1").to_csv(out_dir / f"{split}_per_class.csv", index=False)

    split_df = df[df["split"] == split].reset_index(drop=True)
    pd.DataFrame({
        "image": split_df["image"], "label": res["labels"], "pred": res["preds"],
        "conf": res["probs"].max(1), "iou": res["ious"],
        "x1": res["boxes"][:, 0], "y1": res["boxes"][:, 1],
        "x2": res["boxes"][:, 2], "y2": res["boxes"][:, 3],
    }).to_csv(out_dir / f"{split}_predictions.csv", index=False)
    (out_dir / f"{split}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--name", default=None, help="run folder name under output_dir")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args(argv)

    cfg = load_config(args.config, args.overrides)
    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t, m = cfg["train"], cfg["model"]

    name = args.name or f"{cfg['data']['target']}_seed{cfg['seed']}"
    out_dir = Path(cfg["output_dir"]) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_splits(cfg)
    names = class_names(df)
    train_loader = make_loader(df, cfg, "train", train=True, seed=cfg["seed"])
    val_loader = make_loader(df, cfg, "val", train=False)
    print(f"splits: {df['split'].value_counts().to_dict()} | device: {device}")

    model = DogLocalizer(len(names), m["bifpn_channels"], m["bifpn_repeats"]).to(device)
    groups = model.param_groups(t["lr_backbone"], t["lr_head"], t["lr_box"])
    optimizer = torch.optim.AdamW(groups, weight_decay=t["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups], epochs=t["epochs"],
        steps_per_epoch=len(train_loader), pct_start=t["warmup_pct"])
    scaler = torch.amp.GradScaler(device.type, enabled=t["amp"] and device.type == "cuda")
    ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(t["ema_decay"]), use_buffers=True)
    criterion = LocalizationLoss(t["lambda_box"], t["lambda_l1"], t["label_smoothing"])

    history, best = [], -1.0
    for epoch in range(1, t["epochs"] + 1):
        start = time.time()
        tr = train_one_epoch(model, train_loader, criterion, optimizer, scheduler, scaler, ema,
                             device, t["grad_clip"])
        res = predict(ema.module, val_loader, device, t["amp"])
        val = summarize(res["labels"], res["preds"], res["ious"])
        history.append({"epoch": epoch, **tr, **{f"val_{k}": v for k, v in val.items()},
                        "seconds": round(time.time() - start, 1)})
        marker = ""
        if val[SELECT_METRIC] > best:
            best = val[SELECT_METRIC]
            torch.save({"model": ema.module.state_dict(), "classes": names, "config": cfg,
                        "epoch": epoch}, out_dir / "best.pt")
            marker = " *"
        print(f"epoch {epoch:2d}/{t['epochs']} | loss {tr['loss']:.3f} | "
              f"val acc {val['top1_acc']:.3f} iou {val['mean_iou']:.3f} "
              f"loc@.5 {val['loc_acc@0.50']:.3f} loc@[.5:.95] {val[SELECT_METRIC]:.3f} | "
              f"{history[-1]['seconds']}s{marker}", flush=True)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)

    ckpt = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    test = evaluate_split(model, df, cfg, "test", device, out_dir, names)
    print(f"best epoch {ckpt['epoch']} | test: " +
          ", ".join(f"{k}={v:.4f}" for k, v in test.items() if isinstance(v, float)))


if __name__ == "__main__":
    main()
