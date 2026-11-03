"""Aggregate test metrics over several runs (mean +- std across seeds).

    python -m dogloc.aggregate runs/largest_seed42 runs/largest_seed43 runs/largest_seed44 --out runs/summary/largest.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ["top1_acc", "macro_f1", "mean_iou", "iou@0.50", "loc_acc@0.50", "loc_acc@0.75", "loc_acc@[.50:.95]"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    per_run = {r: json.loads((Path(r) / "test_metrics.json").read_text()) for r in args.runs}
    summary = {"runs": list(per_run), "n_test_images": next(iter(per_run.values()))["n_images"]}
    for k in KEYS:
        vals = np.array([m[k] for m in per_run.values()])
        summary[k] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                      "values": vals.round(4).tolist()}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    for k in KEYS:
        print(f"{k:<20} {summary[k]['mean']*100:6.2f} +- {summary[k]['std']*100:.2f}")


if __name__ == "__main__":
    main()
