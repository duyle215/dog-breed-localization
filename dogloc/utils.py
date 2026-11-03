import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from .data import DogDataset, build_index, load_or_create_split


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """YAML config with optional `section.key=value` overrides from the command line."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits(cfg: dict):
    d = cfg["data"]
    df = build_index(d["root"])
    df = load_or_create_split(df, d["root"], d["split_file"], tuple(d["split"]),
                              d["split_seed"], d["dedup_max_hamming"])
    return df


def make_loader(df, cfg: dict, split: str, train: bool, seed: int = 0) -> DataLoader:
    d, t = cfg["data"], cfg["train"]
    ds = DogDataset(df[df["split"] == split], d["root"], d["img_size"], d["target"], train=train)
    g = torch.Generator().manual_seed(seed)
    return DataLoader(
        ds, batch_size=t["batch_size"], shuffle=train, drop_last=train, generator=g,
        num_workers=t["num_workers"], pin_memory=True,
        persistent_workers=t["num_workers"] > 0,
    )
