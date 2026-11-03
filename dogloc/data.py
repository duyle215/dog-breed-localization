from __future__ import annotations

import random
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import ImageReadMode, read_image
from torchvision.transforms import v2

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PAD_VALUE = 114
MAX_OBJECTS = 8


def parse_annotation(path: Path) -> dict | None:
    root = ET.parse(path).getroot()
    boxes = []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        boxes.append([int(bb.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")])
    if not boxes:
        return None
    return {
        "width": int(root.find("size/width").text),
        "height": int(root.find("size/height").text),
        "boxes": boxes,
    }


def build_index(data_root: str | Path) -> pd.DataFrame:
    """One row per image. Class folders follow the Stanford Dogs naming `<wnid>-<breed>`."""
    data_root = Path(data_root)
    img_dir, anno_dir = data_root / "img_resized", data_root / "anno_resized"
    rows = []
    for cls_dir in sorted(p for p in anno_dir.iterdir() if p.is_dir()):
        for anno in sorted(p for p in cls_dir.iterdir() if p.is_file()):
            img = img_dir / cls_dir.name / f"{anno.name}.jpg"
            parsed = parse_annotation(anno)
            if parsed is None or not img.exists():
                continue
            rows.append({
                "image": img.relative_to(data_root).as_posix(),
                "class_dir": cls_dir.name,
                "breed": cls_dir.name.split("-", 1)[1],
                **parsed,
            })
    df = pd.DataFrame(rows)
    class_dirs = sorted(df["class_dir"].unique())
    df["label"] = df["class_dir"].map({c: i for i, c in enumerate(class_dirs)})
    return df


def class_names(df: pd.DataFrame) -> list[str]:
    return df.drop_duplicates("label").sort_values("label")["breed"].tolist()


def _dhash(path: Path, size: int = 16) -> np.ndarray:
    # Grayscale on purpose: the dataset contains colour and grayscale copies of the same photo.
    im = Image.open(path).convert("L").resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def duplicate_groups(paths: list[Path], max_hamming: int) -> np.ndarray:
    """Group ids so that near-duplicate images (difference hash) share a group."""
    bits = np.stack([_dhash(p) for p in paths])
    n = len(paths)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n - 1):
        dist = (bits[i + 1:] != bits[i]).sum(1)
        for j in np.nonzero(dist <= max_hamming)[0] + i + 1:
            ri, rj = find(i), find(int(j))
            if ri != rj:
                parent[rj] = ri
    roots = [find(i) for i in range(n)]
    _, ids = np.unique(roots, return_inverse=True)
    return ids


def make_split(df: pd.DataFrame, fractions: tuple[float, float, float], seed: int) -> pd.Series:
    """Stratified by class, grouped by near-duplicate group so no group spans two splits."""
    rng = random.Random(seed)
    split = pd.Series("train", index=df.index)
    for _, cls_df in df.groupby("label"):
        groups = cls_df.groupby("group").groups
        keys = list(groups)
        rng.shuffle(keys)
        n = len(cls_df)
        n_val, n_test = round(n * fractions[1]), round(n * fractions[2])
        taken = {"test": 0, "val": 0}
        for g in keys:
            idx = groups[g]
            if taken["test"] < n_test:
                split.loc[idx] = "test"
                taken["test"] += len(idx)
            elif taken["val"] < n_val:
                split.loc[idx] = "val"
                taken["val"] += len(idx)
    return split


def load_or_create_split(df: pd.DataFrame, data_root: str | Path, split_file: str | Path,
                         fractions, seed: int, max_hamming: int) -> pd.DataFrame:
    split_file = Path(split_file)
    if split_file.exists():
        saved = pd.read_csv(split_file)
        df = df.merge(saved[["image", "group", "split"]], on="image", how="inner")
        return df
    df = df.copy()
    df["group"] = duplicate_groups([Path(data_root) / p for p in df["image"]], max_hamming)
    df["split"] = make_split(df, fractions, seed)
    split_file.parent.mkdir(parents=True, exist_ok=True)
    df[["image", "breed", "label", "group", "split"]].to_csv(split_file, index=False)
    return df


def build_transforms(img_size: int, train: bool) -> v2.Compose:
    common = [
        v2.Resize((img_size, img_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(MEAN, STD),
    ]
    if not train:
        return v2.Compose(common)
    return v2.Compose([
        v2.RandomHorizontalFlip(),
        v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), fill=PAD_VALUE),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        v2.RandomGrayscale(p=0.05),
        v2.ClampBoundingBoxes(),
        v2.SanitizeBoundingBoxes(min_size=4),
        *common,
    ])


def select_target(boxes: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "union":
        return torch.cat([boxes[:, :2].min(0).values, boxes[:, 2:].max(0).values])
    if mode == "largest":
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return boxes[area.argmax()]
    raise ValueError(f"unknown target mode: {mode}")


class DogDataset(Dataset):
    """Returns image, training target box, label, all GT boxes (padded) and their mask.

    Boxes are normalised xyxy in [0, 1]. In `union` mode the evaluation GT is the union box;
    in `largest` mode every annotated dog is a valid GT (ImageNet localization protocol).
    """

    def __init__(self, df: pd.DataFrame, data_root: str | Path, img_size: int,
                 target_mode: str, train: bool):
        self.df = df.reset_index(drop=True)
        self.root = Path(data_root)
        self.target_mode = target_mode
        self.train = train
        self.transform = build_transforms(img_size, train)
        self.eval_transform = build_transforms(img_size, False)

    def __len__(self) -> int:
        return len(self.df)

    def _apply(self, transform, image, boxes):
        out = transform({
            "image": image,
            "boxes": boxes,
            "labels": torch.zeros(len(boxes), dtype=torch.int64),
        })
        return out["image"], out["boxes"]

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        image = tv_tensors.Image(read_image(str(self.root / row["image"]), ImageReadMode.RGB))
        h, w = image.shape[-2:]
        boxes = tv_tensors.BoundingBoxes(
            torch.tensor(row["boxes"], dtype=torch.float32), format="XYXY", canvas_size=(h, w))

        img_t, boxes_t = self._apply(self.transform, image, boxes)
        if len(boxes_t) == 0:
            # Augmentation pushed every box out of frame; fall back to the clean view.
            img_t, boxes_t = self._apply(self.eval_transform, image, boxes)

        size = torch.tensor(img_t.shape[-2:][::-1], dtype=torch.float32).repeat(2)
        norm = boxes_t.as_subclass(torch.Tensor) / size
        target = select_target(norm, self.target_mode)
        gt = target[None] if self.target_mode == "union" else norm

        gt_pad = torch.zeros(MAX_OBJECTS, 4)
        mask = torch.zeros(MAX_OBJECTS, dtype=torch.bool)
        k = min(len(gt), MAX_OBJECTS)
        gt_pad[:k], mask[:k] = gt[:k], True
        return img_t.as_subclass(torch.Tensor), target, int(row["label"]), gt_pad, mask
