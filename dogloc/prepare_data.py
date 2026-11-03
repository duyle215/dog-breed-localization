"""Letterbox the 20 Stanford Dogs breeds used here to 512x512 and rescale their boxes.

Download `images.tar` and `annotation.tar` from http://vision.stanford.edu/aditya86/ImageNetDogs/
and extract them, then:

    python -m dogloc.prepare_data --src path/to/stanford_dogs --dst data
"""
import argparse
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

PAD_COLOR = (114, 114, 114)


def _geometry(w: int, h: int, size: int):
    s = min(size / w, size / h)
    new_w, new_h = round(w * s), round(h * s)
    return s, new_w, new_h, (size - new_w) // 2, (size - new_h) // 2


def letterbox_pair(img_src: Path, anno_src: Path, img_dst: Path, anno_dst: Path, size: int) -> None:
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    anno_dst.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(img_src) as im:
        im = im.convert("RGB")
        s, new_w, new_h, left, top = _geometry(*im.size, size)
        canvas = Image.new("RGB", (size, size), PAD_COLOR)
        canvas.paste(im.resize((new_w, new_h), Image.LANCZOS), (left, top))
        canvas.save(img_dst, quality=95)

    tree = ET.parse(anno_src)
    root = tree.getroot()
    w, h = int(root.find("size/width").text), int(root.find("size/height").text)
    s, _, _, left, top = _geometry(w, h, size)
    root.find("size/width").text = root.find("size/height").text = str(size)
    for bb in root.iter("bndbox"):
        for key, offset in (("xmin", left), ("ymin", top), ("xmax", left), ("ymax", top)):
            bb.find(key).text = str(round(int(bb.find(key).text) * s) + offset)
    tree.write(anno_dst)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder containing Images/ and Annotation/")
    ap.add_argument("--dst", default="data")
    ap.add_argument("--split-file", default="splits/split_seed42.csv")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args(argv)

    src, dst = Path(args.src), Path(args.dst)
    classes = sorted({Path(p).parent.name for p in pd.read_csv(args.split_file)["image"]})
    tasks = []
    for cls in classes:
        for anno in sorted((src / "Annotation" / cls).iterdir()):
            img = src / "Images" / cls / f"{anno.name}.jpg"
            if img.exists():
                tasks.append((img, anno, dst / "img_resized" / cls / img.name,
                              dst / "anno_resized" / cls / anno.name, args.size))
    with ThreadPoolExecutor() as pool:
        list(pool.map(lambda t: letterbox_pair(*t), tasks))
    print(f"{len(tasks)} images from {len(classes)} breeds written to {dst}")


if __name__ == "__main__":
    main()
