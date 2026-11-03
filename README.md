# Dog Breed Localization

Single-object localization and classification on **20 breeds of the Stanford Dogs dataset**: for each image the model predicts the breed and one bounding box around the dog. The model is an ImageNet-pretrained **EfficientNet-B0** backbone with a **BiFPN** neck, a classification head, and a box-regression head trained with **CIoU loss**.

The project started as a university deep-learning assignment (`notebooks/ver2.ipynb`). This repository refactors it into a package, audits the data, and fixes how the results are measured.

## Results

Test split: 357 images (about 10% per breed). Localization accuracy follows the ImageNet protocol: a prediction is correct only if the **breed is right and** the box has IoU ≥ threshold with an annotated dog.

| Model | Top-1 acc | Mean box IoU | Loc. acc @0.5 |
| --- | --- | --- | --- |
| Baseline: ImageNet EfficientNet-B0, no training, restricted to the 20 breeds | 97.2% | – | – |
| Baseline: always predict the mean training box | – | 0.536 | 64.7%¹ |
| Original notebook (random split, union box target) | 96.6% | 0.870 | 95.7%² |
| This pipeline (3 seeds) | *training in progress* | | |

¹ Combined with the zero-shot classifier. ² Macro average over breeds.

**How to read this:**
- **Classification:** Stanford Dogs is a subset of ImageNet, and the backbone was pretrained on ImageNet, so it already recognises these breeds (97.2% without any training). Breed accuracy therefore says little about what this project adds.
- **Localization:** this is where the trained model contributes, raising box IoU from 0.54 for a fixed box to about 0.87.

## Data audit

Checks run before training (rendered sample grids, annotation statistics, perceptual hashing):

- **3,537 images, 20 breeds, 151–239 images per breed.** Every image has a valid XML annotation and no box is malformed.
- **Near-duplicate photos:** 20 pairs, e.g. the same photo in colour and grayscale, or slightly re-cropped. A random split can put the two copies in train and test. They are now grouped (difference hash, Hamming distance ≤ 32 of 256 bits; matches were checked visually), and each group stays in a single split.
- **Multiple dogs in one image:** 7.5% of images (267). The original notebook trained on the union of all boxes, which on these images is 17% background on average (36% at the 90th percentile). The model now learns the **largest dog**, and evaluation accepts a match with **any** annotated dog. The old behaviour is still available with `data.target=union`.

## Method

```
image 512×512 ─► EfficientNet-B0 ─► C3 (s8), C4 (s16), C5 (s32)
                                   │  1×1 lateral convs → 64 ch
                                   ▼
                             BiFPN × 3 (fast normalised fusion)
                     ┌─────────────┴──────────────┐
          GAP(P3,P4,P5) → MLP → 20 logits   P3 → 4× strided conv → 4×4 map ⊕ GAP(P4,P5) → MLP → (cx, cy, w, h)
```

- **Loss:** cross-entropy (label smoothing 0.1) + 5 × (CIoU + Smooth-L1). The box loss is computed in fp32 because CIoU's `atan` term and small box differences lose precision under fp16.
- **Augmentation:** horizontal flip, random affine (±10% translation, 0.8–1.2 scale), colour jitter, and random grayscale, with boxes transformed alongside the image (`torchvision.transforms.v2`).
- **Optimisation:** AdamW with three learning-rate groups (backbone 1e-4, heads 1e-3, box head 2e-3 at peak), OneCycle schedule, mixed precision, gradient clipping, and an EMA of the weights (decay 0.998) used for evaluation.
- **Model selection:** the epoch with the best validation loc. acc @[.50:.95], which rewards correct breeds and tight boxes together. The test split is evaluated once, at the end.
- **Reporting:** 95% bootstrap confidence intervals on test metrics, and mean ± std over three training seeds on the same split.

### Changes from the original notebook

| Area | Original | Now |
| --- | --- | --- |
| Split | random per class | per class, near-duplicates kept together; saved to `splits/` |
| Multi-dog target | union of all boxes | largest dog; any dog counts as correct |
| Metrics | "mIoU", "AP50" (actually accuracy at IoU ≥ 0.5) | mean box IoU, loc. acc @0.5 / @0.75 / @[.50:.95], with CIs |
| Baselines | none | ImageNet zero-shot, mean box |
| Class imbalance | weighted loss **and** weighted sampler | neither (classes are close to balanced) |
| Augmentation | photometric + flip | + scale / translate with box updates |
| Training | best val loss, weights kept in memory | EMA weights, checkpoint and history saved per run |
| Code | one notebook with a Windows DLL workaround | `dogloc/` package, YAML config, CLI scripts |

## Getting started

```bash
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Download `images.tar` and `annotation.tar` from the [Stanford Dogs page](http://vision.stanford.edu/aditya86/ImageNetDogs/), extract them, and letterbox the 20 breeds to 512×512:

```bash
python -m dogloc.prepare_data --src path/to/stanford_dogs --dst data
```

Run everything (baselines, three seeds, aggregated metrics, figures):

```bash
PYTHON=python bash scripts/reproduce.sh
```

or the steps one at a time:

```bash
python -m dogloc.baselines
python -m dogloc.train --name my_run seed=42 train.epochs=20   # any config key can be overridden
python -m dogloc.visualize runs/my_run --out assets
```

On a 4 GB laptop GPU (RTX 2050), one epoch takes about 2.3 minutes at 512 px with batch size 8.

## Project structure

```
configs/default.yaml      # all hyper-parameters
dogloc/
  data.py                 # annotation parsing, near-duplicate grouping, split, dataset, transforms
  model.py                # EfficientNet-B0 + BiFPN + classification and box heads
  losses.py               # CIoU + Smooth-L1 + cross-entropy
  metrics.py              # best-match IoU, localization accuracy, bootstrap CIs
  engine.py               # train / predict loops
  train.py                # training entry point (select on val, evaluate on test)
  baselines.py            # ImageNet zero-shot and mean-box baselines
  aggregate.py            # mean ± std across seeds
  visualize.py            # prediction grids, confusion matrix, curves
  prepare_data.py         # letterbox raw Stanford Dogs
scripts/reproduce.sh
splits/split_seed42.csv   # fixed train / val / test assignment
notebooks/                # original EDA and training notebooks
```

## Data and licence

Images and annotations come from the [Stanford Dogs dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) (Khosla et al., 2011), built from ImageNet. They are not redistributed here; download them from the official page.
