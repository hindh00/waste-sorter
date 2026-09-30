"""
prepare_dataset.py

Turns TACO's COCO-format detection/segmentation annotations into an
ImageFolder-style classification dataset that Ultralytics YOLO-cls can
train on directly: one folder per consolidated category, split into
train/ and val/.

TACO (https://github.com/pedropro/TACO) labels every annotation with one
of ~60 fine-grained categories, far too many (and too imbalanced) to train
directly. data/category_map.yaml collapses those into the small set this
app actually uses (see rules.yaml) — edit that YAML file, not this script,
if you want to add/rename/regroup categories.

Expects the raw TACO repo layout:
    <taco-root>/
      data/
        annotations.json
        batch_1/*.jpg
        batch_2/*.jpg
        ...
(TACO's annotations.json stores each image's "file_name" as a path
relative to that data/ folder, e.g. "batch_1/000006.jpg".)

Robustness notes (read before running on a real TACO download):
  - TACO images are Flickr-hosted and downloaded separately by TACO's own
    download.py; some will be missing or corrupted locally. This script
    skips those with a logged reason instead of crashing the whole run.
  - Splitting is done PER CATEGORY, not as one global shuffle — a single
    global split can wipe a rare category (e.g. battery_hazardous) out of
    the validation set entirely.
  - Cigarette butts are frequently tiny in the original TACO photos; crops
    under --min-crop-size are skipped and counted separately so you can
    see how much of that class actually survives.

Usage:
    python scripts/prepare_dataset.py \\
        --taco-root data/raw_taco \\
        --category-map data/category_map.yaml \\
        --output-dir data/classification \\
        --val-ratio 0.15
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image


def load_category_map(path: Path) -> dict[str, str]:
    """Loads category.yaml's {category: [taco labels]} shape and inverts it
    into a flat {taco label: category} lookup for per-annotation use."""
    with path.open("r") as f:
        grouped: dict[str, list[str]] = yaml.safe_load(f) or {}

    flat: dict[str, str] = {}
    for category, taco_labels in grouped.items():
        for label in taco_labels:
            if label in flat:
                warnings.warn(
                    f"TACO label '{label}' is mapped to both "
                    f"'{flat[label]}' and '{category}' — using '{category}'."
                )
            flat[label] = category
    return flat


def crop_instance(img: Image.Image, bbox: list[float], pad_frac: float) -> Image.Image | None:
    """Returns None instead of cropping when the box is degenerate after
    clamping to image bounds — seen on a handful of TACO annotations whose
    recorded width/height don't match the actual downloaded image (e.g. a
    portrait/landscape EXIF mismatch), which can push the box outside the
    image entirely."""
    x, y, w, h = bbox
    pad_x, pad_y = w * pad_frac, h * pad_frac
    left = max(0, int(x - pad_x))
    top = max(0, int(y - pad_y))
    right = min(img.width, int(x + w + pad_x))
    bottom = min(img.height, int(y + h + pad_y))
    if left >= right or top >= bottom:
        return None
    return img.crop((left, top, right, bottom))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--taco-root", type=Path, required=True,
                         help="Path to the cloned+downloaded TACO repo (contains data/annotations.json)")
    parser.add_argument("--annotations-file", type=Path, default=None,
                         help="Override path to annotations.json (default: <taco-root>/data/annotations.json)")
    parser.add_argument("--category-map", type=Path, default=Path("data/category_map.yaml"),
                         help="Path to the TACO-label -> consolidated-category YAML mapping")
    parser.add_argument("--output-dir", type=Path, default=Path("data/classification"),
                         help="Output folder for the classification dataset")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                         help="Fraction of each category's crops held out for validation")
    parser.add_argument("--pad-frac", type=float, default=0.08,
                         help="Padding around each crop, as a fraction of its own width/height")
    parser.add_argument("--min-crop-size", type=int, default=20,
                         help="Skip crops smaller than this (px) on either side after padding")
    parser.add_argument("--min-cigarette-size", type=int, default=32,
                         help="Separate, usually-stricter minimum crop size just for 'cigarette' "
                              "(TACO cigarette annotations are frequently tiny)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from pycocotools.coco import COCO
    except ImportError:
        sys.exit(
            "pycocotools is not installed. Install project requirements first:\n"
            "  pip install -r requirements.txt\n"
            "(and make sure you're using a Python 3.11/3.12 venv, not system Python — see README.md)"
        )

    ann_path = args.annotations_file or (args.taco_root / "data" / "annotations.json")
    img_root = args.taco_root / "data"

    if not ann_path.exists():
        sys.exit(
            f"Could not find {ann_path}.\n"
            "Have you cloned TACO and run its own download.py yet? See README.md."
        )

    category_map = load_category_map(args.category_map)
    coco = COCO(str(ann_path))
    cat_id_to_name = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}

    random.seed(args.seed)

    # Group annotation ids by our consolidated category so each category
    # can be split independently — a global shuffle can accidentally erase
    # a rare class from the validation set entirely.
    by_category: dict[str, list[int]] = defaultdict(list)
    unmapped_labels: set[str] = set()

    for ann_id in coco.getAnnIds():
        ann = coco.loadAnns(ann_id)[0]
        taco_label = cat_id_to_name[ann["category_id"]]
        category = category_map.get(taco_label)
        if category is None:
            unmapped_labels.add(taco_label)
            category = "other_unlabeled"
        by_category[category].append(ann_id)

    if unmapped_labels:
        print(
            "WARNING: the following TACO labels are not in "
            f"{args.category_map} and were bucketed into 'other_unlabeled':\n  "
            + ", ".join(sorted(unmapped_labels))
        )

    counts: dict[str, dict[str, int]] = {}
    skipped_rows: list[dict] = []
    cigarette_skipped_small = 0

    for category, ann_ids in by_category.items():
        shuffled = ann_ids[:]
        random.shuffle(shuffled)
        n_val = max(1, round(len(shuffled) * args.val_ratio)) if len(shuffled) > 1 else 0
        splits = {"val": shuffled[:n_val], "train": shuffled[n_val:]}
        written = {"train": 0, "val": 0}

        for split, ids in splits.items():
            out_dir = args.output_dir / split / category
            out_dir.mkdir(parents=True, exist_ok=True)

            for ann_id in ids:
                ann = coco.loadAnns(ann_id)[0]
                bbox = ann["bbox"]  # COCO format: [x, y, w, h]

                min_size = args.min_cigarette_size if category == "cigarette" else args.min_crop_size
                if bbox[2] < min_size or bbox[3] < min_size:
                    if category == "cigarette":
                        cigarette_skipped_small += 1
                    skipped_rows.append({"ann_id": ann_id, "category": category, "reason": "too_small"})
                    continue

                img_info = coco.loadImgs(ann["image_id"])[0]
                img_path = img_root / img_info["file_name"]
                if not img_path.exists():
                    skipped_rows.append({"ann_id": ann_id, "category": category, "reason": "missing_image"})
                    continue

                try:
                    img = Image.open(img_path).convert("RGB")
                except Exception as e:
                    skipped_rows.append({"ann_id": ann_id, "category": category, "reason": f"corrupt_image:{e}"})
                    continue

                crop = crop_instance(img, bbox, args.pad_frac)
                if crop is None:
                    skipped_rows.append({"ann_id": ann_id, "category": category, "reason": "invalid_bbox"})
                    continue
                crop.save(out_dir / f"{img_info['id']}_{ann_id}.jpg", quality=90)
                written[split] += 1

        counts[category] = written

    # Per-category summary — makes the class-imbalance risk visible instead
    # of discovered only after a confusing training run.
    print("\nInstances written per category (train / val):")
    for category, split_counts in sorted(counts.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(split_counts.values())
        flag = "  <-- very few examples, expect weak accuracy here" if total < 15 else ""
        print(f"  {category:<20} train={split_counts['train']:<5} val={split_counts['val']:<5}{flag}")

    if cigarette_skipped_small:
        print(
            f"\nSkipped {cigarette_skipped_small} 'cigarette' annotations for being smaller than "
            f"--min-cigarette-size={args.min_cigarette_size}px — cigarette butts are often tiny in "
            "TACO's source photos; this is expected, not a bug."
        )

    if skipped_rows:
        skip_log = args.output_dir / "prepare_dataset_skipped.csv"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        with skip_log.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["ann_id", "category", "reason"])
            writer.writeheader()
            writer.writerows(skipped_rows)
        print(f"\nSkipped {len(skipped_rows)} annotations total (see {skip_log} for reasons).")

    print(f"\nDataset written to: {args.output_dir}")


if __name__ == "__main__":
    main()
