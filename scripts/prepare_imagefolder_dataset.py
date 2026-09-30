"""
prepare_imagefolder_dataset.py

Merges an already-ImageFolder-shaped source dataset (RealWaste, the Kaggle
Garbage Classification set, or anything similar — one subfolder per source
label, images directly inside) into this project's data/classification/
tree, using the same {train,val}/<consolidated category>/ layout that
scripts/prepare_dataset.py produces for TACO.

Unlike TACO, these sources are already single-object classification photos
(no COCO boxes to crop), so this script only needs to map each source
folder to a consolidated category and split into train/val.

This is additive: it does NOT clear data/classification first, so you can
run it once per source (TACO via prepare_dataset.py, then this script once
per extra dataset) and they accumulate into one combined dataset. Re-running
it for the same source will duplicate files unless you clear that source's
files first (every filename is prefixed with --source-tag, so you can
`rm data/classification/*/*/<source-tag>_*` to undo just one source).

Usage:
    python scripts/prepare_imagefolder_dataset.py \\
        --source-dir data/realwaste/extracted/realwaste-main/RealWaste \\
        --category-map data/category_map_realwaste.yaml \\
        --source-tag realwaste \\
        --output-dir data/classification \\
        --val-ratio 0.15

After adding a new source, delete data/classification/{train,val}.cache
(Ultralytics classification caches file lists) before retraining, or it
won't see the new images.
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image, UnidentifiedImageError

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def load_category_map(path: Path) -> dict[str, str]:
    """Loads {category: [source folder names]} and inverts it into a flat
    {source folder name: category} lookup, same shape as prepare_dataset.py
    uses for TACO's category_map.yaml."""
    with path.open("r") as f:
        grouped: dict[str, list[str]] = yaml.safe_load(f) or {}

    flat: dict[str, str] = {}
    for category, folder_names in grouped.items():
        for name in folder_names:
            if name in flat:
                warnings.warn(
                    f"Source folder '{name}' is mapped to both "
                    f"'{flat[name]}' and '{category}' — using '{category}'."
                )
            flat[name] = category
    return flat


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source-dir", type=Path, required=True,
                         help="Root of the extracted source dataset (contains one subfolder per label)")
    parser.add_argument("--category-map", type=Path, required=True,
                         help="Path to the {category: [source folder names]} YAML mapping for this source")
    parser.add_argument("--source-tag", required=True,
                         help="Short tag (e.g. 'realwaste', 'garbage_classification') prefixed onto every "
                              "copied filename, so files from different sources never collide and can be "
                              "identified/removed later")
    parser.add_argument("--output-dir", type=Path, default=Path("data/classification"),
                         help="Output folder for the merged classification dataset (same one TACO writes to)")
    parser.add_argument("--val-ratio", type=float, default=0.15,
                         help="Fraction of each category's images held out for validation")
    parser.add_argument("--max-per-category", type=int, default=None,
                         help="Optional cap on images taken per source folder, applied after shuffling — "
                              "use this to avoid one huge folder (e.g. 'clothes') dwarfing everything else")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.source_dir.is_dir():
        sys.exit(f"Source dir not found: {args.source_dir}")

    category_map = load_category_map(args.category_map)
    random.seed(args.seed)

    by_category: dict[str, list[Path]] = defaultdict(list)
    unmapped_folders: set[str] = set()

    for folder in sorted(p for p in args.source_dir.iterdir() if p.is_dir()):
        category = category_map.get(folder.name)
        if category is None:
            unmapped_folders.add(folder.name)
            category = "other_unlabeled"
        images = [p for p in folder.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
        by_category[category].extend(images)

    if unmapped_folders:
        print(
            f"WARNING: the following folders in {args.source_dir} are not in "
            f"{args.category_map} and were bucketed into 'other_unlabeled':\n  "
            + ", ".join(sorted(unmapped_folders))
        )

    counts: dict[str, dict[str, int]] = {}
    corrupt = 0

    for category, paths in by_category.items():
        shuffled = paths[:]
        random.shuffle(shuffled)
        if args.max_per_category is not None:
            shuffled = shuffled[: args.max_per_category]

        n_val = max(1, round(len(shuffled) * args.val_ratio)) if len(shuffled) > 1 else 0
        splits = {"val": shuffled[:n_val], "train": shuffled[n_val:]}
        written = {"train": 0, "val": 0}

        for split, split_paths in splits.items():
            out_dir = args.output_dir / split / category
            out_dir.mkdir(parents=True, exist_ok=True)

            for src_path in split_paths:
                try:
                    with Image.open(src_path) as img:
                        img.verify()
                except (UnidentifiedImageError, OSError) as e:
                    corrupt += 1
                    print(f"  skipping corrupt image {src_path}: {e}")
                    continue

                dest_name = f"{args.source_tag}_{src_path.stem}{src_path.suffix.lower()}"
                dest_path = out_dir / dest_name
                if dest_path.exists():
                    dest_path = out_dir / f"{args.source_tag}_{src_path.parent.name}_{src_path.stem}{src_path.suffix.lower()}"
                shutil.copy2(src_path, dest_path)
                written[split] += 1

        counts[category] = written

    print(f"\nInstances added per category from '{args.source_tag}' (train / val):")
    for category, split_counts in sorted(counts.items(), key=lambda kv: -sum(kv[1].values())):
        total = sum(split_counts.values())
        print(f"  {category:<20} train={split_counts['train']:<5} val={split_counts['val']:<5} (total={total})")

    if corrupt:
        print(f"\nSkipped {corrupt} corrupt/unreadable images.")

    print(f"\nMerged into: {args.output_dir}")
    print(
        "Remember to delete data/classification/{train,val}.cache before retraining "
        "so Ultralytics picks up the new files."
    )


if __name__ == "__main__":
    main()
