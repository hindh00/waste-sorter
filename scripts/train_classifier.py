"""
train_classifier.py

Fine-tunes a YOLO classification model on the dataset produced by
prepare_dataset.py. Thin wrapper around Ultralytics' own training loop —
deliberately no reimplemented training logic here.

Usage:
    python scripts/train_classifier.py --data-dir data/classification --epochs 40

Ultralytics writes results (including weights/best.pt) to
runs/classify/<run-name>/, or <run-name>2, <run-name>3, ... on repeated
runs. This script prints the exact path at the end — copy that best.pt to
weights/best.pt (or update config.yaml's model.weights_path) to make it
the one waste_sorter/classifier.py actually loads.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="data/classification",
                         help="ImageFolder-style dataset root (must contain train/ and val/)")
    parser.add_argument("--base-model", default="yolov8n-cls.pt",
                         help="Pretrained checkpoint to start from (n/s/m/l/x sizes available)")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=10,
                         help="Stop early if val accuracy hasn't improved for this many epochs")
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--run-name", default="waste_sorter")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        sys.exit(
            "ultralytics is not installed. Install project requirements first:\n"
            "  pip install -r requirements.txt\n"
            "(use a Python 3.11/3.12 venv, not system Python — see README.md)"
        )

    model = YOLO(args.base_model)
    model.train(
        data=args.data_dir,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.run_name,
        seed=args.seed,
    )

    metrics = model.val()
    print("\nValidation metrics:")
    print(metrics.results_dict)
    print(
        "\nBefore you trust this: take 20-30 photos yourself, the way you'd actually use the "
        "app (close-up, held item, your lighting) — not TACO's own validation split — and check "
        "accuracy on those specifically. TACO's photos are outdoor litter on the ground; a clean "
        "close-up staged photo is a different visual domain, and that gap is the real test of "
        "whether this model is actually usable yet.\n\n"
        f"Copy the best weights from '{args.project}/{args.run_name}/weights/best.pt' "
        "to 'weights/best.pt' (or update config.yaml's model.weights_path) to use them in the app."
    )


if __name__ == "__main__":
    main()
