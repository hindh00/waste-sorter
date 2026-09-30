"""
classifier.py

Thin wrapper around a trained YOLO classification model so the rest of the
app never touches the Ultralytics API directly.

This project ships as CODE ONLY — no trained weights are included. Until
you run scripts/prepare_dataset.py and scripts/train_classifier.py (see
README.md) and point config.yaml's model.weights_path at the result,
constructing WasteClassifier raises WeightsNotFoundError with the exact
next steps, instead of the app crashing on first use with a confusing
Ultralytics stack trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


class WeightsNotFoundError(RuntimeError):
    """Raised when no trained model weights exist yet."""


@dataclass(frozen=True)
class ClassifierResult:
    category: str
    confidence: float
    is_confident: bool


class WasteClassifier:
    def __init__(self, weights_path: str | Path, confidence_threshold: float):
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise WeightsNotFoundError(
                f"No trained model weights found at '{weights_path}'.\n\n"
                "This project ships with code only, not a trained model. To get one:\n"
                "  1. Clone TACO and run its download.py to fetch the dataset images.\n"
                "  2. python scripts/prepare_dataset.py --taco-root <path> "
                "--output-dir data/classification\n"
                "  3. python scripts/train_classifier.py --data-dir data/classification\n"
                "  4. Copy the resulting weights/best.pt (see runs/classify/<run>/weights/) "
                f"to '{weights_path}', or update config.yaml's model.weights_path.\n\n"
                "See README.md for the full walkthrough."
            )

        # Imported lazily so importing this module never itself requires
        # ultralytics/weights to exist — only constructing this class does.
        from ultralytics import YOLO

        self._model = YOLO(str(weights_path))
        self.confidence_threshold = confidence_threshold

    def predict(self, image) -> ClassifierResult:
        """Runs inference on a single image.

        `image` may be a file path (str/Path), a PIL.Image, or a numpy
        array — anything Ultralytics' predict() accepts.
        """
        results = self._model.predict(image, verbose=False)
        result = results[0]
        top_idx = int(result.probs.top1)
        confidence = float(result.probs.top1conf)
        category = result.names[top_idx]
        return ClassifierResult(
            category=category,
            confidence=confidence,
            is_confident=confidence >= self.confidence_threshold,
        )
