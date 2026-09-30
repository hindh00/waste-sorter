"""
fallback_logger.py

Appends one row per LLM fallback event to a CSV log. This log is the
app's built-in "what am I missing" feedback loop: periodically reviewing
it — especially rows with is_new_category=True — is how you decide what
to add as a real trained class next, instead of guessing.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "timestamp",
    "image_id",
    "predicted_category",
    "confidence",
    "bin",
    "source",
    "is_new_category",
    "reasoning",
]


def log_fallback(
    log_path: str | Path,
    image_id: str,
    predicted_category: str,
    confidence: float,
    bin_name: str,
    known_categories: list[str],
    reasoning: str = "",
) -> None:
    log_path = Path(log_path)
    is_new = predicted_category not in known_categories

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "image_id": image_id,
        "predicted_category": predicted_category,
        "confidence": confidence,
        "bin": bin_name,
        "source": "llm_fallback",
        "is_new_category": is_new,
        "reasoning": reasoning,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
