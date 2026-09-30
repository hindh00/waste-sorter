"""
Tests that constructing WasteClassifier without trained weights fails with
a clear, actionable error rather than a raw Ultralytics stack trace.

Note: this only needs `ultralytics` importable if weights DO exist (the
import is lazy, inside the `if` branch that only runs once the weights
file check passes) — so this test works even before `pip install -r
requirements.txt` has been run, which is expected: there are no weights
yet in a fresh checkout either way.
"""

from waste_sorter.classifier import WasteClassifier, WeightsNotFoundError


def test_missing_weights_raises_clear_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.pt"
    try:
        WasteClassifier(missing_path, confidence_threshold=0.65)
        assert False, "expected WeightsNotFoundError"
    except WeightsNotFoundError as e:
        message = str(e)
        assert "No trained model weights found" in message
        assert "prepare_dataset.py" in message
        assert "train_classifier.py" in message
        assert str(missing_path) in message


def test_missing_weights_error_is_a_runtime_error(tmp_path):
    assert issubclass(WeightsNotFoundError, RuntimeError)
