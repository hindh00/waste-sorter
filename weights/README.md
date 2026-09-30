# weights/

This directory is empty until you train a model.

After running `scripts/train_classifier.py`, Ultralytics writes results to
`runs/classify/<run_name>/weights/best.pt`. Copy (or symlink) that file to
`weights/best.pt` — that's the path `config.yaml`'s `model.weights_path`
points at, and what `waste_sorter/classifier.py` loads.

Until that file exists, the app will raise a clear `WeightsNotFoundError`
with these same instructions rather than crashing unexpectedly.
