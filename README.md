# Waste-sorting assistant

Take a close-up photo of one item → a locally-run YOLO classifier predicts
what it is → `rules.yaml` maps that to a bin + prep steps → if the model
isn't confident, Claude looks at the photo instead and answers, and that
case gets logged so you can add it as a real trained class later.

This repo is a **working scaffold, not a finished app**: all the code is
written and wired together, but no dataset has been downloaded and no
model has been trained yet. See "What's already done" and "What you need
to do" below.

---

## Architecture

```
Item photo
    │
    ▼
Local YOLO-cls model (fine-tuned on TACO crops)
    │
    ├─ confidence ≥ threshold AND category known ──► rules.yaml lookup ──► bin + steps
    │
    └─ confidence < threshold OR category unknown ──► Claude (vision) ──► bin + steps
                                                              │
                                                              ▼
                                                     logged to logs/fallback_log.csv
```

The model's only job is "what is this." Everything about *what to do* with
that answer lives in `rules.yaml`, completely separate from the model —
that's what lets you retarget the whole app to a different city's
recycling program by editing one file, not retraining anything.

---

## What's already done (this scaffold)

- `rules.yaml` — 10 categories with placeholder bin names + prep steps, and
  a `confidence_threshold`. **Replace the bin names with your actual local
  program's names before relying on this.**
- `data/category_map.yaml` — the TACO fine-label → consolidated-category
  mapping (see "Category design" below).
- `scripts/prepare_dataset.py` — converts a downloaded TACO dataset into an
  ImageFolder classification dataset, with graceful handling of missing
  Flickr images, per-category stratified train/val splitting, and special
  handling for tiny cigarette crops.
- `scripts/train_classifier.py` — fine-tunes a YOLO classification model on
  that dataset.
- `data/category_map_realwaste.yaml`, `data/category_map_garbage_classification.yaml`
  — same idea as `data/category_map.yaml`, for two more source datasets.
- `scripts/prepare_imagefolder_dataset.py` — merges those (already plain
  ImageFolder, no cropping needed) into `data/classification/` alongside
  the TACO crops. See "Expanding the dataset" below.
- `waste_sorter/` — the app's core logic (`classifier.py`, `rules_engine.py`,
  `llm_fallback.py`, `fallback_logger.py`, `config.py`), each independently
  testable.
- `app/streamlit_app.py` — the demo UI.
- `tests/` — unit tests for the rules engine, LLM response parsing, and the
  missing-weights error path. These pass today with no dataset or trained
  model required.

## What you need to do

1. **Use Python 3.11 or 3.12 for this project — not your system Python.**
   This machine's system Python is 3.14, which current PyTorch/Ultralytics
   wheels don't yet support. Create a dedicated environment first, e.g.:
   ```bash
   pyenv install 3.11.9      # if you don't already have a 3.11/3.12
   pyenv local 3.11.9
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Run the tests** to confirm the scaffold works before touching data:
   ```bash
   pytest tests/
   ```
   All tests should pass — they don't need a dataset or trained weights.

3. **Get TACO and download its images:**
   ```bash
   git clone https://github.com/pedropro/TACO.git data/raw_taco
   cd data/raw_taco && python3 download.py   # pulls images from Flickr — slow, some links will be dead
   cd ../..
   ```
   Dead/missing images are handled gracefully by `prepare_dataset.py` —
   expect some, don't expect zero.

4. **Build the classification dataset:**
   ```bash
   python scripts/prepare_dataset.py --taco-root data/raw_taco --output-dir data/classification
   ```
   Read the printed per-category train/val counts. Categories like
   `battery_hazardous`, `textile_other`, and `cigarette` are TACO-rare by
   construction (see below) — a low count there is expected, not a bug.
   Before training, open a dozen crops from a few folders by eye — cropped
   litter photos are messy (motion blur, partial objects, weird angles).

5. **Train a baseline model:**
   ```bash
   python scripts/train_classifier.py --data-dir data/classification --epochs 40
   ```
   Watch per-class accuracy, not just overall top-1 — a model that's 95%
   accurate overall but never gets `bottle_glass` right is a model that
   fails silently in the app. Copy the resulting
   `runs/classify/waste_sorter/weights/best.pt` to `weights/best.pt`.

6. **Before you trust it**, take 20-30 photos yourself, the way you'd
   actually use the app (close-up, held item, your own lighting) — not
   TACO's validation split — and check accuracy on those specifically.
   TACO's photos are outdoor litter on the ground; a clean close-up staged
   photo is a genuinely different visual domain, and this is the real test
   of whether that gap is a problem for you.

7. **Add your Anthropic API key:**
   ```bash
   cp .env.example .env
   # edit .env and set ANTHROPIC_API_KEY
   ```

8. **Run the app:**
   ```bash
   streamlit run app/streamlit_app.py
   ```
   Test a confident case, a low-confidence/fallback case, and confirm a
   row lands in `logs/fallback_log.csv` after a fallback.

9. **Tune `confidence_threshold` in `rules.yaml`** once you can see real
   fallback-vs-correct rates — lower means fewer LLM calls but more wrong
   local answers slipping through; higher means the opposite. Don't guess
   it, check accuracy-vs-coverage on your validation set.

10. **Periodically review `logs/fallback_log.csv`**, especially rows with
    `is_new_category=True` — that's your "what am I missing" list. The
    most common misses become your next training classes; re-run steps 4-5
    once you've expanded `data/category_map.yaml` and `rules.yaml`.

---

## Expanding the dataset (RealWaste, Garbage Classification)

TACO alone leaves some categories thin — `battery_hazardous`,
`food_organic_waste`, and `textile_other` each map from only one or two TACO
fine labels. Two more sources fill those gaps; unlike TACO they're already
plain ImageFolder classification sets (no COCO boxes to crop), so
`scripts/prepare_imagefolder_dataset.py` merges them straight into
`data/classification/` alongside the TACO crops.

1. **RealWaste** (https://github.com/sam-single/realwaste, 9 classes,
   ~4,700 images):
   ```bash
   # download realwaste.zip into data/realwaste/, then:
   unzip data/realwaste/realwaste.zip -d data/realwaste/extracted
   python scripts/prepare_imagefolder_dataset.py \
     --source-dir data/realwaste/extracted/realwaste-main/RealWaste \
     --category-map data/category_map_realwaste.yaml \
     --source-tag realwaste \
     --output-dir data/classification \
     --val-ratio 0.15
   ```

2. **Garbage Classification** (Kaggle, 12 classes, ~15,000 images —
   `clothes`/`shoes` alone are ~7,300 of those):
   ```bash
   # extract into data/garbage_classification/, then:
   python scripts/prepare_imagefolder_dataset.py \
     --source-dir data/garbage_classification \
     --category-map data/category_map_garbage_classification.yaml \
     --source-tag garbage_classification \
     --output-dir data/classification \
     --val-ratio 0.15 \
     --max-per-category 2000
   ```
   The `--max-per-category` cap matters here: uncapped, `clothes`+`shoes`
   alone would make `textile_other` ~23x the size of `bottle_plastic` or
   `cigarette`, which neither new source has any images for (both stay
   TACO-only). 2000 keeps `textile_other` in line with the other
   categories without starving it back down to TACO's original 28.

3. **Delete the stale Ultralytics cache before retraining** — it caches the
   file list from the last `data/classification` contents:
   ```bash
   rm -f data/classification/train.cache data/classification/val.cache
   python scripts/train_classifier.py --data-dir data/classification --epochs 40
   ```

Both source folder names are matched case-sensitively against
`data/category_map_realwaste.yaml` / `data/category_map_garbage_classification.yaml`.
Neither dataset distinguishes plastic bottles from other rigid plastic, so
their "Plastic"/"plastic" images go to `plastic_packaging`, not
`bottle_plastic` — check per-class val accuracy on `bottle_plastic`
specifically after training, since it's still TACO-only.

Each run is additive and tags every copied filename with `--source-tag`
(e.g. `realwaste_Cardboard_12.jpg`), so re-running a source without first
deleting its old files (`rm data/classification/*/*/<tag>_*`) will
duplicate it, not replace it.

---

## Category design

TACO (https://github.com/pedropro/TACO) is a detection/instance-segmentation
dataset: every annotation is a bounding box around one litter item in a
scene, labeled with one of ~60 fine-grained categories (e.g. "Drink can",
"Clear plastic bottle", "Aluminium foil"). That's far too fine-grained (and
imbalanced) to train directly, so `data/category_map.yaml` collapses those
60 labels into the 10 practical categories `rules.yaml` actually handles:

| Category | Bin intent |
|---|---|
| `can` | Metal recycling |
| `bottle_plastic` | Plastic recycling |
| `bottle_glass` | Glass recycling |
| `carton_paper` | Paper/cardboard recycling |
| `plastic_packaging` | Soft/mixed plastic recycling |
| `cigarette` | Trash (never recyclable) |
| `food_organic_waste` | Compost/organics |
| `battery_hazardous` | Hazardous waste / e-waste drop-off |
| `textile_other` | Textile recycling/donation |
| `other_unlabeled` | General trash / needs manual judgment |

`plastic_packaging` absorbs by far the most TACO fine labels (~30 of them)
and will dominate the dataset numerically. `battery_hazardous`,
`textile_other`, and `cigarette` each map from only one or two fine labels
and will be TACO-rare — expect thin validation sets for those and treat
early predictions there with proportionate skepticism.

This mapping is a starting point, not gospel. If you add or rename a
category, edit `data/category_map.yaml` **and** `rules.yaml` together, then
re-run `prepare_dataset.py` and retrain.

---

## Known limitations (v1, by design)

- **Single item only.** One centered, close-up photo of one thing — not a
  scattered scene. Multi-item detection is a different, harder problem
  (full-scene detection, not classification) and is intentionally out of
  scope for v1.
- **Rules are hand-authored, not AI-guessed.** The app never asks an LLM to
  invent local recycling law from scratch — `rules.yaml` is yours to write
  and correct for your actual city's program.
- **It won't be right about everything on day one.** That's what the
  fallback + logging loop is for. A persistently high fallback rate is a
  signal to add training classes, not something to paper over.
- **Cigarette butts are a known hard case** — they're often under 64px in
  TACO's original photos, so surviving crops after the minimum-size filter
  are few and lower quality than other categories.
