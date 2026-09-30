# tests/fixtures/

Drop a real close-up photo of a single item here (e.g. `sample_item.jpg`)
if you want to manually smoke-test `waste_sorter/llm_fallback.py`'s live
Claude call, or `waste_sorter/classifier.py` once you have trained weights.

None of the automated tests in `tests/` currently require an image file —
they test the pure logic (rules lookups, JSON parsing, missing-weights
handling) without needing a dataset, trained model, or network access.
