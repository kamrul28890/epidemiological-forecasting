# Data Layout

- `raw/`: active source files used by the pipeline.
- `external/`: side inputs such as population weights.
- `interim/`: cleaned joined panels before feature generation.
- `processed/`: design matrices, splits, and final model-ready tables.

Keep raw files inside `data/raw/` only; avoid dropping loose files directly into `data/`.
