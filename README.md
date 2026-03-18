# Dengue Forecast

Research and forecasting code for weekly dengue incidence modeling in Bangladesh.

## Project Layout

- `configs/`: YAML configuration for data paths, features, splits, and model settings.
- `data/`: active pipeline data staged by lifecycle (`raw`, `external`, `interim`, `processed`).
- `dengue/`: reusable project code used by the data pipeline and forecasting workflows.
- `scripts/`: executable entrypoints grouped by task area (`data`, `training`, `forecasting`, `analysis`, `ops`).
- `reports/`: tracked summary outputs and operational forecast exports.
- `docs/`: lightweight working notes, runbook, decisions, and backlog items.
- `references/`: papers and reference-only source material that support the project.
- `archive/`: preserved starter-pack files and imported legacy assets that are not part of the active workflow.
- `artifacts/`: ignored model outputs, diagnostics, and intermediate experiment assets.

## Recommended Workflow

Run scripts from the repository root with module syntax:

```bash
python -m scripts.data.build_panel
python -m scripts.data.build_features
python -m scripts.data.make_splits
python -m scripts.training.train_glm --fold F1
python -m scripts.analysis.summarize_models
```

## Notes

- Raw operational inputs should live in `data/raw/`, not directly in `data/`.
- Reusable logic belongs in `dengue/`; one-off entrypoints belong in `scripts/`.
- Old scaffold code was moved into `archive/starter_pack/` so the live package reflects the code that is actually used.
- Imported PDFs, snapshots, and legacy duplicate data were moved out of the repo root to reduce clutter.
