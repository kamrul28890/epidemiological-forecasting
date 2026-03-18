# Runbook

## Core Pipeline

From the repo root:

```bash
python -m scripts.data.build_panel
python -m scripts.data.build_features
python -m scripts.data.make_splits
```

## Training

Train individual folds with:

```bash
python -m scripts.training.train_glm --fold F1
python -m scripts.training.train_gbm --fold F1
python -m scripts.training.train_sarima --fold F1
python -m scripts.training.train_persistence --fold F1
python -m scripts.training.train_seasonal_naive --fold F1
```

Or use the ops helpers:

```bash
bash scripts/ops/run_baselines.sh
bash scripts/ops/run_all_baselines_and_stack.sh
bash scripts/ops/run_glm_alpha_sweep.sh
```

## Forecasting And Analysis

```bash
python -m scripts.forecasting.forecast_next12
python -m scripts.forecasting.forecast_operational --mode exoglite
python -m scripts.analysis.stack_ensemble
python -m scripts.analysis.summarize_models
python -m scripts.analysis.diagnostics --forecast artifacts/forecasts/gbm_F2.parquet --tag gbm_F2
```

## Conventions

- Keep reusable code in `dengue/`.
- Keep generated experiment outputs in `artifacts/`.
- Keep tracked summary outputs in `reports/`.
- Treat `archive/` as preserved history, not active code.
