# Manuscript Outline

This outline is designed for the current forecasting evidence in the repo.

Use the latest paper asset bundle under `reports/paper/` together with:

- `docs/article_notes.md`
- `docs/figure_interpretations.md`
- `reports/modeling/modeling_results_interpretation_20260317.md`

## Working Title Options

1. Short-Horizon Weekly Dengue Forecasting in Bangladesh: Strong Baselines Remain Hard to Beat
2. Operational Multi-Geo Dengue Forecasting Under Covariate Gaps and Hierarchy Constraints
3. Weekly Dengue Forecasting in Bangladesh: Evidence From Hierarchy-Aware Backtesting Across Multiple Geographies

## Core Paper Message

The paper should argue that:

- this is a weekly multi-geo count forecasting problem
- short-horizon persistence is the dominant predictive signal
- hierarchy repair is necessary for coherent geographic modeling
- weather is scientifically relevant but not yet operationally dependable because of coverage mismatch
- stronger model classes can help in some folds, but they do not yet beat persistence robustly enough for promotion

## Abstract Structure

### Background

- Explain why weekly dengue forecasting matters operationally.
- State that Bangladesh exhibits multi-geo epidemic dynamics with strong short-run persistence and uneven reporting scale.

### Objective

- Forecast weekly dengue cases across multiple geographies at short operational horizons.

### Methods

- Weekly panel forecasting by `geo_id`
- hierarchy repair with `DHA_DIV_OUT_METRO`
- direct multi-horizon evaluation
- operational track: 1 to 4 weeks
- experimental track: 5 to 12 weeks
- benchmark models: persistence, seasonal naive, count model, tree models, residual correction models

### Results

- Persistence was the best overall operational and experimental validation baseline.
- Residual random forest was the strongest operational challenger.
- Weather-augmented models did not beat persistence in the current setup.

### Conclusion

- Under realistic data constraints, strong autoregressive baselines remain difficult to beat for operational dengue forecasting.

## Introduction

### Section goal

- Establish why dengue forecasting matters.
- Motivate why this is not just a generic time-series task.
- Position the paper as an operational forecasting study rather than a purely mechanistic one.

### Points to cover

- dengue burden and public-health relevance
- need for short-horizon regional forecasts
- challenge of sparse and highly uneven regional incidence
- challenge of overlapping geographic hierarchies
- challenge of climate data gaps
- literature gap:
  many studies test complex models, but fewer evaluate whether they outperform strong simple baselines under real operational constraints

## Data

### Section goal

- Describe the case and climate data, geography definitions, and the hierarchy repair.

### Evidence to cite

- dataset profile from `reports/eda/summary.md`
- latest paper asset table:
  `table_02_dataset_summary.csv`

### Key points

- 9 geographic series
- weekly case coverage through `2025-09-15`
- climate coverage through `2025-06-30`
- 11-week covariate gap
- overlapping `DHA_DIV` and `DHA_METRO`
- repaired hierarchy using `DHA_DIV_OUT_METRO`

## Exploratory Analysis

### Section goal

- Show that the data are persistent, outbreak-driven, overdispersed, and only moderately seasonal.

### Evidence to cite

- `docs/figure_interpretations.md`
- `reports/eda/figures/*`
- `table_02_dataset_summary.csv`

### Key points

- strong lag-1 autocorrelation
- weaker annual recurrence
- substantial zero inflation in smaller geographies
- severe overdispersion
- cross-geo synchronization
- climate appears relevant, but coverage is incomplete for production use

## Methods

### Forecasting task definition

- weekly dengue count forecasting
- one prediction per geography per week
- operational horizon: 1 to 4 weeks
- experimental horizon: 5 to 12 weeks
- direct multi-horizon forecasting, not recursive

### Target and predictors

- target for direct models: future weekly case count
- target for residual models: future count minus current count
- operational predictors:
  case lags, rolling means, `log1p(cases)`, calendar terms, geography identifier
- experimental predictors:
  operational predictors plus climate lags and rolling summaries

### Models

- persistence
- seasonal naive
- regularized count model
- panel histogram GBM
- residual histogram GBM
- residual random forest

### Evaluation

- rolling-origin folds
- per-geo and per-horizon scoring
- macro-averaged MAE as primary metric
- macro-averaged RMSE as secondary metric

## Results

### Main results subsection

- Present the operational leaderboard first.
- Then present the experimental leaderboard.

### Evidence to cite

- `table_03_operational_leaderboard.csv`
- `table_04_experimental_leaderboard.csv`
- `fig_01_operational_leaderboard.png`
- `fig_02_operational_horizon_mae.png`
- `fig_s1_experimental_horizon_mae.png`

### Fold-level results subsection

- Show that some challengers win in certain folds, but persistence is the most stable overall.

### Evidence to cite

- `table_05_fold_winners.csv`
- `fig_03_operational_fold_mae_heatmap.png`

### Geography-level results subsection

- Show how model performance varies by geography.
- Emphasize that Dhaka-centered series should not dominate interpretation.

### Evidence to cite

- `table_07_operational_geo_model_summary.csv`
- `fig_04_operational_geo_mae_heatmap.png`

### Current forecast subsection

- Present the current 1 to 4 week operational forecast as an example deployment output.

### Evidence to cite

- `table_09_current_country_forecast.csv`
- `table_10_current_primary_forecast.csv`
- `fig_05_current_country_forecasts.png`

## Discussion

### Main discussion points

- why persistence is so strong in this dataset
- why seasonal naive is weak despite visible seasonality
- why weather did not become an operational winner
- why some challenger models help in calmer or more regular periods but fail to replace persistence overall
- why hierarchy repair matters for credible multi-geo forecasting

## Limitations

- relatively short series length
- strong dependence on a few outbreak years
- climate coverage gap
- no uncertainty intervals yet
- no externally validated mechanistic covariates

## Conclusion

- Re-state the main practical result:
  short-horizon operational weekly dengue forecasting in this dataset is driven primarily by recent case history, and strong simple baselines remain difficult to beat.

## Figure Plan

1. Operational validation leaderboard
2. Operational MAE by horizon
3. Operational MAE by fold heatmap
4. Operational MAE by geography heatmap
5. Current country forecast by model

## Table Plan

1. Study design
2. Dataset summary
3. Operational leaderboard
4. Experimental leaderboard
5. Fold winners
6. Horizon winners
7. Operational geo-model summary
8. Experimental geo-model summary
9. Current country forecast
10. Current primary forecast

## Remaining Work Before Submission

- add literature citations
- do one more structured error-analysis pass by outbreak phase and geography
- refine the discussion of why challengers fail under regime shift
- decide whether to keep the current forecast output in the main paper or move it to the appendix
