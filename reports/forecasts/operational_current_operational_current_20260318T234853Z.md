# Current Operational Forecast

- Forecast run ID: `operational_current_20260318T234853Z`
- Origin week: `2025-09-15`
- Forecast horizons: `[1, 2, 3, 4]`

## Backtest Ranking Used

- Ranking source: `selection`
- Promotion policy: `cautious_default_persistence_without_true_holdout`
- 1. panel_hist_gbm: MAE_macro=18.429, RMSE_macro=36.923
- 2. residual_elastic_net_global: MAE_macro=18.829, RMSE_macro=35.618
- 3. ar_tweedie_global: MAE_macro=19.901, RMSE_macro=35.575
- 4. conservative_stack: MAE_macro=23.887, RMSE_macro=43.383
- 5. residual_random_forest: MAE_macro=25.216, RMSE_macro=44.040
- 6. residual_hist_gbm: MAE_macro=31.902, RMSE_macro=59.994
- 7. persistence: MAE_macro=33.993, RMSE_macro=61.040
- 8. seasonal_naive_52: MAE_macro=39.410, RMSE_macro=77.366

## Forecast Bundle

### ar_tweedie_global

- Primary model: `False`
- Fit status values: `['trained_full_history_low_origin_guardrail']`
- Country total forecasts: h1=3.8, h2=5.0, h3=6.0, h4=9.2

### conservative_stack

- Primary model: `False`
- Fit status values: `['stack_from_backtest_mae_low_origin_guardrail']`
- Country total forecasts: h1=0.8, h2=3.1, h3=6.5, h4=8.8

### panel_hist_gbm

- Primary model: `False`
- Fit status values: `['trained_full_history_low_origin_guardrail']`
- Country total forecasts: h1=2.9, h2=8.3, h3=9.7, h4=27.9

### persistence

- Primary model: `True`
- Fit status values: `['naive_current_cases']`
- Country total forecasts: h1=0.0, h2=0.0, h3=0.0, h4=0.0

### residual_elastic_net_global

- Primary model: `False`
- Fit status values: `['trained_full_history']`
- Country total forecasts: h1=0.0, h2=0.0, h3=0.0, h4=0.0

### residual_hist_gbm

- Primary model: `False`
- Fit status values: `['trained_full_history', 'trained_full_history_low_origin_guardrail']`
- Country total forecasts: h1=0.2, h2=8.8, h3=0.0, h4=0.0

### residual_random_forest

- Primary model: `False`
- Fit status values: `['trained_full_history_low_origin_guardrail']`
- Country total forecasts: h1=4.9, h2=20.8, h3=59.6, h4=81.2

### seasonal_naive_52

- Primary model: `False`
- Fit status values: `['naive_same_week_last_year']`
- Country total forecasts: h1=5457.0, h2=6800.0, h3=6445.0, h4=6772.0
