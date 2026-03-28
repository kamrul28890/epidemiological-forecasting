# Current Operational Forecast

- Forecast run ID: `operational_current_20260318T233516Z`
- Origin week: `2025-09-15`
- Forecast horizons: `[1, 2, 3, 4]`

## Backtest Ranking Used

- Ranking source: `locked holdout`
- 1. persistence: MAE_macro=92.571, RMSE_macro=125.059
- 2. conservative_stack: MAE_macro=176.526, RMSE_macro=208.304
- 3. seasonal_naive_52: MAE_macro=226.465, RMSE_macro=283.906
- 4. residual_random_forest: MAE_macro=238.695, RMSE_macro=299.302
- 5. residual_elastic_net_global: MAE_macro=247.707, RMSE_macro=286.439
- 6. panel_hist_gbm: MAE_macro=280.907, RMSE_macro=349.719
- 7. ar_tweedie_global: MAE_macro=288.643, RMSE_macro=328.117
- 8. residual_hist_gbm: MAE_macro=303.688, RMSE_macro=404.871

## Forecast Bundle

### ar_tweedie_global

- Primary model: `False`
- Fit status values: `['trained_full_history_low_origin_guardrail']`
- Country total forecasts: h1=3.8, h2=5.0, h3=6.0, h4=9.2

### conservative_stack

- Primary model: `False`
- Fit status values: `['stack_from_backtest_mae_low_origin_guardrail']`
- Country total forecasts: h1=0.8, h2=3.0, h3=6.4, h4=8.3

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
