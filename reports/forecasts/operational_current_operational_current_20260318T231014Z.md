# Current Operational Forecast

- Forecast run ID: `operational_current_20260318T231014Z`
- Origin week: `2025-09-15`
- Forecast horizons: `[1, 2, 3, 4]`

## Backtest Ranking Used

- Ranking source: `locked holdout`
- 1. persistence: MAE_macro=92.571, RMSE_macro=125.059
- 2. conservative_stack: MAE_macro=178.105, RMSE_macro=207.982
- 3. seasonal_naive_52: MAE_macro=226.465, RMSE_macro=283.906
- 4. residual_random_forest: MAE_macro=243.902, RMSE_macro=300.495
- 5. residual_elastic_net_global: MAE_macro=248.049, RMSE_macro=286.444
- 6. panel_hist_gbm: MAE_macro=283.117, RMSE_macro=350.395
- 7. ar_tweedie_global: MAE_macro=289.328, RMSE_macro=328.255
- 8. residual_hist_gbm: MAE_macro=311.740, RMSE_macro=407.897

## Forecast Bundle

### ar_tweedie_global

- Primary model: `False`
- Fit status values: `['trained_full_history']`
- Country total forecasts: h1=7.1, h2=8.7, h3=10.0, h4=15.7

### conservative_stack

- Primary model: `False`
- Fit status values: `['stack_from_backtest_mae']`
- Country total forecasts: h1=2.5, h2=10.8, h3=30.5, h4=40.8

### panel_hist_gbm

- Primary model: `False`
- Fit status values: `['trained_full_history']`
- Country total forecasts: h1=6.0, h2=14.8, h3=16.2, h4=47.8

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
- Fit status values: `['trained_full_history']`
- Country total forecasts: h1=0.6, h2=34.7, h3=0.0, h4=0.0

### residual_random_forest

- Primary model: `False`
- Fit status values: `['trained_full_history']`
- Country total forecasts: h1=9.2, h2=37.2, h3=201.6, h4=273.1

### seasonal_naive_52

- Primary model: `False`
- Fit status values: `['naive_same_week_last_year']`
- Country total forecasts: h1=5457.0, h2=6800.0, h3=6445.0, h4=6772.0
