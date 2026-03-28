# Current Operational Forecast

- Forecast run ID: `operational_current_20260318T225245Z`
- Origin week: `2025-09-15`
- Forecast horizons: `[1, 2, 3, 4]`

## Backtest Ranking Used

- 1. conservative_stack: MAE_macro=150.706, RMSE_macro=195.994
- 2. persistence: MAE_macro=154.218, RMSE_macro=206.390
- 3. residual_random_forest: MAE_macro=169.578, RMSE_macro=223.024
- 4. residual_hist_gbm: MAE_macro=176.275, RMSE_macro=243.340
- 5. ar_tweedie_global: MAE_macro=181.767, RMSE_macro=241.761
- 6. residual_elastic_net_global: MAE_macro=207.649, RMSE_macro=287.271
- 7. panel_hist_gbm: MAE_macro=296.576, RMSE_macro=367.618
- 8. seasonal_naive_52: MAE_macro=520.177, RMSE_macro=714.417

## Forecast Bundle

### ar_tweedie_global

- Primary model: `False`
- Fit status values: `['trained_full_history']`
- Country total forecasts: h1=27.1, h2=34.8, h3=47.1, h4=70.2

### conservative_stack

- Primary model: `True`
- Fit status values: `['stack_from_backtest_mae']`
- Country total forecasts: h1=8.4, h2=13.4, h3=30.3, h4=36.8

### panel_hist_gbm

- Primary model: `False`
- Fit status values: `['trained_full_history']`
- Country total forecasts: h1=8.8, h2=13.6, h3=25.4, h4=53.0

### persistence

- Primary model: `False`
- Fit status values: `['naive_current_cases']`
- Country total forecasts: h1=0.0, h2=0.0, h3=0.0, h4=0.0

### residual_elastic_net_global

- Primary model: `False`
- Fit status values: `['trained_full_history']`
- Country total forecasts: h1=2.1, h2=2.8, h3=4.1, h4=8.0

### residual_hist_gbm

- Primary model: `False`
- Fit status values: `['trained_full_history']`
- Country total forecasts: h1=18.2, h2=24.5, h3=84.7, h4=37.9

### residual_random_forest

- Primary model: `False`
- Fit status values: `['trained_full_history']`
- Country total forecasts: h1=15.0, h2=35.6, h3=77.3, h4=123.2

### seasonal_naive_52

- Primary model: `False`
- Fit status values: `['naive_same_week_last_year']`
- Country total forecasts: h1=5457.0, h2=6800.0, h3=6445.0, h4=6772.0
