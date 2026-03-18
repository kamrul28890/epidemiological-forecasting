# Paper Assets Results Snapshot

- Backtest run: `modeling_tracks_20260317T035929Z`
- Forecast run: `operational_current_20260317T040253Z`

## Main Result

- Operational winner: `persistence` with MAE_macro=154.218 and RMSE_macro=206.390.
- Experimental winner: `persistence` with MAE_macro=403.016 and RMSE_macro=510.986.

## Fold Story

### Experimental

- F1: persistence (MAE_macro=284.218, RMSE_macro=447.106)
- F2: ar_tweedie_global (MAE_macro=33.031, RMSE_macro=55.901)
- F3: seasonal_naive_52 (MAE_macro=39.898, RMSE_macro=76.350)
- F4: seasonal_naive_52 (MAE_macro=1101.265, RMSE_macro=1358.462)
- F5: persistence (MAE_macro=342.099, RMSE_macro=425.875)

### Operational

- F1: panel_hist_gbm (MAE_macro=69.661, RMSE_macro=114.259)
- F2: panel_hist_gbm (MAE_macro=19.382, RMSE_macro=38.764)
- F3: persistence (MAE_macro=24.840, RMSE_macro=50.972)
- F4: residual_hist_gbm (MAE_macro=459.068, RMSE_macro=552.747)
- F5: panel_hist_gbm (MAE_macro=119.001, RMSE_macro=159.206)

## Current Country Forecasts

- ar_tweedie_global (primary=False): h1=27.1, h2=34.8, h3=47.1, h4=70.2
- panel_hist_gbm (primary=False): h1=8.8, h2=13.6, h3=25.4, h4=53.0
- persistence (primary=True): h1=0.0, h2=0.0, h3=0.0, h4=0.0
- residual_hist_gbm (primary=False): h1=18.2, h2=24.5, h3=84.7, h4=37.9
- residual_random_forest (primary=False): h1=15.0, h2=35.6, h3=77.3, h4=123.2
- seasonal_naive_52 (primary=False): h1=5457.0, h2=6800.0, h3=6445.0, h4=6772.0

## Recommended Paper Framing

- The strongest applied result is that persistence remains the most reliable short-horizon operational model under realistic data constraints.
- The best challenger class is persistence-anchored correction models, but they do not yet beat persistence consistently enough for promotion.
- Weather is scientifically relevant but not operationally dependable in the current coverage window.
