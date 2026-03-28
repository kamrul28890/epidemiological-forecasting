# Paper Assets Results Snapshot

- Backtest run: `modeling_tracks_20260318T234734Z`
- Forecast run: `operational_current_20260318T234853Z`

## Main Result

- Operational development winner: `conservative_stack` with MAE_macro=153.448.
- Operational selection winner: `panel_hist_gbm` with MAE_macro=18.429.
- Operational diagnostic winner: `persistence` with MAE_macro=92.571.
- Experimental selection winner: `ar_tweedie_global` with MAE_macro=35.423.
- Experimental diagnostic winner: `persistence` with MAE_macro=183.073.

## Core Interpretation

- Challengers look strong on the fixed selection window.
- Persistence remains the most reliable model on the most recent diagnostic window.
- Because the latest window was used diagnostically, there is no untouched final holdout for promotion claims.

## Fold Story

### Experimental

- D1: seasonal_naive_52 (MAE_macro=39.898, RMSE_macro=76.350)
- D2: seasonal_naive_52 (MAE_macro=1101.265, RMSE_macro=1358.462)
- D3: seasonal_naive_52 (MAE_macro=33.342, RMSE_macro=67.596)
- D4: persistence (MAE_macro=342.099, RMSE_macro=425.875)

### Operational

- D1: persistence (MAE_macro=24.840, RMSE_macro=50.972)
- D2: conservative_stack (MAE_macro=452.796, RMSE_macro=548.230)
- D3: panel_hist_gbm (MAE_macro=15.949, RMSE_macro=26.323)
- D4: conservative_stack (MAE_macro=112.594, RMSE_macro=143.991)

## Current Country Forecasts

- ar_tweedie_global (primary=False): h1=3.8, h2=5.0, h3=6.0, h4=9.2
- conservative_stack (primary=False): h1=0.8, h2=3.1, h3=6.5, h4=8.8
- panel_hist_gbm (primary=False): h1=2.9, h2=8.3, h3=9.7, h4=27.9
- persistence (primary=True): h1=0.0, h2=0.0, h3=0.0, h4=0.0
- residual_elastic_net_global (primary=False): h1=0.0, h2=0.0, h3=0.0, h4=0.0
- residual_hist_gbm (primary=False): h1=0.2, h2=8.8, h3=0.0, h4=0.0
- residual_random_forest (primary=False): h1=4.9, h2=20.8, h3=59.6, h4=81.2
- seasonal_naive_52 (primary=False): h1=5457.0, h2=6800.0, h3=6445.0, h4=6772.0

## Recommended Paper Framing

- The strongest applied result is not a single leaderboard win, but instability across evidence tiers: challengers win on selection, persistence wins on recent diagnostic stress testing.
- The best operational claim is therefore conservative: persistence remains the promoted primary model until a true untouched future holdout is available.
- Weather is scientifically relevant but not operationally dependable in the current coverage window.
