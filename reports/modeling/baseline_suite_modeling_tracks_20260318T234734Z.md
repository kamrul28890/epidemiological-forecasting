# Modeling Tracks Run Summary

- Run ID: `modeling_tracks_20260318T234734Z`

## Hierarchy

- Residual geography: `DHA_DIV_OUT_METRO`
- Max closure error: `0.0`
- Negative raw residual weeks clipped: `0`

## Track Snapshots

### Operational

- Include weather: `False`
- Horizons: `[1, 2, 3, 4]`
- Rows: `1287`
- Geographies: `9`
- Feature count: `32`
- Development folds: `['D1', 'D2', 'D3', 'D4']`
- Selection folds: `['S1']`
- Holdout folds: `[]`
- Diagnostic folds: `['X1']`
- Proxy geographies: `['DHA_DIV_OUT_METRO']`

### Experimental

- Include weather: `True`
- Horizons: `[5, 6, 7, 8, 9, 10, 11, 12]`
- Rows: `1197`
- Geographies: `9`
- Feature count: `70`
- Development folds: `['D1', 'D2', 'D3', 'D4']`
- Selection folds: `['S1']`
- Holdout folds: `[]`
- Diagnostic folds: `['X1']`
- Proxy geographies: `['DHA_DIV_OUT_METRO']`

## Development Leaderboard

### Experimental

- persistence: MAE_macro=444.159, RMSE_macro=560.255
- panel_hist_gbm: MAE_macro=460.410, RMSE_macro=585.644
- ar_tweedie_global: MAE_macro=508.583, RMSE_macro=638.830
- seasonal_naive_52: MAE_macro=527.589, RMSE_macro=677.237

### Operational

- conservative_stack: MAE_macro=153.448, RMSE_macro=196.133
- persistence: MAE_macro=167.212, RMSE_macro=216.485
- residual_random_forest: MAE_macro=170.061, RMSE_macro=220.962
- ar_tweedie_global: MAE_macro=175.108, RMSE_macro=226.418
- residual_hist_gbm: MAE_macro=176.869, RMSE_macro=229.515
- residual_elastic_net_global: MAE_macro=208.699, RMSE_macro=282.558
- panel_hist_gbm: MAE_macro=351.071, RMSE_macro=428.054
- seasonal_naive_52: MAE_macro=525.426, RMSE_macro=674.692

## Selection Leaderboard

### Experimental

- ar_tweedie_global: MAE_macro=35.423, RMSE_macro=61.112
- panel_hist_gbm: MAE_macro=36.231, RMSE_macro=65.192
- seasonal_naive_52: MAE_macro=39.410, RMSE_macro=77.366
- persistence: MAE_macro=178.336, RMSE_macro=300.890

### Operational

- panel_hist_gbm: MAE_macro=18.429, RMSE_macro=36.923
- residual_elastic_net_global: MAE_macro=18.829, RMSE_macro=35.618
- ar_tweedie_global: MAE_macro=19.901, RMSE_macro=35.575
- conservative_stack: MAE_macro=23.887, RMSE_macro=43.383
- residual_random_forest: MAE_macro=25.216, RMSE_macro=44.040
- residual_hist_gbm: MAE_macro=31.902, RMSE_macro=59.994
- persistence: MAE_macro=33.993, RMSE_macro=61.040
- seasonal_naive_52: MAE_macro=39.410, RMSE_macro=77.366

## Diagnostic Leaderboard

### Experimental

- persistence: MAE_macro=183.073, RMSE_macro=199.499
- seasonal_naive_52: MAE_macro=195.636, RMSE_macro=232.594
- ar_tweedie_global: MAE_macro=537.848, RMSE_macro=680.006
- panel_hist_gbm: MAE_macro=673.757, RMSE_macro=861.092

### Operational

- persistence: MAE_macro=92.571, RMSE_macro=125.059
- conservative_stack: MAE_macro=177.161, RMSE_macro=209.539
- seasonal_naive_52: MAE_macro=226.465, RMSE_macro=283.906
- residual_random_forest: MAE_macro=238.695, RMSE_macro=299.302
- residual_elastic_net_global: MAE_macro=247.707, RMSE_macro=286.439
- panel_hist_gbm: MAE_macro=280.907, RMSE_macro=349.719
- ar_tweedie_global: MAE_macro=288.643, RMSE_macro=328.117
- residual_hist_gbm: MAE_macro=303.688, RMSE_macro=404.871

## Development Best By Horizon

### Experimental

- Horizon 5: persistence (MAE_macro=293.402, RMSE_macro=380.144)
- Horizon 6: persistence (MAE_macro=342.179, RMSE_macro=434.557)
- Horizon 7: persistence (MAE_macro=386.648, RMSE_macro=489.166)
- Horizon 8: persistence (MAE_macro=430.105, RMSE_macro=544.540)
- Horizon 9: panel_hist_gbm (MAE_macro=460.451, RMSE_macro=596.064)
- Horizon 10: panel_hist_gbm (MAE_macro=477.075, RMSE_macro=613.271)
- Horizon 11: panel_hist_gbm (MAE_macro=490.191, RMSE_macro=615.059)
- Horizon 12: seasonal_naive_52 (MAE_macro=529.555, RMSE_macro=679.217)

### Operational

- Horizon 1: conservative_stack (MAE_macro=78.955, RMSE_macro=106.623)
- Horizon 2: conservative_stack (MAE_macro=132.486, RMSE_macro=172.328)
- Horizon 3: conservative_stack (MAE_macro=173.348, RMSE_macro=219.455)
- Horizon 4: conservative_stack (MAE_macro=229.004, RMSE_macro=286.127)

## Selection Best By Horizon

### Experimental

- Horizon 5: ar_tweedie_global (MAE_macro=26.664, RMSE_macro=50.361)
- Horizon 6: ar_tweedie_global (MAE_macro=27.705, RMSE_macro=51.340)
- Horizon 7: ar_tweedie_global (MAE_macro=28.412, RMSE_macro=50.512)
- Horizon 8: ar_tweedie_global (MAE_macro=30.885, RMSE_macro=54.244)
- Horizon 9: ar_tweedie_global (MAE_macro=34.020, RMSE_macro=57.625)
- Horizon 10: panel_hist_gbm (MAE_macro=39.284, RMSE_macro=70.667)
- Horizon 11: seasonal_naive_52 (MAE_macro=39.410, RMSE_macro=77.366)
- Horizon 12: seasonal_naive_52 (MAE_macro=39.410, RMSE_macro=77.366)

### Operational

- Horizon 1: residual_random_forest (MAE_macro=14.131, RMSE_macro=24.978)
- Horizon 2: panel_hist_gbm (MAE_macro=16.286, RMSE_macro=31.959)
- Horizon 3: residual_elastic_net_global (MAE_macro=19.654, RMSE_macro=40.181)
- Horizon 4: panel_hist_gbm (MAE_macro=17.951, RMSE_macro=33.787)

## Diagnostic Best By Horizon

### Experimental

- Horizon 5: persistence (MAE_macro=120.537, RMSE_macro=138.135)
- Horizon 6: persistence (MAE_macro=140.968, RMSE_macro=158.695)
- Horizon 7: seasonal_naive_52 (MAE_macro=168.792, RMSE_macro=195.183)
- Horizon 8: seasonal_naive_52 (MAE_macro=179.247, RMSE_macro=211.990)
- Horizon 9: seasonal_naive_52 (MAE_macro=198.900, RMSE_macro=242.558)
- Horizon 10: persistence (MAE_macro=218.545, RMSE_macro=231.087)
- Horizon 11: persistence (MAE_macro=203.879, RMSE_macro=220.399)
- Horizon 12: persistence (MAE_macro=191.949, RMSE_macro=210.343)

### Operational

- Horizon 1: persistence (MAE_macro=43.737, RMSE_macro=75.128)
- Horizon 2: persistence (MAE_macro=72.737, RMSE_macro=108.786)
- Horizon 3: persistence (MAE_macro=107.384, RMSE_macro=142.895)
- Horizon 4: persistence (MAE_macro=146.424, RMSE_macro=173.428)
