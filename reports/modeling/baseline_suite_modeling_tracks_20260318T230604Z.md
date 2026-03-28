# Modeling Tracks Run Summary

- Run ID: `modeling_tracks_20260318T230604Z`

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
- Development folds: `['F1', 'F2', 'F3', 'F4', 'F5']`
- Holdout folds: `['H1']`
- Proxy geographies: `['DHA_DIV_OUT_METRO']`

### Experimental

- Include weather: `True`
- Horizons: `[5, 6, 7, 8, 9, 10, 11, 12]`
- Rows: `1197`
- Geographies: `9`
- Feature count: `70`
- Development folds: `['F1', 'F2', 'F3', 'F4', 'F5']`
- Holdout folds: `['H1']`
- Proxy geographies: `['DHA_DIV_OUT_METRO']`

## Development Leaderboard

### Experimental

- persistence: MAE_macro=403.016, RMSE_macro=510.986
- panel_hist_gbm: MAE_macro=434.239, RMSE_macro=569.917
- ar_tweedie_global: MAE_macro=490.982, RMSE_macro=648.011
- seasonal_naive_52: MAE_macro=521.907, RMSE_macro=716.453

### Operational

- conservative_stack: MAE_macro=138.655, RMSE_macro=182.229
- persistence: MAE_macro=154.218, RMSE_macro=206.390
- ar_tweedie_global: MAE_macro=156.820, RMSE_macro=208.931
- residual_random_forest: MAE_macro=160.143, RMSE_macro=213.688
- residual_hist_gbm: MAE_macro=171.579, RMSE_macro=232.430
- residual_elastic_net_global: MAE_macro=181.353, RMSE_macro=253.734
- panel_hist_gbm: MAE_macro=295.382, RMSE_macro=368.380
- seasonal_naive_52: MAE_macro=520.177, RMSE_macro=714.417

## Locked Holdout Leaderboard

### Experimental

- persistence: MAE_macro=183.073, RMSE_macro=199.499
- seasonal_naive_52: MAE_macro=195.636, RMSE_macro=232.594
- ar_tweedie_global: MAE_macro=542.049, RMSE_macro=684.030
- panel_hist_gbm: MAE_macro=684.090, RMSE_macro=867.956

### Operational

- persistence: MAE_macro=92.571, RMSE_macro=125.059
- conservative_stack: MAE_macro=178.105, RMSE_macro=207.982
- seasonal_naive_52: MAE_macro=226.465, RMSE_macro=283.906
- residual_random_forest: MAE_macro=243.902, RMSE_macro=300.495
- residual_elastic_net_global: MAE_macro=248.049, RMSE_macro=286.444
- panel_hist_gbm: MAE_macro=283.117, RMSE_macro=350.395
- ar_tweedie_global: MAE_macro=289.328, RMSE_macro=328.255
- residual_hist_gbm: MAE_macro=311.740, RMSE_macro=407.897

## Development Best By Horizon

### Experimental

- Horizon 5: persistence (MAE_macro=270.135, RMSE_macro=354.393)
- Horizon 6: persistence (MAE_macro=314.480, RMSE_macro=404.825)
- Horizon 7: persistence (MAE_macro=354.512, RMSE_macro=453.017)
- Horizon 8: persistence (MAE_macro=392.490, RMSE_macro=499.525)
- Horizon 9: persistence (MAE_macro=427.783, RMSE_macro=541.849)
- Horizon 10: panel_hist_gbm (MAE_macro=458.166, RMSE_macro=615.270)
- Horizon 11: panel_hist_gbm (MAE_macro=473.030, RMSE_macro=632.271)
- Horizon 12: persistence (MAE_macro=513.265, RMSE_macro=641.742)

### Operational

- Horizon 1: conservative_stack (MAE_macro=72.304, RMSE_macro=101.222)
- Horizon 2: conservative_stack (MAE_macro=119.205, RMSE_macro=159.557)
- Horizon 3: conservative_stack (MAE_macro=153.418, RMSE_macro=200.806)
- Horizon 4: conservative_stack (MAE_macro=209.692, RMSE_macro=267.329)

## Locked Holdout By Horizon

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
