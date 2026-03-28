# Modeling Tracks Run Summary

- Run ID: `modeling_tracks_20260318T233354Z`

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
- panel_hist_gbm: MAE_macro=423.073, RMSE_macro=557.336
- ar_tweedie_global: MAE_macro=475.809, RMSE_macro=624.951
- seasonal_naive_52: MAE_macro=521.907, RMSE_macro=716.453

### Operational

- conservative_stack: MAE_macro=137.261, RMSE_macro=181.100
- persistence: MAE_macro=154.218, RMSE_macro=206.390
- residual_random_forest: MAE_macro=155.539, RMSE_macro=209.945
- ar_tweedie_global: MAE_macro=156.252, RMSE_macro=208.551
- residual_hist_gbm: MAE_macro=163.515, RMSE_macro=219.570
- residual_elastic_net_global: MAE_macro=181.163, RMSE_macro=253.472
- panel_hist_gbm: MAE_macro=295.031, RMSE_macro=368.385
- seasonal_naive_52: MAE_macro=520.177, RMSE_macro=714.417

## Locked Holdout Leaderboard

### Experimental

- persistence: MAE_macro=183.073, RMSE_macro=199.499
- seasonal_naive_52: MAE_macro=195.636, RMSE_macro=232.594
- ar_tweedie_global: MAE_macro=537.848, RMSE_macro=680.006
- panel_hist_gbm: MAE_macro=673.757, RMSE_macro=861.092

### Operational

- persistence: MAE_macro=92.571, RMSE_macro=125.059
- conservative_stack: MAE_macro=176.526, RMSE_macro=208.304
- seasonal_naive_52: MAE_macro=226.465, RMSE_macro=283.906
- residual_random_forest: MAE_macro=238.695, RMSE_macro=299.302
- residual_elastic_net_global: MAE_macro=247.707, RMSE_macro=286.439
- panel_hist_gbm: MAE_macro=280.907, RMSE_macro=349.719
- ar_tweedie_global: MAE_macro=288.643, RMSE_macro=328.117
- residual_hist_gbm: MAE_macro=303.688, RMSE_macro=404.871

## Development Best By Horizon

### Experimental

- Horizon 5: persistence (MAE_macro=270.135, RMSE_macro=354.393)
- Horizon 6: persistence (MAE_macro=314.480, RMSE_macro=404.825)
- Horizon 7: persistence (MAE_macro=354.512, RMSE_macro=453.017)
- Horizon 8: persistence (MAE_macro=392.490, RMSE_macro=499.525)
- Horizon 9: panel_hist_gbm (MAE_macro=426.647, RMSE_macro=572.596)
- Horizon 10: panel_hist_gbm (MAE_macro=446.106, RMSE_macro=600.294)
- Horizon 11: panel_hist_gbm (MAE_macro=460.333, RMSE_macro=614.621)
- Horizon 12: persistence (MAE_macro=513.265, RMSE_macro=641.742)

### Operational

- Horizon 1: conservative_stack (MAE_macro=71.953, RMSE_macro=101.106)
- Horizon 2: conservative_stack (MAE_macro=118.208, RMSE_macro=158.980)
- Horizon 3: conservative_stack (MAE_macro=152.480, RMSE_macro=200.123)
- Horizon 4: conservative_stack (MAE_macro=206.402, RMSE_macro=264.190)

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
