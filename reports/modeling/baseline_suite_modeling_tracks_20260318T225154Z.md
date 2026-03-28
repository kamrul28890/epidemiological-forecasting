# Modeling Tracks Run Summary

- Run ID: `modeling_tracks_20260318T225154Z`

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
- Feature count: `17`
- Proxy geographies: `['DHA_DIV_OUT_METRO']`

### Experimental

- Include weather: `True`
- Horizons: `[5, 6, 7, 8, 9, 10, 11, 12]`
- Rows: `1197`
- Geographies: `9`
- Feature count: `55`
- Proxy geographies: `['DHA_DIV_OUT_METRO']`

## Validation Leaderboard

### Experimental

- persistence: MAE_macro=403.016, RMSE_macro=510.986
- panel_hist_gbm: MAE_macro=516.184, RMSE_macro=692.385
- seasonal_naive_52: MAE_macro=521.907, RMSE_macro=716.453
- ar_tweedie_global: MAE_macro=540.113, RMSE_macro=714.009

### Operational

- conservative_stack: MAE_macro=150.706, RMSE_macro=195.994
- persistence: MAE_macro=154.218, RMSE_macro=206.390
- residual_random_forest: MAE_macro=169.578, RMSE_macro=223.024
- residual_hist_gbm: MAE_macro=176.275, RMSE_macro=243.340
- ar_tweedie_global: MAE_macro=181.767, RMSE_macro=241.761
- residual_elastic_net_global: MAE_macro=207.649, RMSE_macro=287.271
- panel_hist_gbm: MAE_macro=296.576, RMSE_macro=367.618
- seasonal_naive_52: MAE_macro=520.177, RMSE_macro=714.417

## Best By Horizon

### Experimental

- Horizon 5: persistence (MAE_macro=270.135, RMSE_macro=354.393)
- Horizon 6: persistence (MAE_macro=314.480, RMSE_macro=404.825)
- Horizon 7: persistence (MAE_macro=354.512, RMSE_macro=453.017)
- Horizon 8: persistence (MAE_macro=392.490, RMSE_macro=499.525)
- Horizon 9: persistence (MAE_macro=427.783, RMSE_macro=541.849)
- Horizon 10: persistence (MAE_macro=461.713, RMSE_macro=579.369)
- Horizon 11: persistence (MAE_macro=489.747, RMSE_macro=613.171)
- Horizon 12: persistence (MAE_macro=513.265, RMSE_macro=641.742)

### Operational

- Horizon 1: conservative_stack (MAE_macro=71.927, RMSE_macro=99.872)
- Horizon 2: conservative_stack (MAE_macro=120.678, RMSE_macro=159.704)
- Horizon 3: conservative_stack (MAE_macro=173.591, RMSE_macro=221.344)
- Horizon 4: persistence (MAE_macro=227.320, RMSE_macro=301.635)
