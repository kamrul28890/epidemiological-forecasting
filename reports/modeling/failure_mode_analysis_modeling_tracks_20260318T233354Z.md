# Operational Failure-Mode Analysis - modeling_tracks_20260318T233354Z

This report breaks the operational validation results into the regimes that mattered most after the holdout pass:
low-origin windows, zero-origin windows, sparse-activity behavior, and outbreak-flagged periods.

## Scope

- Rows analyzed: `48240` operational validation predictions.
- Split groups present: `['development', 'holdout']`
- Models present: `['ar_tweedie_global', 'conservative_stack', 'panel_hist_gbm', 'persistence', 'residual_elastic_net_global', 'residual_hist_gbm', 'residual_random_forest', 'seasonal_naive_52']`

## Development Low-Origin Leaderboard

- persistence: MAE_macro=6.214, RMSE_macro=16.467
- conservative_stack: MAE_macro=6.286, RMSE_macro=16.464
- panel_hist_gbm: MAE_macro=6.335, RMSE_macro=16.032
- residual_elastic_net_global: MAE_macro=6.359, RMSE_macro=16.228
- ar_tweedie_global: MAE_macro=6.362, RMSE_macro=16.139
- residual_random_forest: MAE_macro=6.958, RMSE_macro=16.737
- residual_hist_gbm: MAE_macro=7.091, RMSE_macro=17.079
- seasonal_naive_52: MAE_macro=61.509, RMSE_macro=173.624

## Holdout Low-Origin Leaderboard

- persistence: MAE_macro=8.354, RMSE_macro=20.726
- conservative_stack: MAE_macro=9.441, RMSE_macro=21.085
- residual_elastic_net_global: MAE_macro=10.126, RMSE_macro=21.295
- ar_tweedie_global: MAE_macro=10.680, RMSE_macro=21.531
- panel_hist_gbm: MAE_macro=11.731, RMSE_macro=22.115
- residual_random_forest: MAE_macro=12.404, RMSE_macro=22.822
- residual_hist_gbm: MAE_macro=12.586, RMSE_macro=22.728
- seasonal_naive_52: MAE_macro=416.704, RMSE_macro=430.108

## Holdout By Origin Bucket

- `zero` best model: persistence (MAE=0.182, bias=0.182, mean_pred=0.000, mean_y=0.182)
- `1_5` best model: panel_hist_gbm (MAE=17.491, bias=15.683, mean_pred=5.590, mean_y=21.273)
- `6_20` best model: persistence (MAE=12.184, bias=3.605, mean_pred=11.447, mean_y=15.053)
- `21_100` best model: persistence (MAE=46.566, bias=16.453, mean_pred=42.774, mean_y=59.226)
- `100p` best model: persistence (MAE=152.438, bias=-100.806, mean_pred=372.184, mean_y=271.378)

## Holdout Zero-Origin By Horizon

- Horizon 1 best model under zero-origin starts: persistence (MAE=0.000, mean_pred=0.000, mean_y=0.000)
- Horizon 2 best model under zero-origin starts: persistence (MAE=0.000, mean_pred=0.000, mean_y=0.000)
- Horizon 3 best model under zero-origin starts: persistence (MAE=0.000, mean_pred=0.000, mean_y=0.000)
- Horizon 4 best model under zero-origin starts: residual_hist_gbm (MAE=4.300, mean_pred=5.700, mean_y=10.000)

## Holdout By Regime Flag

- `low_origin` best model: persistence (MAE_macro=8.354, RMSE_macro=20.726)
- `not_low_origin` best model: persistence (MAE_macro=106.784, RMSE_macro=139.849)
- `outbreak_flag` best model: persistence (MAE_macro=78.158, RMSE_macro=106.412)
- `not_outbreak_flag` best model: persistence (MAE_macro=116.417, RMSE_macro=160.297)

## Holdout Geography Winners

- BAR: persistence (MAE=209.909, RMSE=262.209, bias=-201.136)
- CHA: persistence (MAE=121.636, RMSE=176.355, bias=-53.091)
- DHA_DIV_OUT_METRO: persistence (MAE=117.273, RMSE=176.665, bias=-50.682)
- DHA_METRO: persistence (MAE=196.023, RMSE=304.389, bias=-94.705)
- KHU: persistence (MAE=66.659, RMSE=91.006, bias=-16.795)
- MYM: persistence (MAE=19.977, RMSE=26.806, bias=-5.250)
- RAJ: persistence (MAE=84.182, RMSE=114.249, bias=-36.864)
- RAN: persistence (MAE=12.568, RMSE=16.025, bias=-0.614)
- SYL: seasonal_naive_52 (MAE=3.909, RMSE=5.931, bias=3.909)

## Interpretation

- The key operational question is whether challengers are overpredicting rebound after low or zero observed incidence.
- The low-origin and zero-origin tables make that visible directly instead of hiding it inside overall MAE.
- These segmented results should be read together with the main holdout leaderboard before promoting any challenger.
