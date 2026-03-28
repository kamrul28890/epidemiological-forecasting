# Operational Improvement Pass - 2026-03-18

This note records the first post-baseline operational model-improvement pass using:

- `residual_elastic_net_global`
- `conservative_stack`

Tracked run:

- Backtest run: `modeling_tracks_20260318T225154Z`
- Forecast run: `operational_current_20260318T225245Z`

## What changed

- Added a new persistence-anchored global residual elastic-net model.
- Added a conservative stacked ensemble using leave-one-fold-out validation MAE to set horizon-specific model weights.
- Fixed an important residual-model constraint:
  raw residual predictions are no longer clipped at zero before they are added back to the persistence anchor.

## Main operational result

The conservative stack is now the strongest operational model overall.

Validation leaderboard:

- `conservative_stack`: MAE `150.706`, RMSE `195.994`
- `persistence`: MAE `154.218`, RMSE `206.390`
- `residual_random_forest`: MAE `169.578`, RMSE `223.024`
- `residual_hist_gbm`: MAE `176.275`, RMSE `243.340`
- `ar_tweedie_global`: MAE `181.767`, RMSE `241.761`
- `residual_elastic_net_global`: MAE `207.649`, RMSE `287.271`

Interpretation:

- This is the first tracked operational model that beats persistence overall.
- The gain is not huge, but it is real and it is on the primary metric.
- The ensemble is doing exactly what we wanted:
  staying close to persistence while borrowing controlled signal from stronger challengers.

## Horizon story

Best model by horizon:

- horizon 1: `conservative_stack`
- horizon 2: `conservative_stack`
- horizon 3: `conservative_stack`
- horizon 4: `persistence`

Interpretation:

- The ensemble improves the near-term part of the operational window most clearly.
- Horizon 4 is still stubborn and remains best served by plain persistence.

## Fold story

Validation MAE by fold:

- `F1`: `conservative_stack` `80.813` vs persistence `88.015`
- `F2`: `conservative_stack` `23.370` vs persistence `33.993`
- `F3`: persistence `24.840` vs `conservative_stack` `28.295`
- `F4`: `conservative_stack` `466.607` vs persistence `470.507`
- `F5`: persistence `153.736` vs `conservative_stack` `154.446`

Interpretation:

- The stack wins clearly in `F1`, `F2`, and slightly in `F4`.
- Persistence still wins `F3` and narrowly wins `F5`.
- This means the new model is better overall, but it is not yet a universal winner in every validation regime.

## Geography story

Best average operational model by geography:

- `BAR`: `conservative_stack`
- `CHA`: `conservative_stack`
- `DHA_DIV_OUT_METRO`: `conservative_stack`
- `DHA_METRO`: `persistence`
- `KHU`: `residual_elastic_net_global`
- `MYM`: `residual_elastic_net_global`
- `RAJ`: `residual_elastic_net_global`
- `RAN`: `residual_elastic_net_global`
- `SYL`: `persistence`

Interpretation:

- The stack helps most on several medium-to-large geographies.
- The elastic-net residual model is interesting because it is strongest in several lower-volume geographies.
- Dhaka Metro and Sylhet still prefer simpler baselines.

## Elastic-net interpretation

The elastic-net model did not win overall, but it taught us something important.

- It was very strong in `F2`.
- It was competitive in several sparse or lower-volume geographies.
- It failed badly in `F5`, which suggests that its log-residual correction is promising but not robust enough yet.

That makes it a useful component model and a useful scientific result, even if it is not yet the promoted production model.

## Conservative stack interpretation

The stack used horizon-specific, leave-one-fold-out validation MAE to assign nonnegative model weights, with a persistence floor of `0.45`.

This makes the ensemble:

- conservative by design
- anchored to the strongest naive baseline
- less likely to overreact to one model's temporary fold win

In practice, this appears to be the right bias for the current dataset.

## Current operational forecast decision

Because `conservative_stack` is now the best operational backtest model, it becomes the new primary operational forecaster.

Latest origin week:

- `2025-09-15`

Country-total primary forecast:

- h1 = `8.4`
- h2 = `13.4`
- h3 = `30.3`
- h4 = `36.8`

Interpretation:

- The new primary forecast is no longer flat zero like persistence.
- It still remains far more conservative than the largest challenger rebounds.
- This is a more believable operational signal than either pure persistence or the most aggressive model-based rebound.

## What this means for the next pass

The next improvement phase should focus on:

- regime-aware features
- hurdle modeling for sparse geographies
- horizon-4 improvement specifically
- locked final evaluation to reduce leaderboard overfitting risk

Most importantly, the project has now moved from
`"nothing beats persistence"`
to
`"a persistence-anchored conservative ensemble beats persistence overall"`

That is a meaningful step forward for both the forecasting system and the paper narrative.
