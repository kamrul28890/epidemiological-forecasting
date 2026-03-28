# Evaluation Protocol Refreeze - 2026-03-18

This note records the protocol correction made after the original locked holdout `H1` was used for diagnostic model refinement.

## Why a refreeze was necessary

- The case panel currently ends at `2025-09-15`.
- There is no newer untouched case block available in the repository.
- After `H1` (`2025-07-01` to `2025-09-15`) was used to diagnose low-origin failure modes and motivate the guardrail pass, it could no longer be treated as a final untouched holdout.

So the scientifically correct action was not to keep pretending that `H1` was still a lockbox.

## New evidence hierarchy

The active split protocol is now:

- `development`
  - `D1`: train through `2022-12-31`, validate through `2023-06-30`
  - `D2`: train through `2023-06-30`, validate through `2023-12-31`
  - `D3`: train through `2023-12-31`, validate through `2024-06-30`
  - `D4`: train through `2024-06-30`, validate through `2024-12-31`
- `selection`
  - `S1`: train through `2024-12-31`, validate through `2025-06-30`
- `diagnostic`
  - `X1`: train through `2025-06-30`, validate through `2025-09-15`

Interpretation:

- `development` is for model building.
- `selection` is for comparing challengers under a fixed pre-diagnostic protocol.
- `diagnostic` is for recent-window stress testing only.
- There is currently no true untouched final holdout left in the available data.

## Refrozen run

- Backtest run: `modeling_tracks_20260318T234734Z`
- Forecast run: `operational_current_20260318T234853Z`

## What the refrozen protocol says

### Operational selection leaderboard

- `panel_hist_gbm`: MAE `18.429`, RMSE `36.923`
- `residual_elastic_net_global`: MAE `18.829`, RMSE `35.618`
- `ar_tweedie_global`: MAE `19.901`, RMSE `35.575`
- `conservative_stack`: MAE `23.887`, RMSE `43.383`
- `persistence`: MAE `33.993`, RMSE `61.040`

### Operational diagnostic leaderboard

- `persistence`: MAE `92.571`, RMSE `125.059`
- `conservative_stack`: MAE `177.161`, RMSE `209.539`
- `seasonal_naive_52`: MAE `226.465`, RMSE `283.906`
- `residual_random_forest`: MAE `238.695`, RMSE `299.302`

## Interpretation

The refrozen protocol reveals a clean disagreement:

- On the `selection` window, several challengers beat persistence convincingly.
- On the most recent `diagnostic` window, persistence still dominates.

That means there is still no challenger with a strong enough robustness case to replace persistence operationally.

## Operational policy after refreeze

Because there is no true untouched final holdout:

- the forecast system may use the `selection` leaderboard for challenger ranking
- but it should not auto-promote a challenger to primary status
- the primary operational model should remain `persistence` until a challenger wins under a newly frozen untouched final block

This is exactly what the refreshed forecast bundle does:

- Ranking source: `selection`
- Promotion policy: `cautious_default_persistence_without_true_holdout`
- Primary model: `persistence`

## Paper implication

For the manuscript, the right claim is now:

- the project has a clean development record
- a fixed selection window where challengers look promising
- a recent diagnostic window where persistence remains most reliable
- and no untouched post-tuning final holdout yet

That is still publishable, but it must be described honestly.

## Next requirement

The next decisive step is not another tuning pass.

It is one of these:

- obtain new post-`2025-09-15` weekly surveillance data and reserve it as a true untouched final evaluation block
- or pre-register a new fixed future evaluation window before any additional model tuning

Until then, challenger improvements should be treated as promising but not final.
