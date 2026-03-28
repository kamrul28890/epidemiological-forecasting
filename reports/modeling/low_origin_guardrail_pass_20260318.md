# Low-Origin Guardrail Pass - 2026-03-18

This note records the next operational-modeling pass after the locked-holdout result showed that challenger models were overpredicting rebounds after low or zero observed counts.

Tracked artifacts:

- Diagnostic run analyzed: `modeling_tracks_20260318T230604Z`
- Failure-mode report before changes: `reports/modeling/failure_mode_analysis_modeling_tracks_20260318T230604Z.md`
- New guarded backtest run: `modeling_tracks_20260318T233354Z`
- Failure-mode report after changes: `reports/modeling/failure_mode_analysis_modeling_tracks_20260318T233354Z.md`
- Refreshed operational forecast: `operational_current_20260318T233516Z`

## What was added

- A dedicated failure-analysis script at `scripts/analysis/analyze_operational_failure_modes.py`.
- Train-derived low-origin guardrails in `dengue/forecasting/modeling_tracks.py`.
- Matching live-forecast guardrails in `dengue/forecasting/current_forecast.py`.

The guardrail is intentionally conservative:

- identify low-origin states using current cases, zero-run length, weeks since nonzero, and recent nonzero activity
- estimate train-time probability of any future cases under similar low-origin states
- shrink positive challenger rebounds toward persistence using that empirical nonzero rate
- cap rebound size using train-time low-origin quantiles

This is a hurdle-style correction, not a new free-running model.

## Why this pass was necessary

The previous holdout analysis showed a very specific failure mode:

- the challenger family was not only losing overall
- it was especially overpredicting after low-count and zero-count origins

Examples from `modeling_tracks_20260318T230604Z`:

- `conservative_stack` holdout low-origin MAE: `20.606`
- `residual_random_forest` holdout low-origin MAE: `46.724`
- `residual_hist_gbm` holdout low-origin MAE: `54.539`

On the strict zero-origin holdout bucket, those models were often predicting rebounds where the realized target remained zero.

## Main result

The guardrail improved the targeted failure regime clearly.

### Holdout low-origin macro MAE

- `conservative_stack`: `20.606` -> `9.441`
- `residual_random_forest`: `46.724` -> `12.404`
- `residual_hist_gbm`: `54.539` -> `12.586`
- `panel_hist_gbm`: `24.645` -> `11.731`
- `ar_tweedie_global`: `13.440` -> `10.680`
- `residual_elastic_net_global`: `12.203` -> `10.126`
- `persistence`: unchanged at `8.354`

### Zero-origin bucket improvement

Examples:

- `conservative_stack` zero-origin MAE: `14.786` -> `1.592`
- `residual_random_forest` zero-origin MAE: `41.834` -> `5.211`
- `residual_hist_gbm` zero-origin MAE: `55.566` -> `5.449`
- `panel_hist_gbm` zero-origin MAE: `23.999` -> `4.744`

Interpretation:

- The safeguard successfully removed most of the unrealistic rebound behavior after zero origins.
- It made the challenger family much safer in the operational regime that motivated the change.

## Overall leaderboard result

The overall operational holdout still does not beat persistence.

Previous holdout leaderboard:

- `persistence`: MAE `92.571`
- `conservative_stack`: MAE `178.105`
- `residual_random_forest`: MAE `243.902`
- `residual_hist_gbm`: MAE `311.740`

New guarded holdout leaderboard:

- `persistence`: MAE `92.571`
- `conservative_stack`: MAE `176.526`
- `residual_random_forest`: MAE `238.695`
- `residual_hist_gbm`: MAE `303.688`

Interpretation:

- The guardrail helps, but only modestly on the full holdout summary.
- Persistence still remains the best operational model by a wide margin.

## Why the overall ranking barely changed

The reason is now clearer.

- We fixed a specific low-origin overprediction problem.
- But the remaining overall error is concentrated in medium and high-incidence windows, especially the `21_100` and `100p` origin buckets.
- In those larger-volume states, persistence is still much harder to beat.

So the low-origin guardrail was useful, but it does not solve the full operational problem by itself.

## Development behavior

The guardrail did not break development performance.

Operational development leaderboard:

- previous `conservative_stack`: MAE `138.655`
- new `conservative_stack`: MAE `137.261`

That is slightly better, not worse.

This is encouraging because it means the safeguard did not simply collapse the challengers back to persistence everywhere.

## Forecasting implication

The live operational forecast remains persistence-led, but challenger forecasts are now more conservative.

Refreshed forecast run:

- `operational_current_20260318T233516Z`

Example country-total challenger changes:

- `conservative_stack`: now `0.8`, `3.0`, `6.4`, `8.3`
- previous `conservative_stack`: `2.5`, `10.8`, `30.5`, `40.8`

Interpretation:

- The challenger forecasts are now much less prone to spuriously large rebounds from a zero-count origin week.
- That is operationally safer, even though they are still not the promoted primary model.

## Important methodological note

This pass was informed by the locked-holdout failure analysis.

That means:

- the original holdout `H1` is no longer a fully untouched final test for model-development claims
- this run should be treated as a diagnostic improvement pass, not as the last word on generalization

If we continue improving the model family, we should create a new final untouched evaluation window or otherwise re-freeze the final test protocol.

## Current conclusion

The project now has a stronger and more nuanced result:

- richer autoregressive challengers can be made much safer in low-origin and sparse-activity regimes
- hurdle-style shrinkage is useful for operational dengue forecasting on this dataset
- but persistence still remains the best overall short-horizon operational model in the current evaluation setup

That is a worthwhile modeling result and a useful paper result.

## Best next move

The next improvement pass should target the remaining error mass:

- medium and high-incidence windows
- outbreak-phase level overshoot
- possibly horizon-specific shrinkage or segmented ensembles

But any further tuning should be paired with a refreshed untouched final evaluation window.
