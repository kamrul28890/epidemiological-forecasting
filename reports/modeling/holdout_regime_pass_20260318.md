# Locked Holdout and Regime-Feature Pass - 2026-03-18

This note records the next operational-modeling pass after the first conservative-stack improvement.

Tracked artifacts:

- Backtest run: `modeling_tracks_20260318T230604Z`
- Holdout-aware forecast run: `operational_current_20260318T231014Z`

## What changed

- Added a locked final holdout fold `H1` in `configs/split.yaml`.
- Kept `F1` to `F5` as development folds only.
- Added regime-aware case features to the operational panel:
  recent sums, rolling standard deviations, log-growth, acceleration, outbreak intensity, outbreak flags, zero-run length, and weeks since last nonzero.
- Increased the operational feature set from `17` to `32` features.
- Forced conservative-stack weights to be learned from development folds only, then applied unchanged to the locked holdout.
- Updated the live forecast selector so the primary operational model follows the locked holdout leaderboard when it exists.

## Why this pass matters

This pass was designed to answer a more important question than
`"Can we improve the development leaderboard?"`

The real question was:

`"Do the regime-aware challengers still beat persistence once we stop tuning against the same folds we are reporting?"`

That is the right operational and paper-quality test.

## Development result

The regime-aware operational stack improved strongly on development folds.

Development leaderboard:

- `conservative_stack`: MAE `138.655`, RMSE `182.229`
- `persistence`: MAE `154.218`, RMSE `206.390`
- `ar_tweedie_global`: MAE `156.820`, RMSE `208.931`
- `residual_random_forest`: MAE `160.143`, RMSE `213.688`
- `residual_hist_gbm`: MAE `171.579`, RMSE `232.430`
- `residual_elastic_net_global`: MAE `181.353`, RMSE `253.734`

Interpretation:

- The new features did what we hoped on the development folds.
- `conservative_stack` now beats persistence by about `15.6` MAE points.
- It is the best development model at horizons `1`, `2`, `3`, and `4`.
- On development averages, it is best in `7` of the `9` modeled geographies, with persistence still best in `RAN` and `SYL`.

So the new feature set is not useless. It is extracting real signal from the training-era outbreak regimes.

## Locked holdout result

The untouched holdout changes the conclusion.

Locked holdout leaderboard:

- `persistence`: MAE `92.571`, RMSE `125.059`
- `conservative_stack`: MAE `178.105`, RMSE `207.982`
- `seasonal_naive_52`: MAE `226.465`, RMSE `283.906`
- `residual_random_forest`: MAE `243.902`, RMSE `300.495`
- `residual_elastic_net_global`: MAE `248.049`, RMSE `286.444`
- `panel_hist_gbm`: MAE `283.117`, RMSE `350.395`
- `ar_tweedie_global`: MAE `289.328`, RMSE `328.255`
- `residual_hist_gbm`: MAE `311.740`, RMSE `407.897`

Interpretation:

- The holdout winner is not close. `persistence` is clearly best.
- The development winner did not generalize to the untouched final window.
- This means the earlier promotion of `conservative_stack` was development-valid but not holdout-valid.
- Operationally, the correct model recommendation must revert to `persistence`.

This is exactly why we locked a final evaluation window before claiming a production model.

## Horizon story

Development best-by-horizon:

- h1: `conservative_stack`
- h2: `conservative_stack`
- h3: `conservative_stack`
- h4: `conservative_stack`

Locked holdout best-by-horizon:

- h1: `persistence`
- h2: `persistence`
- h3: `persistence`
- h4: `persistence`

Interpretation:

- The regime-aware features helped on the development folds across the whole short horizon.
- But on the untouched final block, every operational horizon still prefers the naive current-cases rule.
- The performance gap widens as horizon increases, which suggests that the challenger models are still too eager to learn rebound dynamics that do not hold up in the latest window.

## Geography story

Best holdout model by geography:

- `BAR`: `persistence`
- `CHA`: `persistence`
- `DHA_DIV_OUT_METRO`: `persistence`
- `DHA_METRO`: `persistence`
- `KHU`: `persistence`
- `MYM`: `persistence`
- `RAJ`: `persistence`
- `RAN`: `persistence`
- `SYL`: `seasonal_naive_52`

Interpretation:

- Persistence is best in `8` of `9` holdout geographies.
- `SYL` is the only exception, and the winner there is the very simple `seasonal_naive_52`, not a higher-capacity model.
- So the holdout failure is not confined to one problematic geography. It is broad.

## Bias story

On the holdout, most challenger models show large negative bias values, which here means predicted cases are above observed cases on average.

Examples:

- `conservative_stack` holdout bias: `-169.443`
- `residual_random_forest` holdout bias: `-234.765`
- `residual_elastic_net_global` holdout bias: `-240.002`
- `persistence` holdout bias: `-51.111`

Interpretation:

- All of the main operational models are somewhat too high on the holdout.
- Persistence is still too high, but much less so than the challengers.
- The regime-aware models are learning rebound or continuation behavior too aggressively relative to what happened in the final evaluation window.

This is especially visible at horizons `3` and `4`, where the challenger forecasts diverge upward fastest.

## What this means scientifically

This pass gives us a stronger paper result than a simple leaderboard win would have.

The correct result is now:

- richer autoregressive and regime-aware features improve development backtests
- a conservative persistence-anchored ensemble can beat persistence on development folds
- but once a genuinely untouched final window is introduced, persistence remains the most reliable short-horizon operational forecaster

That is a meaningful and publishable finding. It says the problem is difficult in exactly the way epidemic operational forecasting often is difficult: there is usable signal, but small-sample regime shifts make apparent improvements fragile.

## Operational decision after this pass

The primary operational model should be `persistence`.

Reason:

- it is the best model on the locked holdout
- it wins all operational holdout horizons
- it wins almost every holdout geography
- the current challenger family still overpredicts too strongly in the latest window

This decision is reflected in the refreshed live forecast bundle:

- Forecast run: `operational_current_20260318T231014Z`
- Primary model: `persistence`
- Origin week: `2025-09-15`

Country-total primary forecast:

- h1 = `0.0`
- h2 = `0.0`
- h3 = `0.0`
- h4 = `0.0`

## Next modeling step

The next pass should not be a blind search for more model capacity.

It should focus on:

- error analysis by outbreak phase and recent case regime
- selective challenger shrinkage toward persistence
- sparse-geo hurdle modeling
- explicit safeguards against overpredicting rebounds after low or zero current counts

If a challenger later beats persistence again, it must do so on the locked holdout logic, not only on development folds.
