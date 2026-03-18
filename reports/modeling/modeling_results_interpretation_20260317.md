# Modeling Results Interpretation Report

This report explains what was checked in the time-series data, how the models were trained, what the dependent and independent variables were, how forecasting was done, and how to interpret the latest results.

It summarizes these tracked runs:

- Backtest run: `modeling_tracks_20260317T035929Z`
- Current forecast run: `operational_current_20260317T040253Z`

## 1. What problem was modeled

The forecasting task was defined as:

- Target: weekly dengue case counts
- Spatial unit: `geo_id`
- Temporal unit: weekly
- Primary operational horizon: 1 to 4 weeks ahead
- Experimental horizon: 5 to 12 weeks ahead
- Primary evaluation metric: macro-averaged MAE across geography and horizon
- Secondary metric: macro-averaged RMSE

The hierarchy was repaired before modeling:

- `DHA_DIV` was replaced by `DHA_DIV_OUT_METRO`
- This created a disjoint bottom-level geography set
- The closure check after repair was exact, with max closure error `0.0`

## 2. Did we determine whether the series were stationary?

Yes, but the answer is nuanced.

### Formal statistical test result

From the EDA stationarity table:

- All 9 raw series were flagged as stationary candidates under the combined ADF and KPSS check.
- 8 of 9 `log1p`-transformed series were also flagged as stationary candidates.
- All 9 first-differenced series were flagged as stationary candidates.
- Seasonal differencing at lag 52 was much less convincing and was not broadly helpful.

### Practical modeling interpretation

Even though the raw series passed the formal ADF/KPSS screening, we should not interpret them as smooth, stable textbook stationary processes.

Other diagnostics showed:

- very high short-lag autocorrelation
- strong outbreak pulses
- major shifts in rolling mean and rolling standard deviation
- heavy overdispersion
- strong scale changes across outbreak periods

So the correct interpretation is:

- the series are not trend-drifting in a simple deterministic way
- but they are still outbreak-driven, heteroskedastic, and highly persistent
- because of that, we did not rely on a simple stationarity assumption to justify the forecasting models

In short:

- formal tests: often compatible with stationarity
- practical behavior: still volatile enough that we treated them as unstable epidemic count series

## 3. What standard time-series checks were done

The EDA covered both standard time-series diagnostics and outbreak-forecasting diagnostics.

### Standard checks

- missing-week and coverage checks
- ADF and KPSS stationarity tests
- STL decomposition
- ACF diagnostics
- rolling mean and rolling standard deviation
- seasonal weekly profiles

### Forecasting-specific checks

- zero inflation by geography
- variance-to-mean ratio for overdispersion
- annual outbreak peak timing
- cross-geography correlation
- hierarchy consistency
- climate coverage gap
- climate lead-lag correlation
- simple baseline forecastability by horizon

The key conclusion from these checks was:

- short-run persistence is very strong
- seasonality exists, but does not dominate
- weather is potentially informative, but not reliable enough yet to be the operational backbone

## 4. What the dependent variables were

There were two kinds of dependent variables.

### For direct case models

Used by:

- `ar_tweedie_global`
- `panel_hist_gbm`

Dependent variable:

- `y_true`
- This is the observed dengue case count at `target_week`
- `target_week = origin_week + horizon`

### For residual correction models

Used by:

- `residual_hist_gbm`
- `residual_random_forest`

Dependent variable:

- `y_true - cases_at_origin`

This means those models did not try to forecast the full future count directly.
Instead, they tried to learn the correction around the persistence baseline.

Final prediction:

- `forecast = current_cases + predicted_residual`

### For rule-based baselines

Used by:

- `persistence`
- `seasonal_naive_52`

These did not have a learned dependent variable because they are rule-based baselines.

## 5. What the independent variables were

### Operational track inputs

Used in 1 to 4 week forecasting:

- current weekly cases at the forecast origin
- case lags: `1, 2, 3, 4, 8, 13, 26, 52`
- case rolling means: `2, 4, 8`
- `log1p(cases)`
- calendar terms:
  `week_of_year`, `woy_sin`, `woy_cos`, `monsoon_flag`
- `geo_id` as a categorical geography identifier

### Experimental track inputs

Used in 5 to 12 week forecasting:

- all operational inputs above
- plus lagged and rolled climate features
- climate variables:
  `rain`, `tmean`, `humidity`, `dewpoint`, `soil_moist_0_7cm`
- climate lags:
  `1, 2, 4, 8`
- climate rolling means and sums

## 6. What was intentionally left out

The following were intentionally not used in the operational production track:

- weather as a required production input
- recursive use of previous forecasts as features
- deep learning models
- prediction intervals

Reasons:

- climate coverage ends 11 weeks before the latest case coverage
- recursive forecasting would propagate error and was unnecessary at this stage
- deep learning is high-risk given the dataset size
- intervals are phase two after point forecasts stabilize

## 7. Did the multi-step forecast use previous forecasts?

No.

This is important.

The forecasting setup was direct multi-horizon, not recursive.

That means:

- horizon 1 had its own model or rule
- horizon 2 had its own model or rule
- horizon 3 had its own model or rule
- horizon 4 had its own model or rule

We did **not** do:

- forecast `f1`
- then feed `f1` into the model to get `f2`
- then feed `f1` and `f2` into the model to get `f3`
- and so on

Instead, every horizon was forecast directly from the latest observed history at the origin week.

Why that matters:

- no recursive error accumulation
- cleaner interpretation of horizon-specific performance
- safer for a short and noisy epidemic dataset

## 8. How training and validation were set up

Rolling-origin folds were used.

The main folds were:

- `F1`: validation in 2024
- `F2`: validation through 2025-06-30
- `F3`: validation in the first half of 2023
- `F4`: validation in the second half of 2023
- `F5`: validation in the second half of 2024

Why this matters:

- `F4` is especially important because it contains the major 2023 outbreak regime
- models that look good on calmer folds can fail badly there

## 9. How to interpret the operational training results

### Overall operational leaderboard

Validation macro metrics:

- `persistence`: MAE `154.218`, RMSE `206.390`
- `residual_random_forest`: MAE `179.371`, RMSE `234.354`
- `ar_tweedie_global`: MAE `181.767`, RMSE `241.761`
- `residual_hist_gbm`: MAE `183.082`, RMSE `249.362`
- `panel_hist_gbm`: MAE `296.576`, RMSE `367.618`
- `seasonal_naive_52`: MAE `520.177`, RMSE `714.417`

### Main interpretation

- Persistence is still the strongest operational model overall.
- Seasonal naive is much too weak, so annual repetition alone is not enough.
- The residual models are better challengers than the free-running panel GBM.
- The regularized count model is competitive but not best.

### Fold-level interpretation

The fold breakdown is important because the overall average hides different behaviors.

- `F1`: `panel_hist_gbm` was best.
- `F2`: `panel_hist_gbm` was best, with `ar_tweedie_global` close behind.
- `F3`: persistence was best.
- `F4`: `residual_hist_gbm` was best, with `residual_random_forest`, `ar_tweedie_global`, and persistence all clustered.
- `F5`: `panel_hist_gbm` was best.

What this means:

- the model-based challengers can beat persistence in calmer or more regular periods
- but they are not robust enough across all folds
- persistence wins overall because it is the most stable across regime changes

### Overfit and instability interpretation

The train-vs-validation gap also matters.

- `panel_hist_gbm` had extremely low train error but much worse validation error
- `ar_tweedie_global` needed a stability guard in some subsets
- the residual models trained reasonably, but still did not generalize well enough to beat persistence overall

Interpretation:

- some models are learning useful structure
- but they are still vulnerable to outbreak-regime shift
- we are not yet at the point where a learned model should replace persistence in production

## 10. How to interpret the experimental training results

### Overall experimental leaderboard

Validation macro metrics:

- `persistence`: MAE `403.016`, RMSE `510.986`
- `panel_hist_gbm`: MAE `516.184`, RMSE `692.385`
- `seasonal_naive_52`: MAE `521.907`, RMSE `716.453`
- `ar_tweedie_global`: MAE `540.113`, RMSE `714.009`

### Interpretation

- Weather-augmented experimental modeling did not beat persistence.
- That does not mean weather is useless.
- It means weather did not provide a strong enough, robust enough improvement in this setup.
- Weather may still help in explanation or in future refined models, but it is not yet a winning forecasting signal here.

## 11. How to interpret the current forecast run

Current forecast run:

- `operational_current_20260317T040253Z`
- origin week: `2025-09-15`

Primary operational model:

- `persistence`

Why it was chosen:

- it remained the best 1 to 4 week model in the latest operational backtest

### What the primary forecast says

- The country-total persistence forecast is `0.0` for horizons 1 to 4.

Interpretation:

- the latest observed bottom-level weekly counts at the origin were zero
- the best current operational model therefore projects a flat near-zero path

### What the challenger forecasts say

The challengers allow small rebounds:

- `residual_random_forest`: country totals rise from about `15` to `123`
- `ar_tweedie_global`: country totals rise from about `27` to `70`
- `residual_hist_gbm`: country totals rise from about `18` to `85`, then lower
- `panel_hist_gbm`: small rebound only
- `seasonal_naive_52`: very large rebound based on last year, but this baseline is weak in backtesting and should not be trusted operationally

Interpretation:

- challengers are expressing some rebound risk
- but the backtest evidence is not strong enough yet to justify replacing the primary persistence forecast

## 12. What the next step should be

The next step is not to jump to deep learning.

The next step should be:

- interpret errors by geography and outbreak phase
- identify where persistence is weakest
- tune the persistence-anchored challengers, especially `residual_random_forest` and `residual_hist_gbm`
- add outbreak-regime features rather than generic complexity
- only after a challenger beats persistence consistently should it be promoted to primary production forecasting

## 13. Bottom line

The current evidence says:

- the time series were formally tested for stationarity and other standard properties
- the models were trained on weekly future case counts, with carefully defined feature sets
- the production track used case-history and calendar information, not weather
- the forecasting setup was direct, not recursive
- persistence is still the most reliable short-horizon model
- challenger models are promising, but they are not yet strong enough to replace it
