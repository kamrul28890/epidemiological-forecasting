# Article Notes

This file logs the current problem framing, EDA findings, and modeling decisions in a form we can reuse later in the article.

## Forecasting Task

- Task: weekly multi-series dengue case forecasting across Bangladesh geographies.
- Unit of prediction: one forecast for each `geo_id` and forecast week.
- Target: weekly dengue case counts on the original count scale.
- Primary use case: short-term operational forecasting.
- Secondary use case: medium-horizon experimental forecasting.

## Final Framing

- Primary horizon: 1 to 4 weeks ahead.
- Secondary horizon: 5 to 12 weeks ahead.
- Temporal grain: weekly.
- Spatial grain: `geo_id`.
- Primary KPI: MAE, macro-averaged across geography and horizon.
- Secondary KPI: RMSE on counts.
- Uncertainty: phase two. First stabilize point forecasts, then add prediction intervals.

## Data Profile

- Number of geographic series: 9.
- Case coverage: `2021-12-27` to `2025-09-15`.
- Climate coverage: `2021-12-20` to `2025-06-30`.
- Climate gap to latest case week: 11 weeks for every geography.
- The panel is complete at the weekly level with no missing weeks in the case series summary.

## Core Empirical Findings

### 1. The task is short-horizon and autoregressive first

- Lag-1 autocorrelation is very high across all geographies, roughly `0.95` to `0.98`.
- Persistence is the strongest baseline at every tested horizon from 1 to 12 weeks.
- This means recent case history is the strongest immediate signal in the current dataset.

### 2. The dataset is outbreak-driven and heteroskedastic

- All series show sharp epidemic waves rather than smooth low-variance dynamics.
- The 2023 wave is dominant in nearly every geography and heavily shapes the sample.
- Rolling mean and rolling standard deviation both move substantially over time.
- Variance-to-mean ratios are far above 1 in every geography, which is strong evidence of overdispersion.

### 3. Seasonality exists, but it is not clean enough to carry the model by itself

- Seasonal profiles rise mainly in late monsoon and post-monsoon weeks.
- Peak activity is concentrated mostly around ISO weeks 39 to 45, with some geography-specific variation.
- STL seasonal strength is present but not overwhelming.
- Lag-52 autocorrelation is weak to moderate, much weaker than lag-1 persistence.
- Seasonal naive performs much worse than persistence, so annual repetition alone is not a strong forecasting rule here.

### 4. Zero inflation differs a lot by geography

- `SYL` and `RAN` are highly sparse, with roughly half or more zero weeks.
- `RAJ` is also sparse relative to Dhaka-centered series.
- `DHA_DIV` and `DHA_METRO` have very low zero shares and dominate total volume.
- This scale imbalance means pooled metrics can hide weak performance on smaller geographies.

### 5. Weather is informative, but it is not ready to be the production backbone

- Climate variables show meaningful lead-lag association with cases.
- Dewpoint is the strongest recurring climate signal, often at lags around 8 to 10 weeks.
- That is useful for experimental modeling and mechanistic interpretation.
- However, climate features stop 11 weeks before the latest case observations, so they are currently incomplete for operational forecasting.
- Because of that coverage gap, weather should remain optional for now, not the core production signal.

## Hierarchy Fix

The current geography set contains an overlap:

- `DHA_DIV` includes `DHA_METRO`.
- Using both as separate bottom-level series creates a non-disjoint hierarchy.

Recommended fix:

- Keep `DHA_METRO` as one bottom-level node.
- Replace `DHA_DIV` with `dhaka_div_out_metro`.
- Use the disjoint bottom-level set:
  `BAR`, `CHA`, `DHA_METRO`, `dhaka_div_out_metro`, `KHU`, `MYM`, `RAJ`, `RAN`, `SYL`.

Implication:

- If we need coherent aggregation later, reconciliation becomes well-defined.
- For descriptive reporting, we can still show the original `DHA_DIV` series, but it should not be treated as an independent bottom-level series.

## Modeling Tracks

### Track A: Operational

- Goal: robust weekly forecasts for horizons 1 to 4.
- Core inputs: case lags, rolling summaries, seasonal/calendar indicators, geography identity.
- Weather: optional add-on only if coverage is refreshed and aligned.
- Success criterion: beat persistence on macro-averaged MAE and do so consistently across multiple geographies, not just high-volume ones.

### Track B: Experimental

- Goal: test whether longer-lead or richer-feature models add value.
- Horizon: 5 to 12 weeks, or 1 to 12 in side-by-side experiments.
- Inputs may include climate lags, hierarchy-aware features, and more flexible model classes.
- This track is for research and article insight first, not immediate deployment.

## Baseline Ladder

We should build from strong and interpretable baselines upward.

### Required baselines

- Persistence
- Seasonal naive
- Autoregressive count model

### Recommended first production candidates

- Global panel GBM with lag features
- Regularized count model with lag and calendar features

### Additional useful candidates

- Local per-geo autoregressive count models
- Global linear or elastic-net panel model on lagged features
- Simple hierarchical reconciliation layer after base forecasts are generated

### Deep learning note

- Deep learning is not forbidden, but it should be handled carefully.
- With about 195 weeks per series and only 9 series, the dataset is small for sequence models.
- Recurrent or transformer-style models can be run in the experimental track, but they should be treated as high-overfit-risk baselines rather than expected winners.

## Evaluation Protocol

- Use rolling-origin evaluation.
- Score each geography separately at each horizon.
- Then macro-average across geography and horizon.
- Keep MAE as the main ranking metric.
- Also report RMSE because missed spikes still matter.
- Include a per-geo results table so strong Dhaka performance cannot hide weak performance elsewhere.
- Keep a pooled total only as a secondary dashboard, not the main model selection criterion.

## Stationarity and Transformation Notes

- Raw counts are not stable in level or variance, which is expected for outbreak-driven incidence data.
- First differencing and `log1p` plus differencing improve stationarity in many geography-transform pairs.
- For machine learning models, feature engineering can absorb some of this behavior.
- For classical count or linear models, transformed targets or differenced features may be useful depending on the specification.

## Recommended Next Workflow

### Phase 1: Lock the target definition

- Freeze the operational forecasting task as weekly case forecasts for each `geo_id` at horizons 1 to 4.
- Freeze the evaluation rule as per-geo, per-horizon scoring with macro averages.
- Freeze the hierarchy decision before training reconciliation-aware models.

### Phase 2: Build the clean training panel

- Create the disjoint hierarchy version with `dhaka_div_out_metro`.
- Keep two panels:
  one operational panel without required weather,
  and one experimental panel with climate lags where coverage allows.
- Audit date alignment again after hierarchy repair.

### Phase 3: Run the baseline suite

- Persistence
- Seasonal naive
- Autoregressive count model
- Global lag-feature linear model
- Global lag-feature GBM

Expected outcome:

- We should establish a credible reference ladder and identify whether the global lag-feature models beat persistence at horizons 1 to 4.

### Phase 4: Promote one production candidate

- Choose the best short-horizon point forecast model.
- Preference order:
  1. stable gains over persistence,
  2. good performance across small and large geographies,
  3. operational simplicity.

Expected outcome:

- One model becomes the default operational forecaster for 1 to 4 week horizons.

### Phase 5: Run the experimental track

- Add climate covariates where available.
- Test longer horizons.
- Try deeper models only as controlled experiments with strict backtesting.

Expected outcome:

- We learn whether climate or higher-capacity models help enough to justify added complexity.

### Phase 6: Error analysis and article packaging

- Break down errors by geography, horizon, outbreak phase, and outbreak size.
- Document where persistence remains hard to beat.
- Convert the article notes, EDA findings, and modeling results into methods and results sections.

Expected outcome:

- We end up with a paper-ready narrative:
  this is a short-horizon, weekly, multi-geo count forecasting problem where strong autoregressive baselines matter, climate is promising but not yet operationally reliable, and hierarchy must be fixed before coherent aggregation is claimed.

## First Tracked Modeling Run

- Run ID: `modeling_tracks_20260317T032226Z`
- Run summary: `artifacts/runs/modeling_tracks_20260317T032226Z/summaries/run_summary.md`
- Human-readable report snapshot: `reports/modeling/baseline_suite_modeling_tracks_20260317T032226Z.md`

### What was implemented

- The hierarchy was fixed by replacing `DHA_DIV` with `DHA_DIV_OUT_METRO`.
- Two modeling tracks were materialized:
  operational for horizons 1 to 4 without weather as a required signal,
  and experimental for horizons 5 to 12 with weather features available.
- Detailed forecasts, metrics, fit-status logs, and config snapshots were written to a run-specific artifact directory.

### Results that matter

- Persistence is still the strongest validation baseline in both tracks.
- Operational validation leaderboard:
  persistence first, `ar_tweedie_global` second, `panel_hist_gbm` third, `seasonal_naive_52` last.
- Experimental validation leaderboard:
  persistence first, `panel_hist_gbm` second, `seasonal_naive_52` third, `ar_tweedie_global` fourth.
- Seasonal naive remains clearly too weak to be a serious operational benchmark.
- The panel GBM is useful as a benchmark, but in its current form it does not beat persistence.

### Important modeling observation

- The regularized count model needed a stability guard in some fold-horizon combinations during major outbreak regime shifts.
- When its predictions became numerically implausible, the pipeline logged the issue and fell back to persistence for those subsets.
- This means the count-model idea is still worth studying, but it is not yet production-ready in its current specification.

### Current implication

- The data and evaluation pipeline are now in place.
- The next improvement phase should focus on better lag-feature design, model tuning, and outbreak-regime handling rather than on making weather the center of the production stack.

## Second Tracked Modeling Run

- Run ID: `modeling_tracks_20260317T035929Z`
- Run summary: `artifacts/runs/modeling_tracks_20260317T035929Z/summaries/run_summary.md`

### What changed

- Added persistence-anchored challenger models for the operational track:
  `residual_hist_gbm` and `residual_random_forest`.

### What changed in the ranking

- Persistence still stayed first on the operational track.
- `residual_random_forest` moved into second place on operational validation.
- `ar_tweedie_global` moved to third.
- No challenger beat persistence at horizons 1 to 4.

### Interpretation

- The operational task is still dominated by very strong short-run momentum.
- Learning residual corrections around persistence is more sensible than free-running models, but even that is not enough yet to displace the naive current-case baseline.

## Current Operational Forecast Run

- Forecast run ID: `operational_current_20260317T040253Z`
- Forecast summary: `reports/forecasts/operational_current_operational_current_20260317T040253Z.md`

### Forecasting decision

- The primary operational model is currently `persistence` because it remains the best backtested 1 to 4 week model.
- Challenger forecasts were also generated from `residual_random_forest`, `ar_tweedie_global`, `residual_hist_gbm`, `panel_hist_gbm`, and `seasonal_naive_52`.

### Current forecast interpretation

- The latest modeled origin week is `2025-09-15`.
- Under the primary persistence model, the 1 to 4 week country-total forecast is flat at zero because the latest observed weekly counts at the origin are zero across the modeled bottom-level geographies.
- Challenger models allow small rebounds, but the backtest evidence does not currently justify promoting them over persistence.

## Paper Hardening Assets

- Manuscript outline: `reports/paper/manuscript_outline.md`
- Latest paper asset bundle: `reports/paper/paper_assets_20260318T003344Z`

### What is now available for writing

- methods table
- dataset summary table
- operational and experimental leaderboard tables
- fold-winner and horizon-winner tables
- geography-level model summary tables
- current forecast tables
- manuscript-facing figures for leaderboard, horizon behavior, fold behavior, geography behavior, and current operational forecasts

### Implication

- The project now has enough structure to begin drafting a full paper with a real results section, not just notes.

## First LaTeX Manuscript Draft

- Draft source: `reports/paper/manuscript_draft.tex`
- First compiled PDF: `reports/paper/manuscript_draft.pdf`

### What is already written

- Abstract
- Introduction
- data and problem-definition section
- EDA and time-series diagnostics section
- methods section covering target variables, predictors, direct forecasting strategy, and validation
- results section covering operational and experimental tracks
- discussion, limitations, and conclusion

### Current status

- The manuscript is now in a proper `.tex` file rather than notes-only markdown.
- The first draft compiles successfully and uses the generated paper figures from `reports/paper/paper_assets_20260318T003344Z/figures`.
- The next writing passes should focus on citations, author metadata, stronger literature positioning, and polishing tables/figures for the target venue.

## Manuscript Upgrade For Submission-Grade Drafting

- The manuscript source was upgraded from an internal-results narrative into a literature-grounded paper draft.
- A bibliography file was added at `reports/paper/references.bib`.
- The LaTeX build now compiles end to end with citations and figures into `reports/paper/manuscript_draft.pdf`.

### What improved

- stronger introduction and study motivation
- related-work and contribution framing
- clearer methods language for predictors, targets, validation, and operational safeguards
- richer results interpretation across horizon, fold, and geography
- journal-oriented back matter for data availability, code availability, competing interests, funding, and acknowledgements

### Important note

- This is still a journal-agnostic manuscript draft.
- Once a target journal is chosen, the next step should be porting the content into that journal's official LaTeX template and tightening the reference style, abstract structure, and submission metadata accordingly.
