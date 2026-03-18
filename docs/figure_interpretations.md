# Figure Interpretations

This file translates the EDA figures into plain-language observations and explains what each figure means for the forecasting problem.

The figures were generated from:

- Cases through `2025-09-15`
- Climate covariates through `2025-06-30`

## Shared Figures

### [cases_by_geo_small_multiples.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/cases_by_geo_small_multiples.png)

What the figure shows:

- Weekly case trajectories for each geography on its own scale.

What it means here:

- The series are synchronized in broad timing, but not equal in scale.
- Nearly all geographies show a large common wave in 2023, then smaller later waves.
- Dhaka-centered geographies dominate volume, while `SYL` and `RAN` remain much smaller.
- This supports a global multi-series model because outbreak timing is related across geographies.
- It also warns us that pooled metrics will be dominated by the biggest series unless we macro-average.

Specific observations:

- `DHA_DIV` and `DHA_METRO` have the largest outbreak amplitudes by far.
- `BAR`, `CHA`, and `KHU` show clear multi-wave epidemic structure.
- `RAN` and `SYL` have low-count intermittent behavior with brief bursts rather than long plateaus.

### [weekly_seasonal_profiles.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/weekly_seasonal_profiles.png)

What the figure shows:

- Average cases by ISO week for each geography.

What it means here:

- Dengue activity is concentrated in a recurring seasonal window rather than evenly spread across the year.
- Most geographies start rising around mid-year and peak roughly in weeks 39 to 45.
- This confirms seasonality is real and should be represented in features.
- But the profiles are broad and uneven, not a single precise repeating spike, which is one reason seasonal naive alone is weak.

Specific observations:

- Dhaka-centered series have the strongest seasonal amplitude.
- `KHU` has one of the clearest seasonal ramps and one of the stronger STL seasonal signals.
- `SYL` has a smaller and flatter profile, consistent with its weak seasonal-strength estimate.

### [zero_share_by_geo.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/zero_share_by_geo.png)

What the figure shows:

- The share of zero-case weeks in each geography.

What it means here:

- The dataset mixes dense endemic-like series with sparse intermittent series.
- Models that work well on Dhaka may still fail on `SYL`, `RAN`, or `RAJ` because those series spend much more time at zero.
- This is a strong argument for per-geo evaluation and for count-aware modeling choices.

Specific observations:

- `SYL` and `RAN` have about half or more zero weeks.
- `DHA_DIV` and `DHA_METRO` have very few zero weeks, so they behave like continuous outbreak series.
- `MYM` sits in the middle and may benefit from methods that can handle both zeros and bursts.

### [overdispersion_by_geo.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/overdispersion_by_geo.png)

What the figure shows:

- Variance-to-mean ratios by geography.

What it means here:

- Every series is overdispersed relative to a simple Poisson assumption.
- The raw case process is much more volatile than its average level suggests.
- This supports Negative Binomial style count models, robust loss functions, or feature-based models that can absorb regime changes better than a naive Poisson setup.

Specific observations:

- `DHA_DIV` and `DHA_METRO` are the most overdispersed because they contain very large epidemic waves.
- `KHU`, `CHA`, and `BAR` are also strongly overdispersed.
- Even low-count series like `SYL` are not stable; they just fluctuate on a smaller scale.

### [cross_geo_correlation_heatmap.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/cross_geo_correlation_heatmap.png)

What the figure shows:

- Spearman correlation of case trajectories across geographies.

What it means here:

- Geographies move together strongly in rank-order outbreak timing.
- A global panel model is justified because there is shared temporal structure across places.
- The very high correlation between `DHA_DIV` and `DHA_METRO` also exposes the hierarchy overlap problem.

Specific observations:

- Most correlations are high, which suggests national-scale outbreak drivers and shared timing.
- `DHA_DIV` and `DHA_METRO` are almost perfectly correlated, consistent with the metro series being embedded inside the division series.
- `SYL` is still positively related to the others, but less strongly, which fits its sparser case behavior.

### [climate_gap_weeks.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/climate_gap_weeks.png)

What the figure shows:

- The number of weeks between the latest case observation and the latest climate covariate for each geography.

What it means here:

- The operational dataset is not climate-complete at the forecast origin.
- Any production model that requires current weather features will break or need imputation near the most recent dates.
- This is the clearest practical reason not to make climate the core production signal yet.

Specific observations:

- The gap is exactly 11 weeks in every geography.
- This is a data-pipeline problem, not a geography-specific modeling issue.

### [baseline_forecastability.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/baseline_forecastability.png)

What the figure shows:

- Macro-averaged MAE and RMSE by horizon for persistence and seasonal naive baselines.

What it means here:

- Short-horizon forecastability is dominated by recent momentum, not by same-week-last-year recurrence.
- Persistence wins at every horizon from 1 to 12 weeks.
- This tells us the first serious model must beat a very strong autoregressive benchmark before it is considered useful.

Specific observations:

- Persistence starts far below seasonal naive at 1 week ahead and remains better throughout the horizon range.
- Seasonal naive changes only gradually with horizon, which suggests it is anchored to a weak annual pattern rather than short-term outbreak dynamics.
- The growing persistence error curve still shows that difficulty rises quickly as horizon length increases.

### [climate_best_spearman_heatmap.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/climate_best_spearman_heatmap.png)

What the figure shows:

- The strongest rank-based climate association by geography and covariate.

What it means here:

- Climate is not random noise in this dataset.
- Dewpoint is the most consistently strong climate signal, followed in several places by temperature, humidity, rain, and soil moisture.
- That makes climate worth testing in the experimental track.
- It does not override the coverage-gap issue or the possibility of confounding with shared seasonality.

Specific observations:

- Dewpoint is the strongest and most consistent signal across almost all geographies.
- Wind direction is mostly negative and weaker, so it looks less useful as a main driver.
- `SYL` has weaker climate associations overall, which fits its noisier and sparser case series.

### [climate_best_lag_heatmap.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/climate_best_lag_heatmap.png)

What the figure shows:

- The lag in weeks where each climate covariate reaches its strongest association with cases.

What it means here:

- Climate effects, where present, are delayed rather than immediate.
- The strongest climate signals often occur around 8 to 12 weeks before cases.
- That pattern makes climate more promising for medium-lead experimental forecasting than for ultra-short 1-week forecasts.

Specific observations:

- Dewpoint peaks mostly around 8 to 10 weeks.
- Several temperature and wind-related covariates hit their best association at the largest tested lags, suggesting slower or more indirect links.
- The lag structure is not identical across all geographies, so a single fixed climate lag may be too simplistic.

## Per-Geography Diagnostic Sets

Each geography has three related figures:

- STL decomposition
- Autocorrelation function
- Rolling mean and rolling standard deviation

Together they answer three different questions:

- STL asks whether the series has recurring seasonal structure.
- ACF asks how much memory the series has at different lags.
- Rolling moments ask whether the raw process is stable in level and variance.

### BAR

Files:

- [stl_BAR.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/stl_BAR.png)
- [acf_BAR.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/acf_BAR.png)
- [rolling_moments_BAR.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/rolling_moments_BAR.png)

Interpretation:

- `BAR` has a clear large 2023 wave and smaller later waves, so the series is episodic rather than steady.
- The ACF decays slowly at short lags, which confirms strong autoregressive memory.
- The lag-52 signal is weak, so annual repetition exists only loosely.
- Rolling mean and volatility both jump during outbreaks, showing non-stationary raw dynamics.
- Forecast implication: lag features should help; pure seasonal rules will not be enough.

### CHA

Files:

- [stl_CHA.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/stl_CHA.png)
- [acf_CHA.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/acf_CHA.png)
- [rolling_moments_CHA.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/rolling_moments_CHA.png)

Interpretation:

- `CHA` shows one of the cleaner repeated epidemic shapes with large 2023 activity and smaller but visible later waves.
- The STL view indicates a real seasonal component, though not an overwhelmingly dominant one.
- The ACF remains high across several short lags, so recent cases should be a strong predictor.
- Rolling volatility expands sharply during the main outbreak period, again showing heteroskedasticity.
- Forecast implication: `CHA` is a good candidate for benefiting from both lag features and seasonal indicators.

### DHA_DIV

Files:

- [stl_DHA_DIV.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/stl_DHA_DIV.png)
- [acf_DHA_DIV.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/acf_DHA_DIV.png)
- [rolling_moments_DHA_DIV.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/rolling_moments_DHA_DIV.png)

Interpretation:

- `DHA_DIV` is the dominant high-volume series and carries the largest epidemic amplitude in the dataset.
- STL shows seasonal structure, but the raw level is mostly driven by outbreak regimes rather than a stable smooth trend.
- The ACF is extremely high at short lags, which is exactly why persistence is so hard to beat.
- Rolling standard deviation explodes during the major wave, which means errors on this geography can dominate pooled metrics.
- Forecast implication: this series is essential for operational performance, but evaluation must not let it drown out smaller geographies.

### DHA_METRO

Files:

- [stl_DHA_METRO.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/stl_DHA_METRO.png)
- [acf_DHA_METRO.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/acf_DHA_METRO.png)
- [rolling_moments_DHA_METRO.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/rolling_moments_DHA_METRO.png)

Interpretation:

- `DHA_METRO` closely mirrors `DHA_DIV`, which is expected because it is embedded in that higher-level geography.
- The ACF is extremely persistent at short lags and shows only limited annual rebound.
- The rolling plots show very large regime-dependent shifts in both mean and variance.
- The STL trend panel should not be overread as a stable structural decline; the overall trend strength is very weak and the decomposition is being shaped by a few extreme epidemic waves.
- Forecast implication: this is a strong signal series for autoregressive models, but it reinforces the need to repair the hierarchy before coherent aggregation claims are made.

### KHU

Files:

- [stl_KHU.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/stl_KHU.png)
- [acf_KHU.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/acf_KHU.png)
- [rolling_moments_KHU.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/rolling_moments_KHU.png)

Interpretation:

- `KHU` has one of the strongest seasonal signatures in the dataset.
- The seasonal rise and fall are visually clearer here than in many other geographies.
- The ACF remains high at short lags and shows a somewhat stronger annual echo than the weakest series.
- Rolling moments still move substantially, so the raw process is not stationary even though the seasonal shape is clearer.
- Forecast implication: `KHU` is a strong candidate for models that combine autoregression with explicit seasonal features.

### MYM

Files:

- [stl_MYM.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/stl_MYM.png)
- [acf_MYM.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/acf_MYM.png)
- [rolling_moments_MYM.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/rolling_moments_MYM.png)

Interpretation:

- `MYM` is a smaller-scale series, but it still shows a recognizable seasonal wave structure.
- The ACF is strong at short lags, so recent counts still carry useful predictive information.
- Rolling mean and volatility are much lower than the Dhaka series, but they still shift enough to reject simple stationarity assumptions on the raw scale.
- Forecast implication: this geography should benefit from the same lag-based framework, but the model must be calibrated for smaller counts and intermittent activity.

### RAJ

Files:

- [stl_RAJ.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/stl_RAJ.png)
- [acf_RAJ.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/acf_RAJ.png)
- [rolling_moments_RAJ.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/rolling_moments_RAJ.png)

Interpretation:

- `RAJ` is much sparser than the large geographies, but it still exhibits concentrated epidemic periods.
- The ACF remains high at short lags, meaning outbreak momentum matters once activity starts.
- The annual component is weaker than the short-lag structure.
- Rolling variance spikes during active periods, so forecast difficulty changes across the year.
- Forecast implication: this is exactly the kind of geography where macro-averaged evaluation matters, because performance here can look poor even if Dhaka looks strong.

### RAN

Files:

- [stl_RAN.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/stl_RAN.png)
- [acf_RAN.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/acf_RAN.png)
- [rolling_moments_RAN.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/rolling_moments_RAN.png)

Interpretation:

- `RAN` is highly intermittent, with many zeros and short bursts of activity.
- The ACF still shows strong short memory, but the seasonal structure is weaker and less stable than in larger geographies.
- Rolling moments show that most of the variance is concentrated in a few outbreak windows.
- Forecast implication: `RAN` is difficult because it combines sparsity with burstiness; models should be judged on whether they avoid both false alarms and missed bursts.

### SYL

Files:

- [stl_SYL.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/stl_SYL.png)
- [acf_SYL.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/acf_SYL.png)
- [rolling_moments_SYL.png](/d:/My Projects/dengue-forecast-starter-pack/reports/eda/figures/rolling_moments_SYL.png)

Interpretation:

- `SYL` is the sparsest and weakest seasonal series in the panel.
- It still has an outbreak pulse, but the raw counts are low and many weeks are zero.
- The ACF starts high at lag 1 but loses structure faster than the denser geographies.
- Rolling moments stay low most of the time and spike only during short active windows.
- Forecast implication: `SYL` is a stress test for generalization. A model that only learns high-volume outbreak behavior may perform poorly here.

## Overall Takeaway From The Figure Set

Taken together, the figures support the same operational conclusion:

- This is a weekly multi-geo count forecasting problem with strong short-term persistence.
- The raw series are non-stationary and overdispersed.
- Seasonality exists, but it is not strong enough to replace autoregressive structure.
- Geography scale differs sharply, so evaluation must be macro-oriented.
- Climate is promising for research, especially with medium lags, but the current data gap makes it unsuitable as the backbone of the production model.
