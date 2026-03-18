# EDA Summary

## Dataset Overview
- Geographies: 9
- Case range: 2021-12-27 to 2025-09-15
- Weather range: 2021-12-20 to 2025-06-30
- Climate coverage gap to latest cases: 11 weeks

## Key Findings
- Highest zero-inflation geographies: SYL, RAN, RAJ
- Highest overdispersion geographies: DHA_DIV, DHA_METRO, KHU
- Strongest STL seasonality: KHU, DHA_METRO, DHA_DIV
- Number of geo/transform pairs flagged as stationary candidates: 36

## Hierarchy Checks
- dhaka_division_equals_metro_plus_out_metro: max_abs_diff = 0.000
- country_total_equals_bottom_level_sum: max_abs_diff = 70.000

## Climate Coverage Gaps
- BAR: 11 weeks
- CHA: 11 weeks
- DHA_DIV: 11 weeks
- DHA_METRO: 11 weeks
- KHU: 11 weeks
- MYM: 11 weeks
- RAJ: 11 weeks
- RAN: 11 weeks
- SYL: 11 weeks

## Best Baseline By Horizon (MAE)
- Horizon 1: persistence (MAE=73.62, RMSE=150.58)
- Horizon 2: persistence (MAE=131.61, RMSE=263.78)
- Horizon 3: persistence (MAE=188.36, RMSE=365.22)
- Horizon 4: persistence (MAE=239.28, RMSE=453.74)
- Horizon 5: persistence (MAE=285.34, RMSE=529.86)
- Horizon 6: persistence (MAE=332.56, RMSE=600.29)
- Horizon 7: persistence (MAE=381.57, RMSE=669.03)
- Horizon 8: persistence (MAE=431.78, RMSE=738.12)
- Horizon 9: persistence (MAE=478.32, RMSE=804.00)
- Horizon 10: persistence (MAE=522.94, RMSE=864.51)
- Horizon 11: persistence (MAE=562.27, RMSE=918.83)
- Horizon 12: persistence (MAE=598.27, RMSE=966.01)

## Strongest Climate Lead-Lag Associations
- DHA_DIV / dewpoint: lag 8, spearman=0.838
- DHA_METRO / dewpoint: lag 8, spearman=0.837
- CHA / dewpoint: lag 8, spearman=0.793
- MYM / dewpoint: lag 8, spearman=0.777
- KHU / dewpoint: lag 10, spearman=0.768
