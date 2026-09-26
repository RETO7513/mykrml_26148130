# mykrml_26148130

Reusable ML utilities for UTS 36120 (Advanced Machine Learning Application).
**Author:** Rohan Yadav (Roh) — **Student ID:** 26148130

Originally built for AT1 (NBA career-longevity classification), extended for
AT2 (weather intelligence: Climate Comfort Index regression + Weather Hazard
Category classification for Sydney, using the Open-Meteo API).

## Install (from TestPyPI)
```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple mykrml_26148130
```

## Modules

### AT1 (retained)
- `data.sets` — `pop_target`, `save_sets`, `load_sets`, `subset_x_y`, `split_sets_by_time`, `split_sets_random`
- `features.dates` — `convert_to_date`
- `models.performance` — `print_regressor_scores`, `print_classifier_scores`, `print_auprc`, `assess_classifier_set`, `fit_assess_classifier`

### AT2 (new)
- `data.weather` — `fetch_hourly`, `aggregate_daily`, `get_daily_weather`, `split_by_date`, `time_split`
- `targets.weather` — `calculate_cci`, `calculate_whi`, `whi_to_class`, `add_targets`
- `features.timeseries` — `make_lags`, `make_rolling`, `make_calendar`, `build_features`, `shift_target`, `make_supervised`
- `models.evaluation` — `evaluate_regressor`, `evaluate_multiclass`, `confusion`
- `models.artifacts` — `save_model`, `load_model`

## Example
```python
from mykrml_26148130.data.weather import get_daily_weather
from mykrml_26148130.targets.weather import add_targets
from mykrml_26148130.features.timeseries import make_supervised

daily = get_daily_weather("2000-01-01", "2025-12-31", cache_path="sydney.csv")
labelled = add_targets(daily)
X_cci, y_cci = make_supervised(labelled, "CCI", horizon=3)
X_whc, y_whc = make_supervised(labelled, "WHC", horizon=7)
```

## Tests
```bash
pytest tests/ -q
```
