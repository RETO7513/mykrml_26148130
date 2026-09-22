"""Unit tests for the AT2 target derivation — verify formulas against hand calcs."""
import pandas as pd
from mykrml_26148130.targets.weather import calculate_cci, calculate_whi, whi_to_class, add_targets


def _row(**kw):
    base = dict(temperature_2m=22.0, relative_humidity_2m=50.0, wind_speed_10m=10.0,
                cloud_cover=0.0, precipitation=0.0, wind_gusts_10m=0.0, snowfall=0.0)
    base.update(kw)
    return pd.DataFrame([base])


def test_cci_perfect_day_is_100():
    # temp=22, humid=50, wind=10, cloud=0, rain=0 -> all component scores = 1 -> CCI 100
    d = _row()
    assert round(float(calculate_cci(d).iloc[0]), 6) == 100.0


def test_cci_clamps_at_zero():
    # extreme values push component scores below 0, which clamp to 0
    d = _row(temperature_2m=80, relative_humidity_2m=100, wind_speed_10m=200,
             cloud_cover=100, precipitation=100)
    assert float(calculate_cci(d).iloc[0]) == 0.0


def test_whi_calm_day_is_zero():
    d = _row(temperature_2m=22.0)  # no rain/wind/cloud/snow, temp at ideal -> WHI 0
    assert round(float(calculate_whi(d).iloc[0]), 6) == 0.0


def test_whi_components_cap_at_one():
    # very hazardous day: every sub-score saturates at 1 -> WHI 100
    d = _row(precipitation=60, wind_gusts_10m=200, cloud_cover=100,
             snowfall=30, temperature_2m=100)
    assert round(float(calculate_whi(d).iloc[0]), 6) == 100.0


def test_whi_to_class_boundaries():
    s = pd.Series([0, 24.9, 25, 49.9, 50, 74.9, 75, 100])
    classes = list(whi_to_class(s))
    assert classes == [0, 0, 1, 1, 2, 2, 3, 3]


def test_add_targets_columns():
    d = _row(precipitation=5, wind_gusts_10m=40)
    out = add_targets(d)
    for col in ("CCI", "WHI", "WHC"):
        assert col in out.columns
