"""Unit tests for leakage-safe time-series features."""
import numpy as np
import pandas as pd
from mykrml_26148130.features.timeseries import (
    make_lags, make_rolling, make_calendar, shift_target, make_supervised, WEATHER_COLS,
)


def _frame(n=40):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    data = {c: np.arange(n, dtype=float) for c in WEATHER_COLS}
    return pd.DataFrame(data, index=idx)


def test_lag_shifts_correctly():
    df = _frame()
    lags = make_lags(df, ["temperature_2m"], lags=(1,))
    # lag1 at row t equals raw value at t-1
    assert lags["temperature_2m_lag1"].iloc[5] == df["temperature_2m"].iloc[4]


def test_rolling_excludes_today():
    df = _frame()
    roll = make_rolling(df, ["temperature_2m"], windows=(3,))
    # rollmean3 at row t = mean of t-3..t-1 (today excluded)
    expected = df["temperature_2m"].iloc[2:5].mean()  # rows 2,3,4 feed row 5
    assert np.isclose(roll["temperature_2m_rollmean3"].iloc[5], expected)


def test_calendar_ranges():
    cal = make_calendar(_frame())
    assert cal["month"].between(1, 12).all()
    assert cal["doy_sin"].between(-1, 1).all()


def test_shift_target_is_future():
    s = pd.Series(np.arange(10, dtype=float))
    shifted = shift_target(s, 3)
    assert shifted.iloc[0] == s.iloc[3]


def test_make_supervised_aligned_and_no_nan():
    df = _frame()
    df["CCI"] = np.arange(len(df), dtype=float)
    X, y = make_supervised(df, "CCI", horizon=3)
    assert len(X) == len(y)
    assert not X.isna().any().any()
    assert not y.isna().any()
