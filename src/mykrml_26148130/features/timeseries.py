"""Leakage-safe time-series feature engineering for AT2.

All features are strictly backward-looking: for a prediction made on day t,
only information up to and including day t is used. Rolling windows shift by one
day before aggregating so the current day is excluded. Targets are shifted
forward by the forecast horizon (CCI +3, WHC +7).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

WEATHER_COLS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "cloud_cover",
    "precipitation",
    "wind_gusts_10m",
    "snowfall",
]


def make_lags(df, cols=WEATHER_COLS, lags=(1, 2, 3, 7)):
    """Lagged values of each column."""
    return pd.DataFrame(
        {f"{c}_lag{L}": df[c].shift(L) for c in cols for L in lags},
        index=df.index,
    )


def make_rolling(df, cols=WEATHER_COLS, windows=(3, 7, 14)):
    """Backward-looking rolling mean/max/min (current day excluded)."""
    out = {}
    for c in cols:
        base = df[c].shift(1)  # exclude today -> no leakage
        for w in windows:
            out[f"{c}_rollmean{w}"] = base.rolling(w).mean()
            out[f"{c}_rollmax{w}"] = base.rolling(w).max()
            out[f"{c}_rollmin{w}"] = base.rolling(w).min()
    return pd.DataFrame(out, index=df.index)


def make_calendar(df):
    """Month and cyclical day-of-year features for seasonality."""
    doy = df.index.dayofyear
    return pd.DataFrame(
        {
            "month": df.index.month,
            "doy_sin": np.sin(2 * np.pi * doy / 365.25),
            "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        },
        index=df.index,
    )


def build_features(df, cols=WEATHER_COLS):
    """Assemble the shared, backward-looking feature pool used by both targets."""
    return pd.concat(
        [make_lags(df, cols), make_rolling(df, cols), make_calendar(df)], axis=1
    )


def shift_target(series, horizon):
    """Shift a target series so row t holds the value at t+horizon (the forecast)."""
    return series.shift(-horizon)


def make_supervised(labelled, target_col, horizon, cols=WEATHER_COLS):
    """Build (X, y) for one target: shared features + horizon-shifted target.

    Rows with NaNs from the feature warmup (lags/rolling) and the target shift
    tail are dropped, so X and y are aligned and complete.
    """
    X = build_features(labelled, cols)
    y = shift_target(labelled[target_col], horizon)
    data = X.copy()
    data["__y__"] = y
    data = data.dropna()
    return data.drop(columns="__y__"), data["__y__"]
