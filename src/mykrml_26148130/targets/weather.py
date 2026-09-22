"""Derivation of the two AT2 target variables from daily weather.

  calculate_cci   -> Climate Comfort Index (regression target, 0-100)
  calculate_whi   -> Weather Hazard Index  (continuous score behind WHC)
  whi_to_class    -> map WHI to the 4 hazard classes (0..3)
  add_targets     -> attach CCI, WHI, WHC columns to a daily frame

These are the single source of truth for target creation and are imported by
both the experimentation notebooks and the FastAPI service.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def calculate_cci(d):
    """Climate Comfort Index (0-100). Higher = more comfortable."""
    temp_score = np.maximum(0, 1 - (d["temperature_2m"] - 22).abs() / 20)
    humid_score = np.maximum(0, 1 - (d["relative_humidity_2m"] - 50).abs() / 50)
    wind_score = np.maximum(0, 1 - (d["wind_speed_10m"] - 10).abs() / 40)
    cloud_score = np.maximum(0, 1 - d["cloud_cover"] / 100)
    rain_score = np.maximum(0, 1 - d["precipitation"] / 20)
    return 100 * (
        0.35 * temp_score
        + 0.20 * humid_score
        + 0.15 * wind_score
        + 0.15 * cloud_score
        + 0.15 * rain_score
    )


def calculate_whi(d):
    """Weather Hazard Index (0-100). Higher = more hazardous."""
    rain_haz = np.minimum(d["precipitation"] / 30, 1)
    wind_haz = np.minimum(d["wind_gusts_10m"] / 100, 1)
    cloud_haz = np.minimum(d["cloud_cover"] / 100, 1)
    snow_haz = np.minimum(d["snowfall"] / 15, 1)
    temp_haz = np.minimum((d["temperature_2m"] - 22).abs() / 25, 1)
    return 100 * (
        0.30 * rain_haz
        + 0.30 * wind_haz
        + 0.20 * cloud_haz
        + 0.10 * snow_haz
        + 0.10 * temp_haz
    )


def whi_to_class(whi):
    """Bin WHI into hazard classes: 0 Low, 1 Moderate, 2 High, 3 Extreme.

    Boundaries are left-closed [lower, upper) exactly as the brief specifies:
    WHI<25 -> 0, 25<=WHI<50 -> 1, 50<=WHI<75 -> 2, WHI>=75 -> 3.
    """
    return pd.cut(
        whi, bins=[-0.1, 25, 50, 75, 1e9], labels=[0, 1, 2, 3], right=False
    ).astype(int)


def add_targets(daily_df):
    """Return a copy of daily_df with CCI, WHI and WHC columns attached."""
    out = daily_df.copy()
    out["CCI"] = calculate_cci(out)
    out["WHI"] = calculate_whi(out)
    out["WHC"] = whi_to_class(out["WHI"])
    return out
