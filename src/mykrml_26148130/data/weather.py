"""Open-Meteo historical weather ingestion and daily aggregation.

The archive API is queried for HOURLY variables which are then aggregated to
DAILY using the aggregation rules required by the AT2 target formulas. Keeping
this logic in the package guarantees the experimentation notebooks and the
deployed FastAPI service build their data identically.
"""
from __future__ import annotations
import os
import pandas as pd
import requests

SYDNEY_LAT = -33.8688
SYDNEY_LON = 151.2093
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_gusts_10m",
    "cloud_cover",
    "precipitation",
    "snowfall",
]

# Daily aggregation rules mandated by the assignment.
DAILY_AGG = {
    "temperature_2m": "mean",
    "relative_humidity_2m": "mean",
    "wind_speed_10m": "mean",
    "cloud_cover": "mean",
    "precipitation": "sum",
    "snowfall": "sum",
    "wind_gusts_10m": "max",
}


def fetch_hourly(start, end, lat=SYDNEY_LAT, lon=SYDNEY_LON, variables=None, timeout=60):
    """Retrieve hourly weather for a date range as a DataFrame indexed by local time."""
    variables = variables or HOURLY_VARS
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(variables),
        "timezone": "Australia/Sydney",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
        "temperature_unit": "celsius",
    }
    resp = requests.get(ARCHIVE_URL, params=params, timeout=timeout)
    resp.raise_for_status()
    hourly = resp.json()["hourly"]
    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])
    return df.set_index("time")


def aggregate_daily(hourly_df, agg=None):
    """Collapse hourly observations to daily using the assignment's aggregation rules."""
    agg = agg or DAILY_AGG
    daily = hourly_df.resample("D").agg(agg)
    return daily.dropna(how="all")


def get_daily_weather(start, end, cache_path=None, use_cache=True, **kwargs):
    """Return daily weather, using a local cache when available.

    The Open-Meteo call remains the source of truth; the cache is only a
    convenience for repeated notebook runs. Delete the cache file to force a
    fresh fetch. In production (the API) call with use_cache=False.
    """
    if use_cache and cache_path and os.path.exists(cache_path):
        return pd.read_csv(cache_path, parse_dates=["time"]).set_index("time")
    daily = aggregate_daily(fetch_hourly(start, end, **kwargs))
    if cache_path:
        daily.to_csv(cache_path)
    return daily


def split_by_date(df, cutoff="2026-01-01"):
    """Split into (training, production) on the date cutoff.

    Everything before `cutoff` is available for train/val/test; everything from
    `cutoff` onward is production data and must not be used during the project.
    """
    train = df[df.index < cutoff]
    production = df[df.index >= cutoff]
    return train, production


def time_split(X, y, train_end, val_end, embargo=0):
    """Chronological train/val/test split with an optional embargo gap.

    Weather is a continuous physical process: consecutive days are strongly
    autocorrelated, and features here draw on multi-day rolling windows while
    the target is shifted forward by the forecast horizon. A day just after a
    split boundary would therefore build its features from observations that
    fall in the previous block, letting information bleed across the cut. The
    ``embargo`` removes a buffer of that many days immediately after each
    boundary so no evaluation row is constructed from data belonging to an
    earlier split. Set it to at least (longest feature window + forecast
    horizon); with 14-day windows and a 7-day horizon, 21 days is sufficient.
    """
    import pandas as pd

    train_end = pd.Timestamp(train_end)
    val_end = pd.Timestamp(val_end)
    gap = pd.Timedelta(days=embargo)

    tr = X.index <= train_end
    va = (X.index > train_end + gap) & (X.index <= val_end)
    te = X.index > val_end + gap
    return (X[tr], y[tr]), (X[va], y[va]), (X[te], y[te])
