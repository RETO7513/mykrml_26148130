"""Tests for the chronological split and its embargo gap."""
import numpy as np, pandas as pd
from mykrml_26148130.data.weather import time_split


def _xy(n=400):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    X = pd.DataFrame({"f": np.arange(n)}, index=idx)
    y = pd.Series(np.arange(n), index=idx)
    return X, y


def test_split_is_chronological():
    X, y = _xy()
    (X_train, _), (X_val, _), (X_test, _) = time_split(X, y, "2020-06-30", "2020-09-30")
    assert X_train.index.max() <= pd.Timestamp("2020-06-30")
    assert X_val.index.min() > pd.Timestamp("2020-06-30")
    assert X_test.index.min() > pd.Timestamp("2020-09-30")


def test_embargo_removes_boundary_rows():
    X, y = _xy()
    (_, a0), (b0, _), (c0, _) = time_split(X, y, "2020-06-30", "2020-09-30", embargo=0)
    (_, a1), (b1, _), (c1, _) = time_split(X, y, "2020-06-30", "2020-09-30", embargo=10)
    # val and test each lose 10 leading days
    assert len(b1) == len(b0) - 10
    assert len(c1) == len(c0) - 10


def test_embargo_gap_present():
    X, y = _xy()
    (_, _), (X_val, _), (_, _) = time_split(X, y, "2020-06-30", "2020-09-30", embargo=10)
    # first val day is 10 days after the train_end boundary
    assert X_val.index.min() == pd.Timestamp("2020-07-11")
