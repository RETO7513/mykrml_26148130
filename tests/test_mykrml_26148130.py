"""Unit tests for the mykrml_26148130 package.

Run with:  pytest
"""
import os
import numpy as np
import pandas as pd
import pytest

from mykrml_26148130.data.sets import (
    pop_target, save_sets, load_sets, subset_x_y,
    split_sets_by_time, split_sets_random,
)
from mykrml_26148130.features.dates import convert_to_date
from mykrml_26148130.models.performance import (
    print_classifier_scores, print_auprc,
    assess_classifier_set, fit_assess_classifier,
)


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------
@pytest.fixture
def sample_df():
    """A small dataframe with a target column."""
    return pd.DataFrame({
        "feature_a": [1, 2, 3, 4, 5],
        "feature_b": [10, 20, 30, 40, 50],
        "target": [0, 1, 0, 1, 0],
    })


@pytest.fixture
def date_df():
    return pd.DataFrame({
        "d1": ["2020-01-01", "2021-06-15", "2019-12-31"],
        "d2": ["2000-05-05", "2010-10-10", "1995-03-03"],
        "other": [1, 2, 3],
    })


# --------------------------------------------------------------------------
# pop_target
# --------------------------------------------------------------------------
def test_pop_target_returns_features_and_target(sample_df):
    features, target = pop_target(df=sample_df, target_col="target")
    assert "target" not in features.columns
    assert list(target) == [0, 1, 0, 1, 0]
    assert features.shape == (5, 2)


def test_pop_target_does_not_mutate_original(sample_df):
    original_cols = list(sample_df.columns)
    _ = pop_target(df=sample_df, target_col="target")
    assert list(sample_df.columns) == original_cols  # original unchanged


# --------------------------------------------------------------------------
# save_sets / load_sets
# --------------------------------------------------------------------------
def test_save_and_load_sets_roundtrip(tmp_path):
    path = str(tmp_path) + "/"
    X_train = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    y_train = pd.Series([0, 1])
    X_val = pd.DataFrame({"a": [5], "b": [6]})
    y_val = pd.Series([1])

    save_sets(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, path=path)

    # files should exist
    assert os.path.isfile(path + "X_train.joblib")
    assert os.path.isfile(path + "y_train.joblib")

    loaded = load_sets(path=path)
    X_train_l, y_train_l, X_val_l, y_val_l, X_test_l, y_test_l = loaded

    pd.testing.assert_frame_equal(X_train_l, X_train)
    pd.testing.assert_series_equal(y_train_l, y_train)
    assert X_test_l is None  # never saved -> should be None


def test_load_sets_missing_returns_none(tmp_path):
    """Loading from an empty folder returns all None."""
    result = load_sets(path=str(tmp_path) + "/")
    assert all(item is None for item in result)


# --------------------------------------------------------------------------
# subset_x_y
# --------------------------------------------------------------------------
def test_subset_x_y_slices_both(sample_df):
    features = sample_df[["feature_a", "feature_b"]]
    target = sample_df["target"]
    f_sub, t_sub = subset_x_y(target=target, features=features,
                              start_index=1, end_index=3)
    assert len(f_sub) == 2
    assert len(t_sub) == 2
    assert list(t_sub) == [1, 0]


# --------------------------------------------------------------------------
# split_sets_by_time
# --------------------------------------------------------------------------
def test_split_sets_by_time_sizes():
    features = pd.DataFrame({"a": range(100)})
    target = pd.Series(range(100))
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_sets_by_time(
        features, target, test_ratio=0.2)
    # 20% test, 20% val, 60% train
    assert len(X_te) == 20
    assert len(X_val) == 20
    assert len(X_tr) == 60
    # order preserved: train is the earliest rows
    assert X_tr["a"].iloc[0] == 0


# --------------------------------------------------------------------------
# split_sets_random
# --------------------------------------------------------------------------
def test_split_sets_random_reproducible():
    features = pd.DataFrame({"a": range(100), "b": range(100, 200)})
    target = pd.Series([0, 1] * 50)
    out1 = split_sets_random(features, target, test_ratio=0.2, random_state=42)
    out2 = split_sets_random(features, target, test_ratio=0.2, random_state=42)
    # same seed -> identical splits
    pd.testing.assert_frame_equal(out1[0], out2[0])


def test_split_sets_random_no_overlap():
    features = pd.DataFrame({"a": range(100)})
    target = pd.Series([0, 1] * 50)
    X_tr, y_tr, X_val, y_val, X_te, y_te = split_sets_random(
        features, target, test_ratio=0.2, random_state=42)
    # the three sets should be disjoint and cover all rows
    total = len(X_tr) + len(X_val) + len(X_te)
    assert total == 100
    idx = set(X_tr.index) | set(X_val.index) | set(X_te.index)
    assert len(idx) == 100  # no duplicated rows across sets


# --------------------------------------------------------------------------
# convert_to_date
# --------------------------------------------------------------------------
def test_convert_to_date(date_df):
    out = convert_to_date(date_df, cols=["d1", "d2"])
    assert pd.api.types.is_datetime64_any_dtype(out["d1"])
    assert pd.api.types.is_datetime64_any_dtype(out["d2"])
    # untouched column stays the same
    assert list(out["other"]) == [1, 2, 3]


def test_convert_to_date_does_not_mutate_original(date_df):
    _ = convert_to_date(date_df, cols=["d1"])
    # original column still object/string dtype
    assert not pd.api.types.is_datetime64_any_dtype(date_df["d1"])


# --------------------------------------------------------------------------
# performance scoring functions (smoke tests: they print, return None)
# --------------------------------------------------------------------------
def test_print_classifier_scores_runs(capsys):
    y_actuals = [0, 1, 0, 1]
    y_preds = [0, 1, 1, 1]
    print_classifier_scores(y_preds=y_preds, y_actuals=y_actuals, set_name="Test")
    captured = capsys.readouterr()
    assert "Accuracy" in captured.out
    assert "F1" in captured.out


def test_print_auprc_runs(capsys):
    y_actuals = [0, 1, 0, 1]
    y_scores = [0.1, 0.9, 0.2, 0.8]
    print_auprc(y_scores=y_scores, y_actuals=y_actuals, set_name="Test")
    captured = capsys.readouterr()
    assert "AUPRC" in captured.out


def test_fit_assess_classifier_returns_model():
    from sklearn.linear_model import LogisticRegression
    X_train = pd.DataFrame({"a": range(20), "b": range(20, 40)})
    y_train = pd.Series([0, 1] * 10)
    X_val = pd.DataFrame({"a": range(5), "b": range(5, 10)})
    y_val = pd.Series([0, 1, 0, 1, 0])
    model = fit_assess_classifier(
        LogisticRegression(max_iter=200), X_train, y_train, X_val, y_val)
    # returns a fitted model that can predict
    assert hasattr(model, "predict")
    assert len(model.predict(X_val)) == 5