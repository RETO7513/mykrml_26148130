"""Unit tests for the regression and multiclass evaluators."""
from mykrml_26148130.models.evaluation import evaluate_regressor, evaluate_multiclass


def test_regressor_perfect():
    m = evaluate_regressor([1, 2, 3], [1, 2, 3])
    assert m["MAE"] == 0.0 and m["R2"] == 1.0


def test_multiclass_perfect():
    m = evaluate_multiclass([0, 1, 2, 3], [0, 1, 2, 3])
    assert m["accuracy"] == 1.0 and m["macro_F1"] == 1.0


def test_multiclass_keys():
    m = evaluate_multiclass([0, 1, 1, 0], [0, 0, 1, 0])
    assert set(m) == {"accuracy", "balanced_acc", "macro_F1"}
