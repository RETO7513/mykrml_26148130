"""Evaluation helpers for the AT2 regression and classification targets.

Extends the AT1 binary-classifier scoring (models/performance.py) with a
regression evaluator and a multiclass evaluator suited to the imbalanced
Weather Hazard Category task (macro-F1 and balanced accuracy lead).
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)


def evaluate_regressor(y_true, y_pred, digits=3):
    """Return MAE, RMSE and R2 for a regression target."""
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), digits),
        "RMSE": round(float(mean_squared_error(y_true, y_pred) ** 0.5), digits),
        "R2": round(float(r2_score(y_true, y_pred)), digits),
    }


def evaluate_multiclass(y_true, y_pred, digits=3):
    """Return accuracy, balanced accuracy and macro-F1 for a multiclass target.

    Under heavy imbalance, accuracy is misleading; macro_F1 and balanced_acc are
    the honest headline metrics.
    """
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), digits),
        "balanced_acc": round(float(balanced_accuracy_score(y_true, y_pred)), digits),
        "macro_F1": round(float(f1_score(y_true, y_pred, average="macro")), digits),
    }


def confusion(y_true, y_pred, labels=None):
    """Confusion matrix as a plain numpy array (rows=actual, cols=predicted)."""
    return confusion_matrix(y_true, y_pred, labels=labels)
