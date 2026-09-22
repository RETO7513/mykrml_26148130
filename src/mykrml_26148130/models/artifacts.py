"""Persist and load model artefacts bundled with their metadata.

A saved artefact is a dict {'model', 'metadata'}. The metadata carries exactly
what the API /model-metadata endpoint must report (target, algorithm, horizon,
features, feature importance, performance) so the service never recomputes it.
"""
from __future__ import annotations
import joblib


def save_model(model, metadata, path):
    """Save an estimator plus its metadata dict to `path` (joblib)."""
    joblib.dump({"model": model, "metadata": metadata}, path)
    return path


def load_model(path):
    """Load a {'model', 'metadata'} bundle saved by save_model."""
    return joblib.load(path)
