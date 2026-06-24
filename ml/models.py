"""Model definitions and (de)serialization helpers.

Three forecasting models are supported:
  * Linear Regression   (scikit-learn pipeline w/ scaling) -- baseline
  * Random Forest       (scikit-learn)
  * LSTM                (tensorflow-cpu, optional / lightweight)

TensorFlow is imported lazily so the rest of the system works even if TF is not
installed or fails to import on a constrained environment.
"""
import json
import logging

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import Config

logger = logging.getLogger(__name__)

# Flag exposed so callers can report whether the LSTM path is available.
TENSORFLOW_AVAILABLE = None


def _check_tensorflow():
    global TENSORFLOW_AVAILABLE
    if TENSORFLOW_AVAILABLE is not None:
        return TENSORFLOW_AVAILABLE
    try:
        import tensorflow  # noqa: F401
        TENSORFLOW_AVAILABLE = True
    except Exception as exc:  # noqa: BLE001
        logger.warning("TensorFlow not available: %s", exc)
        TENSORFLOW_AVAILABLE = False
    return TENSORFLOW_AVAILABLE


# --------------------------------------------------------------------------- #
# Scikit-learn models
# --------------------------------------------------------------------------- #
def build_linear_regression():
    """Linear Regression baseline inside a scaling pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])


def build_random_forest():
    """A modest Random Forest, sized to run on a small VPS."""
    return RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=42,
    )


def save_sklearn_model(model, path):
    joblib.dump(model, path)


def load_sklearn_model(path):
    return joblib.load(path)


# --------------------------------------------------------------------------- #
# LSTM (tensorflow-cpu) -- optional
# --------------------------------------------------------------------------- #
def build_lstm(seq_len, n_features=1):
    """A deliberately small LSTM so training is fast on CPU.

    Returns a compiled Keras model, or None if TensorFlow is unavailable.
    """
    if not _check_tensorflow():
        return None
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Input

    model = Sequential([
        Input(shape=(seq_len, n_features)),
        LSTM(32),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def save_lstm_model(model, path):
    model.save(path)


def load_lstm_model(path):
    if not _check_tensorflow():
        return None
    from tensorflow.keras.models import load_model
    return load_model(path)


# --------------------------------------------------------------------------- #
# Selected-model metadata
# --------------------------------------------------------------------------- #
def save_selected_model(meta, path=None):
    path = path or Config.SELECTED_MODEL_PATH
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)


def load_selected_model(path=None):
    path = path or Config.SELECTED_MODEL_PATH
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
