"""Forecast evaluation metrics: MAE, RMSE, MAPE."""
import numpy as np


def mae(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    if len(y_true) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%). Ignores zero-actual points to avoid
    division by zero, which is common with intermittent pharmacy demand."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mask = y_true != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(y_true, y_pred):
    """Return a metrics dict for one model."""
    return {
        "mae": round(mae(y_true, y_pred), 4),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mape": round(mape(y_true, y_pred), 4),
    }


def skill_score(model_mae, baseline_mae):
    """Forecast skill vs a naive baseline.

    1.0 = perfect, 0.0 = no better than the naive baseline, <0 = worse than naive.
    This is the honest way to judge a forecast: beating a naive baseline is the bar.
    """
    if baseline_mae in (None, 0) or np.isnan(baseline_mae) or np.isnan(model_mae):
        return None
    return round(1.0 - (model_mae / baseline_mae), 4)
