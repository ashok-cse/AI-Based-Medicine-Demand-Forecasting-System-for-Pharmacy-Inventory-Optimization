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


def conformal_offsets(residuals, level=0.8):
    """Split-conformal prediction-interval offsets from held-out residuals.

    Returns (lo_offset, hi_offset) such that [pred + lo, pred + hi] is a `level`
    (e.g. 80%) prediction interval. Uses the empirical quantiles of the
    residuals (actual - predicted), so the interval is distribution-free and can
    be asymmetric -- appropriate for skewed, non-negative demand.
    """
    residuals = np.asarray(residuals, float)
    residuals = residuals[~np.isnan(residuals)]
    if len(residuals) < 10:
        return None
    alpha = 1.0 - level
    lo = float(np.quantile(residuals, alpha / 2))
    hi = float(np.quantile(residuals, 1 - alpha / 2))
    return lo, hi


def interval_coverage(y_true, y_pred, lo_offset, hi_offset):
    """Empirical coverage: fraction of actuals inside [pred+lo, pred+hi].

    For a valid 80% interval this should be ~0.80. Reporting it is the honest
    test of whether the prediction interval means anything.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    lower = y_pred + lo_offset
    upper = y_pred + hi_offset
    inside = (y_true >= lower) & (y_true <= upper)
    return round(float(np.mean(inside)), 4) if len(y_true) else None


def skill_score(model_mae, baseline_mae):
    """Forecast skill vs a naive baseline.

    1.0 = perfect, 0.0 = no better than the naive baseline, <0 = worse than naive.
    This is the honest way to judge a forecast: beating a naive baseline is the bar.
    """
    if baseline_mae in (None, 0) or np.isnan(baseline_mae) or np.isnan(model_mae):
        return None
    return round(1.0 - (model_mae / baseline_mae), 4)
