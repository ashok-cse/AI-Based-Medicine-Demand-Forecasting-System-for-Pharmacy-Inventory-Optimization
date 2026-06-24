"""Unit tests for the forecast evaluation metrics (pure functions, no DB/TF)."""
import math

import numpy as np

from ml.evaluation import (
    mae, rmse, mape, evaluate, skill_score,
    conformal_offsets, interval_coverage,
)


def test_mae_basic():
    assert mae([2, 4, 6], [2, 4, 6]) == 0.0
    assert mae([1, 2, 3], [2, 2, 2]) == (1 + 0 + 1) / 3


def test_rmse_basic():
    # errors 3 and 4 -> sqrt((9+16)/2) = sqrt(12.5)
    assert math.isclose(rmse([0, 0], [3, 4]), math.sqrt(12.5))


def test_mape_ignores_zero_actuals():
    # zero-actual points are excluded to avoid divide-by-zero
    assert mape([0, 100], [50, 110]) == 10.0  # only the 100->110 point counts


def test_mape_all_zero_actuals_is_nan():
    assert math.isnan(mape([0, 0], [1, 2]))


def test_evaluate_returns_all_metrics():
    out = evaluate([10, 20, 30], [12, 18, 33])
    assert set(out) == {"mae", "rmse", "mape"}
    assert out["mae"] > 0


def test_skill_score():
    # model half the error of baseline -> 0.5 skill
    assert skill_score(2.0, 4.0) == 0.5
    # worse than baseline -> negative
    assert skill_score(5.0, 4.0) < 0
    # guards
    assert skill_score(1.0, 0) is None
    assert skill_score(float("nan"), 4.0) is None


def test_empty_inputs_are_nan():
    assert math.isnan(mae([], []))
    assert math.isnan(rmse([], []))


def test_conformal_offsets_and_coverage():
    rng = np.random.default_rng(0)
    # residuals ~ N(0, 1); an 80% interval should cover ~80% of fresh residuals
    residuals = rng.normal(0, 1, 5000)
    lo, hi = conformal_offsets(residuals, level=0.8)
    assert lo < 0 < hi
    # offsets should be near the +/-1.28 sigma quantiles of a standard normal
    assert abs(lo + 1.28) < 0.15
    assert abs(hi - 1.28) < 0.15
    # coverage of the same distribution should be close to 0.8
    y_pred = np.zeros(5000)
    y_true = rng.normal(0, 1, 5000)
    cov = interval_coverage(y_true, y_pred, lo, hi)
    assert 0.74 < cov < 0.86


def test_conformal_offsets_too_few():
    assert conformal_offsets([1, 2, 3], level=0.8) is None
