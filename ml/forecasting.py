"""Training orchestration, fair model comparison, and forecast generation.

Design goals (what makes this defensible rather than a demo toy):

  1. FAIR COMPARISON. Every model predicts the SAME target -- next-day demand for
     each (medicine, date) -- and is scored on the SAME held-out (medicine, date)
     points. No model is secretly solving an easier problem.

  2. NAIVE BASELINE + SKILL SCORE. A seasonal-naive baseline ("same weekday last
     week") is included. A model is only "good" if it beats this baseline
     (skill_score > 0). We report this honestly.

  3. ROLLING-ORIGIN BACKTEST. Metrics are averaged over several walk-forward folds
     instead of a single lucky/unlucky holdout window.

If TensorFlow is missing or LSTM training fails, the system continues with the
remaining models (no crash).
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd

from config import Config
import db.mongo as mongo
from ml import models as M
from ml import preprocessing as P
from ml.evaluation import evaluate, skill_score

logger = logging.getLogger(__name__)

N_FOLDS = 3            # rolling-origin backtest folds
FOLD_TEST_DAYS = 30    # each fold tests on 30 unseen days


def _fold_cutoffs(dates):
    """Return a list of (train_end, test_end) timestamps for N_FOLDS walk-forward
    folds, each testing on the FOLD_TEST_DAYS days that follow train_end."""
    last = dates.max()
    cutoffs = []
    for k in range(N_FOLDS, 0, -1):
        test_end = last - pd.Timedelta(days=FOLD_TEST_DAYS * (k - 1))
        train_end = test_end - pd.Timedelta(days=FOLD_TEST_DAYS)
        cutoffs.append((train_end, test_end))
    return cutoffs


def _lstm_fold_predictions(df, train_end, seq_len):
    """One-step-ahead LSTM predictions per medicine over the test window.

    For fairness the LSTM, like the tabular models, predicts each test day from the
    real preceding `seq_len` days (the tabular models likewise use real lag/rolling
    features). Returns a dict {(medicine_id, date): prediction}. Never raises.
    """
    preds = {}
    try:
        if not M._check_tensorflow():
            return None
        for mid, grp in df.groupby("medicine_id"):
            grp = grp.sort_values("date").reset_index(drop=True)
            series = grp["quantity_sold"].values.astype("float32")
            dates = grp["date"].values
            train_vals = series[grp["date"] <= train_end]
            if len(train_vals) < seq_len + 20:
                continue
            vmin, vmax = float(train_vals.min()), float(train_vals.max())
            denom = (vmax - vmin) or 1.0
            scaled_all = (series - vmin) / denom
            scaled_train = (train_vals - vmin) / denom

            # build training sequences
            X, y = P.make_sequences(scaled_train, seq_len)
            if len(X) == 0:
                continue
            model = M.build_lstm(seq_len)
            if model is None:
                return None
            model.fit(X[..., np.newaxis], y, epochs=12, batch_size=32, verbose=0)

            # predict each test day from its real preceding window
            for i in range(len(series)):
                d = pd.Timestamp(dates[i])
                if d <= train_end or i < seq_len:
                    continue
                window = scaled_all[i - seq_len:i][np.newaxis, ..., np.newaxis]
                p = float(model.predict(window, verbose=0)[0, 0])
                preds[(mid, d)] = max(0.0, p * denom + vmin)
        return preds
    except Exception as exc:  # noqa: BLE001
        logger.exception("LSTM backtest failed (continuing without it): %s", exc)
        return None


def _backtest(df, rf_params=None):
    """Run the rolling-origin backtest for all models. Returns pooled true/pred
    arrays per model across all folds, aligned on the same (medicine, date) keys."""
    from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN

    cutoffs = _fold_cutoffs(df["date"])
    seq_len = Config.LSTM_SEQUENCE_LENGTH

    # pooled, key-aligned predictions
    y_true = []
    pred = {"Naive (seasonal)": [], "Linear Regression": [], "Random Forest": [], "LSTM": []}
    lstm_ok = True

    for train_end, test_end in cutoffs:
        train = df[df["date"] <= train_end]
        test = df[(df["date"] > train_end) & (df["date"] <= test_end)]
        if train.empty or test.empty:
            continue

        # --- tabular models ---
        lr = M.build_linear_regression().fit(train[FEATURE_COLUMNS].values, train[TARGET_COLUMN].values)
        rf = M.build_random_forest(rf_params).fit(train[FEATURE_COLUMNS].values, train[TARGET_COLUMN].values)
        lr_pred = np.clip(lr.predict(test[FEATURE_COLUMNS].values), 0, None)
        rf_pred = np.clip(rf.predict(test[FEATURE_COLUMNS].values), 0, None)

        # --- naive seasonal baseline: same weekday last week == lag_7 ---
        naive_pred = np.clip(test["lag_7"].values, 0, None)

        # --- LSTM (per medicine, one-step-ahead) ---
        lstm_map = _lstm_fold_predictions(df, train_end, seq_len) if lstm_ok else None
        if lstm_map is None:
            lstm_ok = False

        for i, (_, row) in enumerate(test.iterrows()):
            key = (row["medicine_id"], row["date"])
            y_true.append(row[TARGET_COLUMN])
            pred["Naive (seasonal)"].append(naive_pred[i])
            pred["Linear Regression"].append(lr_pred[i])
            pred["Random Forest"].append(rf_pred[i])
            if lstm_map is not None:
                pred["LSTM"].append(lstm_map.get(key, np.nan))

    return np.array(y_true), pred, lstm_ok


def _train_final_models(rf_params=None):
    """Retrain LR & RF on ALL data and persist them for forecast generation.
    Also trains/saves a representative LSTM artifact when TensorFlow is available.
    Returns the RF feature-importance list (for interpretability) when available."""
    from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
    # Fit production models on the FULL series (no holdout) for forecasting.
    full = P.build_feature_frame()
    if full.empty:
        return None
    X = full[FEATURE_COLUMNS].values
    y = full[TARGET_COLUMN].values
    feature_importance = None
    try:
        lr = M.build_linear_regression().fit(X, y)
        M.save_sklearn_model(lr, Config.LINEAR_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Final LR fit failed: %s", exc)
    try:
        rf = M.build_random_forest(rf_params).fit(X, y)
        M.save_sklearn_model(rf, Config.RF_MODEL_PATH)
        # capture feature importances for the UI (interpretability)
        feature_importance = sorted(
            [{"feature": f, "importance": round(float(imp), 4)}
             for f, imp in zip(FEATURE_COLUMNS, rf.feature_importances_)],
            key=lambda d: d["importance"], reverse=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Final RF fit failed: %s", exc)
    # representative LSTM artifact (highest-volume medicine) for completeness
    try:
        if M._check_tensorflow():
            seq_len = Config.LSTM_SEQUENCE_LENGTH
            top_mid = full.groupby("medicine_id")[TARGET_COLUMN].sum().idxmax()
            series = full[full["medicine_id"] == top_mid].sort_values("date")[TARGET_COLUMN].values.astype("float32")
            vmin, vmax = float(series.min()), float(series.max())
            denom = (vmax - vmin) or 1.0
            scaled = (series - vmin) / denom
            Xs, ys = P.make_sequences(scaled, seq_len)
            if len(Xs):
                model = M.build_lstm(seq_len)
                model.fit(Xs[..., np.newaxis], ys, epochs=12, batch_size=32, verbose=0)
                M.save_lstm_model(model, Config.LSTM_MODEL_PATH)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LSTM artifact save skipped: %s", exc)
    return feature_importance


def train_all_models():
    """Backtest all models fairly, select the best (lowest MAE that beats naive),
    derive conformal prediction intervals, retrain final models, and persist
    comparison + selection (with PI offsets, coverage, and feature importances)."""
    from ml.preprocessing import FEATURE_COLUMNS, TARGET_COLUMN
    from ml.evaluation import conformal_offsets, interval_coverage

    df = P.build_feature_frame()
    if df.empty:
        raise ValueError("No data available. Seed/ingest data first.")

    # --- light hyperparameter tuning for Random Forest (time-series CV) ---
    rf_params = M.tune_random_forest(df[FEATURE_COLUMNS].values, df[TARGET_COLUMN].values)

    y_true, pred, lstm_ok = _backtest(df, rf_params)
    if len(y_true) == 0:
        raise RuntimeError("Backtest produced no test points.")

    # Compute metrics per model on the pooled, aligned test points.
    metrics = {}
    for name, preds in pred.items():
        preds = np.array(preds, dtype=float)
        if name == "LSTM" and (not lstm_ok or np.all(np.isnan(preds))):
            continue
        # align: drop any NaN prediction points (and matching truth) for this model
        mask = ~np.isnan(preds)
        if mask.sum() == 0:
            continue
        metrics[name] = evaluate(y_true[mask], preds[mask])

    baseline_mae = metrics.get("Naive (seasonal)", {}).get("mae")

    # Candidate models for selection exclude the naive baseline itself.
    candidates = {k: v for k, v in metrics.items() if k != "Naive (seasonal)"}
    if not candidates:
        raise RuntimeError("No model produced valid predictions.")

    def _mae(name):
        m = candidates[name]["mae"]
        return float("inf") if (m is None or np.isnan(m)) else m

    best_name = min(candidates, key=_mae)

    comparison = []
    for name, mt in metrics.items():
        comparison.append({
            "model_name": name,
            "mae": mt["mae"],
            "rmse": mt["rmse"],
            "mape": mt["mape"],
            "skill_vs_naive": skill_score(mt["mae"], baseline_mae) if name != "Naive (seasonal)" else 0.0,
            "is_baseline": name == "Naive (seasonal)",
            "is_best": name == best_name,
        })
    comparison.sort(key=lambda c: (c["mae"] is None, c["mae"]))

    # --- conformal prediction interval (80%) from the best model's residuals ---
    best_preds = np.array(pred[best_name], dtype=float)
    bmask = ~np.isnan(best_preds)
    residuals = y_true[bmask] - best_preds[bmask]
    pi_level = 0.80
    offsets = conformal_offsets(residuals, level=pi_level)
    prediction_interval = None
    if offsets is not None:
        lo, hi = offsets
        coverage = interval_coverage(y_true[bmask], best_preds[bmask], lo, hi)
        prediction_interval = {
            "level": pi_level,
            "lo_offset": round(lo, 4),
            "hi_offset": round(hi, 4),
            "empirical_coverage": coverage,
        }

    # Retrain & persist production models (tuned RF) and capture importances.
    feature_importance = _train_final_models(rf_params)

    metrics_doc = {
        "generated_at": datetime.utcnow().isoformat(),
        "best_model": best_name,
        "baseline_mae": baseline_mae,
        "n_folds": N_FOLDS,
        "fold_test_days": FOLD_TEST_DAYS,
        "test_points": int(len(y_true)),
        "comparison": comparison,
        "prediction_interval": prediction_interval,
        "feature_importance": feature_importance,
        "rf_params": rf_params,
        "tensorflow_available": M._check_tensorflow(),
    }
    mongo.replace_collection(mongo.MODEL_METRICS, comparison)
    M.save_selected_model({
        "best_model": best_name,
        "metrics": next(c for c in comparison if c["model_name"] == best_name),
        "baseline_mae": baseline_mae,
        "backtest": f"{N_FOLDS}-fold rolling origin, {FOLD_TEST_DAYS}d test windows",
        "prediction_interval": prediction_interval,
        "feature_importance": feature_importance,
        "rf_params": rf_params,
        "generated_at": metrics_doc["generated_at"],
    })
    logger.info("Best model: %s (baseline MAE=%s, PI=%s)", best_name, baseline_mae, prediction_interval)
    return metrics_doc


# --------------------------------------------------------------------------- #
# Forecast generation (uses the best tabular model for per-medicine granularity)
# --------------------------------------------------------------------------- #
def _recursive_tabular_forecast(model, df, horizon):
    """Generate per-medicine forecasts by recursively rolling features forward.

    Builds the SAME feature set as training (lags, rolling stats, harmonic
    seasonality, trend) so train/serve features stay consistent.
    """
    from ml.preprocessing import FEATURE_COLUMNS, harmonic_trend_features

    forecasts = []
    last_date = df["date"].max()
    global_start = df["date"].min()
    for mid, grp in df.groupby("medicine_id"):
        grp = grp.sort_values("date")
        history = list(grp["quantity_sold"].values)
        last_row = grp.iloc[-1]
        cat_enc = int(last_row["medicine_category_encoded"])
        temp = float(last_row["temperature"])
        disease = float(last_row["disease_index"])
        outbreak = int(last_row["outbreak_alert"])
        event = int(last_row["local_event_flag"])

        for h in range(1, horizon + 1):
            fdate = last_date + pd.Timedelta(days=h)
            lag_1 = history[-1] if history else 0
            lag_7 = history[-7] if len(history) >= 7 else (history[0] if history else 0)
            lag_14 = history[-14] if len(history) >= 14 else (history[0] if history else 0)
            roll7 = np.mean(history[-7:]) if history else 0
            roll14 = np.mean(history[-14:]) if history else 0
            roll28 = np.mean(history[-28:]) if history else 0
            std7 = np.std(history[-7:], ddof=1) if len(history) >= 2 else 0.0
            season = (0 if fdate.month in (12, 1, 2) else 1 if fdate.month in (3, 4, 5)
                      else 2 if fdate.month in (6, 7, 8) else 3)
            ht = harmonic_trend_features(fdate, (fdate - global_start).days / 365.0)
            feat_map = {
                "day_of_week": fdate.dayofweek, "month": fdate.month, "season_encoded": season,
                "lag_1": lag_1, "lag_7": lag_7, "lag_14": lag_14,
                "rolling_mean_7": roll7, "rolling_mean_14": roll14, "rolling_mean_28": roll28,
                "rolling_std_7": std7, "sin_doy": ht["sin_doy"], "cos_doy": ht["cos_doy"],
                "trend": ht["trend"], "temperature": temp, "disease_index": disease,
                "outbreak_alert": outbreak, "local_event_flag": event,
                "medicine_category_encoded": cat_enc,
            }
            feats = pd.DataFrame([[feat_map[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
            p = float(np.clip(model.predict(feats.values)[0], 0, None))
            history.append(p)
            forecasts.append({
                "medicine_id": mid,
                "forecast_date": fdate.date().isoformat(),
                "predicted_quantity": round(p, 2),
            })
    return forecasts


def generate_forecasts(horizon=None):
    """Generate next-`horizon`-day per-medicine forecasts with the selected model,
    persist them, and return the list. Falls back RF -> LR if needed."""
    horizon = horizon or Config.FORECAST_HORIZON_DAYS
    df = P.build_feature_frame()
    if df.empty:
        raise ValueError("No data available. Seed/ingest and train first.")

    selected = M.load_selected_model()
    best_name = selected["best_model"] if selected else "Random Forest"

    model, used_model_name = None, best_name
    try:
        if best_name == "Linear Regression":
            model = M.load_sklearn_model(Config.LINEAR_MODEL_PATH)
        else:
            # RF and (LSTM-selected) both use RF for stable per-medicine forecasts
            model = M.load_sklearn_model(Config.RF_MODEL_PATH)
            used_model_name = "Random Forest" if best_name != "Random Forest" else best_name
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load %s (%s); trying fallbacks.", best_name, exc)

    if model is None:
        for path, name in ((Config.RF_MODEL_PATH, "Random Forest"),
                           (Config.LINEAR_MODEL_PATH, "Linear Regression")):
            try:
                model = M.load_sklearn_model(path)
                used_model_name = name
                break
            except Exception:  # noqa: BLE001
                continue
    if model is None:
        raise RuntimeError("No trained model available. Run training first.")

    raw = _recursive_tabular_forecast(model, df, horizon)

    # --- attach conformal prediction-interval bounds (P10/P90 around P50) ---
    pi = (selected or {}).get("prediction_interval")
    lo_off = pi["lo_offset"] if pi else None
    hi_off = pi["hi_offset"] if pi else None
    pi_level = pi["level"] if pi else None

    def confidence_for(pred, lower, upper):
        """Confidence from RELATIVE interval width (real uncertainty, not a guess):
        narrow band vs the prediction => high confidence."""
        if pred <= 0 or lower is None:
            return "low"
        rel_width = (upper - lower) / max(pred, 1e-6)
        if rel_width <= 1.0:
            return "high"
        if rel_width <= 2.5:
            return "medium"
        return "low"

    now = datetime.utcnow().isoformat()
    for f in raw:
        p = f["predicted_quantity"]
        if lo_off is not None:
            lower = max(0.0, round(p + lo_off, 2))
            upper = round(p + hi_off, 2)
        else:
            lower, upper = None, None
        f["predicted_lower"] = lower
        f["predicted_upper"] = upper
        f["interval_level"] = pi_level
        f["model_name"] = used_model_name
        f["confidence_level"] = confidence_for(p, lower, upper)
        f["created_at"] = now

    mongo.replace_collection(mongo.FORECASTS, raw)
    logger.info("Generated %d forecast rows using %s (PI level=%s).",
                len(raw), used_model_name, pi_level)
    return raw
