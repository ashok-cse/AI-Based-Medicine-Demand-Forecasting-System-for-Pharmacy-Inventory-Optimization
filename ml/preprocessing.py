"""Data preprocessing & feature engineering for demand forecasting.

Loads sales / inventory / external-factor data (Mongo or CSV fallback),
aggregates daily sales per medicine, engineers time-series features, and
prepares both tabular (for Linear Regression / Random Forest) and sequence
(for LSTM) datasets.
"""
import logging

import numpy as np
import pandas as pd

from config import Config
import db.mongo as mongo

logger = logging.getLogger(__name__)

SEASON_MAP = {"Winter": 0, "Spring": 1, "Summer": 2, "Autumn": 3}

# Tabular feature columns used by Linear Regression & Random Forest.
# Includes lag/rolling history, harmonic (Fourier) seasonality, a trend index,
# and external/category context. Harmonic terms let linear models capture smooth
# annual seasonality without one-hot month explosion.
FEATURE_COLUMNS = [
    "day_of_week",
    "month",
    "season_encoded",
    "lag_1",
    "lag_7",
    "lag_14",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
    "sin_doy",          # annual harmonic (sin of day-of-year)
    "cos_doy",          # annual harmonic (cos of day-of-year)
    "trend",            # normalized linear trend index
    "temperature",
    "disease_index",
    "outbreak_alert",
    "local_event_flag",
    "medicine_category_encoded",
]
TARGET_COLUMN = "quantity_sold"


def harmonic_trend_features(date, trend_index):
    """Compute the harmonic seasonality + trend features for a single date.

    Centralized so both the training frame and the recursive forecaster build
    identical features. `trend_index` is days since the series start / 365.
    """
    doy = date.timetuple().tm_yday
    return {
        "sin_doy": float(np.sin(2 * np.pi * doy / 365.25)),
        "cos_doy": float(np.cos(2 * np.pi * doy / 365.25)),
        "trend": float(trend_index),
    }


def load_raw_frames():
    """Load the three core frames. Returns (sales_df, external_df, medicines_df)."""
    sales = mongo.load_dataframe(mongo.SALES)
    external = mongo.load_dataframe(mongo.EXTERNAL_FACTORS)
    medicines = mongo.load_dataframe(mongo.MEDICINES)
    return sales, external, medicines


def build_feature_frame():
    """Build the full engineered feature frame across all medicines.

    Returns a DataFrame sorted by (medicine_id, date) with engineered columns,
    or an empty DataFrame if there is no data.
    """
    sales, external, medicines = load_raw_frames()
    if sales.empty or medicines.empty:
        logger.warning("No sales/medicine data available for preprocessing.")
        return pd.DataFrame()

    sales = sales.copy()
    sales["sale_date"] = pd.to_datetime(sales["sale_date"])

    # Aggregate daily sales per medicine (collapse multiple rows per day).
    daily = (
        sales.groupby(["medicine_id", "sale_date"], as_index=False)["quantity_sold"]
        .sum()
        .rename(columns={"sale_date": "date"})
    )

    # Build a continuous daily index per medicine so lags/rolling are correct.
    frames = []
    for mid, grp in daily.groupby("medicine_id"):
        grp = grp.sort_values("date")
        full_idx = pd.date_range(grp["date"].min(), grp["date"].max(), freq="D")
        grp = grp.set_index("date").reindex(full_idx)
        grp["medicine_id"] = mid
        grp["quantity_sold"] = grp["quantity_sold"].fillna(0)
        grp = grp.rename_axis("date").reset_index()
        frames.append(grp)
    daily = pd.concat(frames, ignore_index=True)

    # --- merge external factors (by date) ---
    if not external.empty:
        ext = external.copy()
        ext["date"] = pd.to_datetime(ext["date"])
        ext["season_encoded"] = ext["season"].map(SEASON_MAP).fillna(0).astype(int)
        ext_cols = ["date", "temperature", "disease_index", "outbreak_alert",
                    "local_event_flag", "season_encoded"]
        daily = daily.merge(ext[ext_cols], on="date", how="left")
    else:
        # Sensible defaults if external factors are missing.
        daily["temperature"] = 20.0
        daily["disease_index"] = 0.3
        daily["outbreak_alert"] = 0
        daily["local_event_flag"] = 0
        daily["season_encoded"] = daily["date"].dt.month.map(
            lambda m: 0 if m in (12, 1, 2) else 1 if m in (3, 4, 5) else 2 if m in (6, 7, 8) else 3
        )

    # --- merge medicine category encoding ---
    med = medicines.copy()
    cat_codes = {c: i for i, c in enumerate(sorted(med["category"].unique()))}
    med["medicine_category_encoded"] = med["category"].map(cat_codes)
    daily = daily.merge(
        med[["medicine_id", "medicine_category_encoded", "category"]],
        on="medicine_id", how="left",
    )

    # --- calendar features ---
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month

    # --- harmonic (Fourier) annual seasonality + global trend ---
    doy = daily["date"].dt.dayofyear
    daily["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
    daily["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)
    start = daily["date"].min()
    daily["trend"] = (daily["date"] - start).dt.days / 365.0

    # --- lag & rolling features (per medicine; shift(1) prevents leakage) ---
    daily = daily.sort_values(["medicine_id", "date"])
    g = daily.groupby("medicine_id")["quantity_sold"]
    daily["lag_1"] = g.shift(1)
    daily["lag_7"] = g.shift(7)
    daily["lag_14"] = g.shift(14)
    prev = g.shift(1)
    daily["rolling_mean_7"] = prev.rolling(7, min_periods=1).mean()
    daily["rolling_mean_14"] = prev.rolling(14, min_periods=1).mean()
    daily["rolling_mean_28"] = prev.rolling(28, min_periods=1).mean()
    daily["rolling_std_7"] = prev.rolling(7, min_periods=1).std()

    # --- handle missing values ---
    for col in ["lag_1", "lag_7", "lag_14", "rolling_mean_7", "rolling_mean_14",
                "rolling_mean_28", "rolling_std_7"]:
        daily[col] = daily[col].fillna(0)
    for col in ["temperature", "disease_index"]:
        daily[col] = daily[col].fillna(daily[col].median() if daily[col].notna().any() else 0)
    for col in ["outbreak_alert", "local_event_flag", "season_encoded",
                "medicine_category_encoded"]:
        daily[col] = daily[col].fillna(0).astype(int)

    return daily


def train_test_split_by_date(df, test_days=None):
    """Split chronologically: last `test_days` days reserved for testing.

    Using a common cutoff date keeps the test period comparable across models.
    """
    test_days = test_days or Config.TEST_SIZE_DAYS
    if df.empty:
        return df, df
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train = df[df["date"] <= cutoff]
    test = df[df["date"] > cutoff]
    return train, test


def get_tabular_dataset(test_days=None):
    """Return (X_train, y_train, X_test, y_test, full_df) for tabular models."""
    df = build_feature_frame()
    if df.empty:
        return None
    train, test = train_test_split_by_date(df, test_days)
    X_train = train[FEATURE_COLUMNS].values
    y_train = train[TARGET_COLUMN].values
    X_test = test[FEATURE_COLUMNS].values
    y_test = test[TARGET_COLUMN].values
    return X_train, y_train, X_test, y_test, df


def make_sequences(series, seq_len):
    """Turn a 1-D array into (X, y) sliding windows for LSTM."""
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    if not X:
        return np.empty((0, seq_len)), np.empty((0,))
    return np.array(X), np.array(y)


def get_lstm_dataset(df=None, seq_len=None, test_days=None):
    """Prepare LSTM sequence data aggregated to total daily demand.

    To keep the LSTM lightweight on a small VPS, we model the *aggregate* daily
    demand (sum across medicines) as a univariate series. Returns a dict with
    train/test sequences and the scaler params, or None if insufficient data.
    """
    seq_len = seq_len or Config.LSTM_SEQUENCE_LENGTH
    test_days = test_days or Config.TEST_SIZE_DAYS
    if df is None:
        df = build_feature_frame()
    if df is None or df.empty:
        return None

    # Aggregate to total daily demand.
    agg = df.groupby("date")["quantity_sold"].sum().sort_index()
    if len(agg) < seq_len + test_days + 5:
        return None

    values = agg.values.astype("float32")
    # Min-max scale for stable LSTM training.
    vmin, vmax = float(values.min()), float(values.max())
    denom = (vmax - vmin) or 1.0
    scaled = (values - vmin) / denom

    split = len(scaled) - test_days
    train_series = scaled[:split]
    # include the tail of train as warm-up context for the test windows
    test_series = scaled[split - seq_len:]

    X_train, y_train = make_sequences(train_series, seq_len)
    X_test, y_test = make_sequences(test_series, seq_len)
    if len(X_train) == 0 or len(X_test) == 0:
        return None

    return {
        "X_train": X_train[..., np.newaxis],
        "y_train": y_train,
        "X_test": X_test[..., np.newaxis],
        "y_test": y_test,
        "scaler": {"min": vmin, "max": vmax},
        "seq_len": seq_len,
        "full_scaled": scaled,
        "dates": agg.index,
    }
