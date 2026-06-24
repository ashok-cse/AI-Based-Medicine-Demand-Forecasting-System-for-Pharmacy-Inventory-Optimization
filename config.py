"""Central configuration loaded from environment variables.

This is a university software-engineering prototype. No patient-level data is
used anywhere in the system. All data is synthetic pharmacy sales/inventory.
"""
import os
from dotenv import load_dotenv

# Load .env if present (local development). In production (Docker/Easypanel),
# environment variables are injected by the platform.
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    # --- Flask ---
    SECRET_KEY = os.getenv("SECRET_KEY", "change-this-secret")
    FLASK_ENV = os.getenv("FLASK_ENV", "production")
    DEBUG = FLASK_ENV != "production"
    PORT = int(os.getenv("PORT", "5000"))

    # --- MongoDB ---
    # MONGO_URI is the single source of truth for the database connection.
    MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/medicine_forecasting")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "medicine_forecasting")

    # --- Paths ---
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")

    # CSV fallback files (used when MongoDB is unavailable)
    SALES_CSV = os.path.join(DATA_DIR, "synthetic_sales.csv")
    INVENTORY_CSV = os.path.join(DATA_DIR, "synthetic_inventory.csv")
    EXTERNAL_CSV = os.path.join(DATA_DIR, "synthetic_external_factors.csv")
    MEDICINES_CSV = os.path.join(DATA_DIR, "synthetic_medicines.csv")

    # Model artifact paths
    LINEAR_MODEL_PATH = os.path.join(MODELS_DIR, "linear_regression.pkl")
    RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest.pkl")
    LSTM_MODEL_PATH = os.path.join(MODELS_DIR, "lstm_model.keras")
    SELECTED_MODEL_PATH = os.path.join(MODELS_DIR, "selected_model.json")

    # --- ML / Forecasting defaults ---
    FORECAST_HORIZON_DAYS = 30
    LSTM_SEQUENCE_LENGTH = 14
    TEST_SIZE_DAYS = 30  # last N days reserved as the common test period

    # Minimum sales history (days with sales) before we trust a forecast
    MIN_HISTORY_FOR_CONFIDENCE = 30


# Ensure data/model directories exist at import time.
os.makedirs(Config.DATA_DIR, exist_ok=True)
os.makedirs(Config.MODELS_DIR, exist_ok=True)
