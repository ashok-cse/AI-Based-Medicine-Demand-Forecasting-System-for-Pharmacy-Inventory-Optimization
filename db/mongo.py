"""MongoDB access layer with a CSV fallback.

The system prefers MongoDB (connected via MONGO_URI). If MongoDB is unreachable
the data-loading helpers fall back to reading the synthetic CSV files in /data so
the prototype still works for a demo on a machine without a database.

Collections:
    medicines, sales, inventory, external_factors, forecasts, alerts, model_metrics
"""
import os
import logging

import pandas as pd

from config import Config

logger = logging.getLogger(__name__)

# Collection name constants
MEDICINES = "medicines"
SALES = "sales"
INVENTORY = "inventory"
EXTERNAL_FACTORS = "external_factors"
FORECASTS = "forecasts"
ALERTS = "alerts"
MODEL_METRICS = "model_metrics"

# Cache the client/db across calls within a process.
_client = None
_db = None
_connection_failed = False


def get_db():
    """Return a MongoDB database handle, or None if MongoDB is unavailable.

    A short server-selection timeout is used so the app degrades gracefully to
    the CSV fallback instead of hanging when no database is reachable.
    """
    global _client, _db, _connection_failed
    if _db is not None:
        return _db
    if _connection_failed:
        return None
    try:
        from pymongo import MongoClient

        _client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=2000)
        # Force a round-trip so we know quickly whether Mongo is reachable.
        _client.admin.command("ping")
        _db = _client[Config.MONGO_DB_NAME]
        logger.info("Connected to MongoDB database '%s'", Config.MONGO_DB_NAME)
        return _db
    except Exception as exc:  # noqa: BLE001 - we deliberately catch everything
        logger.warning("MongoDB unavailable (%s). Falling back to CSV files.", exc)
        _connection_failed = True
        return None


def is_mongo_available():
    return get_db() is not None


# --------------------------------------------------------------------------- #
# Write helpers
# --------------------------------------------------------------------------- #
def replace_collection(name, records):
    """Replace an entire collection's contents with `records` (list of dicts).

    Returns True if written to Mongo, False if it was written to CSV fallback.
    """
    db = get_db()
    if db is not None:
        coll = db[name]
        coll.delete_many({})
        if records:
            coll.insert_many([dict(r) for r in records])
        return True
    # CSV fallback
    _write_csv_fallback(name, records)
    return False


def insert_records(name, records):
    """Append records to a collection (or CSV fallback)."""
    if not records:
        return
    db = get_db()
    if db is not None:
        db[name].insert_many([dict(r) for r in records])
        return
    # CSV fallback: append by re-reading + concatenating
    existing = load_collection(name)
    combined = existing + [dict(r) for r in records]
    _write_csv_fallback(name, combined)


def _csv_path_for(name):
    mapping = {
        MEDICINES: Config.MEDICINES_CSV,
        SALES: Config.SALES_CSV,
        INVENTORY: Config.INVENTORY_CSV,
        EXTERNAL_FACTORS: Config.EXTERNAL_CSV,
    }
    # Generated collections (forecasts/alerts/model_metrics) get their own files.
    return mapping.get(name, os.path.join(Config.DATA_DIR, f"_{name}.csv"))


def _write_csv_fallback(name, records):
    path = _csv_path_for(name)
    df = pd.DataFrame(list(records)) if records else pd.DataFrame()
    # Drop Mongo's _id if it leaked in.
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    df.to_csv(path, index=False)


# --------------------------------------------------------------------------- #
# Read helpers
# --------------------------------------------------------------------------- #
def load_collection(name, query=None):
    """Load a collection as a list of dicts from Mongo or CSV fallback."""
    db = get_db()
    if db is not None:
        cursor = db[name].find(query or {}, {"_id": 0})
        return list(cursor)
    # CSV fallback
    path = _csv_path_for(name)
    if not os.path.exists(path):
        return []
    df = pd.read_csv(path)
    return df.to_dict(orient="records")


def load_dataframe(name, query=None):
    """Load a collection directly into a pandas DataFrame."""
    return pd.DataFrame(load_collection(name, query))


def collection_count(name):
    db = get_db()
    if db is not None:
        return db[name].count_documents({})
    path = _csv_path_for(name)
    if not os.path.exists(path):
        return 0
    try:
        return len(pd.read_csv(path))
    except Exception:  # noqa: BLE001
        return 0


def database_is_empty():
    """True when there are no medicines/sales loaded yet."""
    return collection_count(MEDICINES) == 0 or collection_count(SALES) == 0
