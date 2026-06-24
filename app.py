"""Flask application: pages + JSON API for the medicine demand-forecasting demo.

This is a university software-engineering prototype. It does NOT provide medical
advice and uses only synthetic, non-personal data.
"""
import io
import logging

import pandas as pd
from flask import Flask, render_template, jsonify, request

from config import Config
import db.mongo as mongo
from seed_data import seed
from ingest_real import ingest as ingest_real
from ml.forecasting import train_all_models, generate_forecasts
from ml.models import load_selected_model
from inventory.optimizer import optimize_inventory
from inventory.alerts import generate_alerts

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config.from_object(Config)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def load_data(prefer_real=True):
    """Load data into the system. Prefers the REAL Kaggle pharma-sales dataset;
    falls back to the synthetic generator if the real CSV isn't present."""
    if prefer_real:
        try:
            return ingest_real()
        except FileNotFoundError as exc:
            logger.warning("Real dataset unavailable (%s) — using synthetic data.", exc)
    return seed()


def _ensure_data():
    """Auto-load if the database is empty so the demo is never blank."""
    try:
        if mongo.database_is_empty():
            logger.info("Database empty — auto-loading data.")
            load_data()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Auto-load skipped: %s", exc)


def _ok(payload=None, **extra):
    body = {"status": "ok"}
    if payload is not None:
        body["data"] = payload
    body.update(extra)
    return jsonify(body)


def _err(message, code=400):
    return jsonify({"status": "error", "message": str(message)}), code


# --------------------------------------------------------------------------- #
# Page routes
# --------------------------------------------------------------------------- #
@app.route("/")
def dashboard():
    return render_template("dashboard.html", active="dashboard")


@app.route("/medicines")
def medicines_page():
    return render_template("medicines.html", active="medicines")


@app.route("/forecasts")
def forecasts_page():
    return render_template("forecasts.html", active="forecasts")


@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html", active="alerts")


@app.route("/model-comparison")
def model_comparison_page():
    return render_template("model_comparison.html", active="model-comparison")


@app.route("/upload")
def upload_page():
    return render_template("upload.html", active="upload")


# --------------------------------------------------------------------------- #
# API: health & data
# --------------------------------------------------------------------------- #
@app.route("/api/health")
def api_health():
    return _ok(mongo_available=mongo.is_mongo_available(),
               db=Config.MONGO_DB_NAME)


@app.route("/api/dashboard-summary")
def api_dashboard_summary():
    _ensure_data()
    medicines = mongo.load_collection(mongo.MEDICINES)
    optimized = optimize_inventory()
    alerts = mongo.load_collection(mongo.ALERTS)
    selected = load_selected_model()

    low_stock = sum(1 for o in optimized if o["current_stock"] <= o["reorder_point"])
    expiry_risk = sum(1 for o in optimized if o["expiry_risk"] in ("high", "expired"))
    total_rec_qty = sum(o["recommended_order_quantity"] for o in optimized)

    # category-wise stock
    cat_stock = {}
    for o in optimized:
        cat_stock[o["category"]] = cat_stock.get(o["category"], 0) + o["current_stock"]

    # alert severity distribution
    sev_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
    for a in alerts:
        sev_counts[a.get("severity", "low")] = sev_counts.get(a.get("severity", "low"), 0) + 1

    best_model = selected["best_model"] if selected else None
    best_mae = selected["metrics"]["mae"] if selected else None

    return _ok({
        "total_medicines": len(medicines),
        "low_stock_alerts": low_stock,
        "expiry_risk_medicines": expiry_risk,
        "total_recommended_order_quantity": int(total_rec_qty),
        "best_model": best_model,
        "best_model_mae": best_mae,
        "category_stock": cat_stock,
        "alert_severity": sev_counts,
        "total_alerts": len(alerts),
    })


@app.route("/api/medicines")
def api_medicines():
    _ensure_data()
    # Join medicine master data with optimized inventory metrics.
    optimized = {o["medicine_id"]: o for o in optimize_inventory()}
    medicines = mongo.load_collection(mongo.MEDICINES)
    inventory = {i["medicine_id"]: i for i in mongo.load_collection(mongo.INVENTORY)}
    rows = []
    for m in medicines:
        mid = m["medicine_id"]
        inv = inventory.get(mid, {})
        opt = optimized.get(mid, {})
        rows.append({**m,
                     "current_stock": inv.get("current_stock"),
                     "expiry_date": inv.get("expiry_date"),
                     "reorder_threshold": inv.get("reorder_threshold"),
                     "lead_time_days": inv.get("lead_time_days"),
                     "reorder_point": opt.get("reorder_point"),
                     "recommended_order_quantity": opt.get("recommended_order_quantity"),
                     "days_until_stockout": opt.get("days_until_stockout"),
                     "expiry_risk": opt.get("expiry_risk")})
    return _ok(rows)


@app.route("/api/forecasts")
def api_forecasts():
    """Return forecasts. Optional ?medicine_id=M001 filters to one medicine."""
    mid = request.args.get("medicine_id")
    query = {"medicine_id": mid} if mid else None
    forecasts = mongo.load_collection(mongo.FORECASTS, query)
    # also return inventory recommendations for the forecast table
    optimized = optimize_inventory()
    return _ok({"forecasts": forecasts, "optimization": optimized})


@app.route("/api/alerts")
def api_alerts():
    alerts = mongo.load_collection(mongo.ALERTS)
    # newest first
    alerts.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    return _ok(alerts)


@app.route("/api/model-comparison")
def api_model_comparison():
    comparison = mongo.load_collection(mongo.MODEL_METRICS)
    selected = load_selected_model()
    return _ok({"comparison": comparison,
                "selected": selected})


# --------------------------------------------------------------------------- #
# API: actions
# --------------------------------------------------------------------------- #
@app.route("/api/seed", methods=["POST"])
def api_seed():
    """Load data. Defaults to the REAL dataset; pass {"synthetic": true} to force
    the synthetic generator instead."""
    try:
        force_synthetic = bool((request.get_json(silent=True) or {}).get("synthetic"))
        result = load_data(prefer_real=not force_synthetic)
        src = result.get("source", "synthetic")
        return _ok(result, message=f"Data loaded ({src}).")
    except Exception as exc:  # noqa: BLE001
        logger.exception("Data load failed")
        return _err(f"Data load failed: {exc}", 500)


@app.route("/api/train", methods=["POST"])
def api_train():
    try:
        result = train_all_models()
        return _ok(result, message=f"Training complete. Best model: {result['best_model']}.")
    except ValueError as exc:
        return _err(str(exc), 400)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Training failed")
        return _err(f"Training failed: {exc}", 500)


@app.route("/api/generate-forecasts", methods=["POST"])
def api_generate_forecasts():
    try:
        forecasts = generate_forecasts()
        optimized = optimize_inventory()
        alerts = generate_alerts(optimized)
        return _ok({
            "forecast_rows": len(forecasts),
            "optimized_medicines": len(optimized),
            "alerts": len(alerts),
        }, message="Forecasts, inventory recommendations, and alerts updated.")
    except ValueError as exc:
        return _err(str(exc), 400)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Forecast generation failed")
        return _err(f"Forecast generation failed: {exc}", 500)


# --------------------------------------------------------------------------- #
# API: CSV uploads
# --------------------------------------------------------------------------- #
def _read_uploaded_csv():
    if "file" not in request.files:
        raise ValueError("No file uploaded (expected form field 'file').")
    file = request.files["file"]
    if not file.filename:
        raise ValueError("Empty filename.")
    content = file.read()
    return pd.read_csv(io.BytesIO(content))


def _handle_upload(collection, required_cols):
    df = _read_uploaded_csv()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    records = df.to_dict(orient="records")
    mongo.replace_collection(collection, records)
    return len(records)


@app.route("/api/upload/sales", methods=["POST"])
def api_upload_sales():
    try:
        n = _handle_upload(mongo.SALES,
                           ["medicine_id", "sale_date", "quantity_sold"])
        return _ok({"rows": n}, message=f"Imported {n} sales rows.")
    except Exception as exc:  # noqa: BLE001
        return _err(f"Sales upload failed: {exc}", 400)


@app.route("/api/upload/inventory", methods=["POST"])
def api_upload_inventory():
    try:
        n = _handle_upload(mongo.INVENTORY,
                           ["medicine_id", "current_stock", "expiry_date"])
        return _ok({"rows": n}, message=f"Imported {n} inventory rows.")
    except Exception as exc:  # noqa: BLE001
        return _err(f"Inventory upload failed: {exc}", 400)


@app.route("/api/upload/external-factors", methods=["POST"])
def api_upload_external():
    try:
        n = _handle_upload(mongo.EXTERNAL_FACTORS, ["date"])
        return _ok({"rows": n}, message=f"Imported {n} external-factor rows.")
    except Exception as exc:  # noqa: BLE001
        return _err(f"External factors upload failed: {exc}", 400)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG)
