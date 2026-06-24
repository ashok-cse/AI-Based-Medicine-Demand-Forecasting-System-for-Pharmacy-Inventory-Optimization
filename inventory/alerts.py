"""Alert generation from optimized inventory metrics.

Alert types:
  LOW_STOCK           current_stock <= reorder_point
  EXPIRY_RISK         expiry_date within 30 days (or already expired)
  RESTOCK_NEEDED      recommended_order_quantity > 0
  OVERSTOCK_WARNING   stock >> predicted demand AND expiry risk present
  MODEL_LOW_CONFIDENCE  medicine has too little sales history

Severity: critical | high | medium | low
"""
import uuid
import logging
from datetime import datetime

import pandas as pd

from config import Config
import db.mongo as mongo
from inventory.optimizer import optimize_inventory

logger = logging.getLogger(__name__)


def _history_days_by_medicine():
    """Number of days with positive sales per medicine (history depth)."""
    sales = mongo.load_dataframe(mongo.SALES)
    if sales.empty:
        return {}
    pos = sales[sales["quantity_sold"] > 0]
    return pos.groupby("medicine_id")["sale_date"].nunique().to_dict()


def _make_alert(mid, alert_type, severity, message):
    return {
        "alert_id": str(uuid.uuid4())[:8],
        "medicine_id": mid,
        "alert_type": alert_type,
        "severity": severity,
        "message": message,
        "created_at": datetime.utcnow().isoformat(),
    }


def generate_alerts(optimized=None):
    """Build the alert list from optimized inventory and persist it.

    Returns the list of alert dicts.
    """
    if optimized is None:
        optimized = optimize_inventory()
    history = _history_days_by_medicine()
    alerts = []

    for row in optimized:
        mid = row["medicine_id"]
        name = row.get("medicine_name", mid)
        stock = row["current_stock"]
        reorder = row["reorder_point"]
        rec_qty = row["recommended_order_quantity"]
        forecast_30 = row["forecasted_demand"]
        expiry_risk = row["expiry_risk"]
        days_to_expiry = row.get("days_to_expiry")
        dus = row.get("days_until_stockout")

        # --- LOW_STOCK ---
        if stock <= reorder:
            if dus is not None and dus <= 3:
                sev = "critical"
            elif dus is not None and dus <= 7:
                sev = "high"
            else:
                sev = "medium"
            msg = (f"{name}: stock {stock} at/below reorder point "
                   f"{reorder:.0f}" + (f"; ~{dus} days to stockout." if dus is not None else "."))
            alerts.append(_make_alert(mid, "LOW_STOCK", sev, msg))

        # --- EXPIRY_RISK ---
        if expiry_risk in ("expired", "high"):
            if expiry_risk == "expired":
                sev, msg = "critical", f"{name}: batch already EXPIRED."
            else:
                sev = "high" if (days_to_expiry is not None and days_to_expiry <= 14) else "medium"
                msg = f"{name}: expires in {days_to_expiry} days."
            alerts.append(_make_alert(mid, "EXPIRY_RISK", sev, msg))

        # --- RESTOCK_NEEDED ---
        if rec_qty > 0:
            sev = "high" if stock <= reorder else "low"
            msg = f"{name}: recommended order quantity {rec_qty} units (next 30 days)."
            alerts.append(_make_alert(mid, "RESTOCK_NEEDED", sev, msg))

        # --- OVERSTOCK_WARNING (stock much higher than demand AND expiry risk) ---
        if forecast_30 > 0 and stock > forecast_30 * 2.5 and expiry_risk in ("high", "medium", "expired"):
            sev = "high" if expiry_risk in ("high", "expired") else "medium"
            msg = (f"{name}: overstocked ({stock} units vs ~{forecast_30:.0f} forecast) "
                   f"with expiry risk ({expiry_risk}).")
            alerts.append(_make_alert(mid, "OVERSTOCK_WARNING", sev, msg))

        # --- MODEL_LOW_CONFIDENCE ---
        if history.get(mid, 0) < Config.MIN_HISTORY_FOR_CONFIDENCE:
            msg = (f"{name}: limited sales history "
                   f"({history.get(mid, 0)} days) — forecast confidence is low.")
            alerts.append(_make_alert(mid, "MODEL_LOW_CONFIDENCE", "low", msg))

    mongo.replace_collection(mongo.ALERTS, alerts)
    logger.info("Generated %d alerts.", len(alerts))
    return alerts
