"""Inventory optimization.

For each medicine, combine the demand forecast with current inventory to compute
restocking metrics using practical (textbook) formulas:

    safety_stock              = demand_std * sqrt(lead_time_days)
    reorder_point             = forecasted_demand_during_lead_time + safety_stock
    recommended_order_qty     = max(0, forecasted_demand_next_30d + safety_stock - current_stock)
    days_until_stockout       = current_stock / average_daily_demand  (if avg > 0)
"""
import math
import logging
from datetime import datetime, date

import numpy as np
import pandas as pd

import db.mongo as mongo

logger = logging.getLogger(__name__)


def _parse_date(value):
    if isinstance(value, (datetime, date)):
        return value if isinstance(value, date) else value.date()
    try:
        return datetime.fromisoformat(str(value)[:10]).date()
    except Exception:  # noqa: BLE001
        return None


def optimize_inventory():
    """Compute per-medicine inventory metrics. Returns a list of dicts.

    Requires that forecasts have been generated; if not, demand falls back to a
    recent historical average so the page still renders.
    """
    medicines = mongo.load_dataframe(mongo.MEDICINES)
    inventory = mongo.load_dataframe(mongo.INVENTORY)
    forecasts = mongo.load_dataframe(mongo.FORECASTS)
    sales = mongo.load_dataframe(mongo.SALES)

    if medicines.empty or inventory.empty:
        return []

    # Aggregate forecast demand per medicine (next-30-day total + daily stats).
    fc_stats = {}
    if not forecasts.empty:
        for mid, grp in forecasts.groupby("medicine_id"):
            q = grp["predicted_quantity"].astype(float)
            fc_stats[mid] = {
                "total_30": float(q.sum()),
                "avg_daily": float(q.mean()),
                "std": float(q.std(ddof=0)) if len(q) > 1 else 0.0,
            }

    # Historical fallback (last 30 days average) per medicine.
    hist_avg = {}
    if not sales.empty:
        sales = sales.copy()
        sales["sale_date"] = pd.to_datetime(sales["sale_date"])
        cutoff = sales["sale_date"].max() - pd.Timedelta(days=30)
        recent = sales[sales["sale_date"] > cutoff]
        daily = recent.groupby(["medicine_id", recent["sale_date"].dt.date])["quantity_sold"].sum()
        for mid in medicines["medicine_id"]:
            vals = daily.loc[mid].values if mid in daily.index.get_level_values(0) else []
            hist_avg[mid] = {
                "avg": float(np.mean(vals)) if len(vals) else 0.0,
                "std": float(np.std(vals)) if len(vals) else 0.0,
            }

    inv_by_id = {row["medicine_id"]: row for _, row in inventory.iterrows()}
    med_by_id = {row["medicine_id"]: row for _, row in medicines.iterrows()}
    today = datetime.utcnow().date()

    results = []
    for mid, med in med_by_id.items():
        inv = inv_by_id.get(mid)
        if inv is None:
            continue
        current_stock = float(inv.get("current_stock", 0) or 0)
        lead_time = float(inv.get("lead_time_days", 7) or 7)

        # Demand inputs: prefer forecast, fallback to history.
        fc = fc_stats.get(mid)
        h = hist_avg.get(mid, {"avg": 0.0, "std": 0.0})
        if fc:
            forecasted_30 = fc["total_30"]
            avg_daily = fc["avg_daily"]
            demand_std = fc["std"] or h["std"]
        else:
            avg_daily = h["avg"]
            forecasted_30 = avg_daily * 30
            demand_std = h["std"]

        # --- practical inventory formulas ---
        safety_stock = demand_std * math.sqrt(lead_time)
        demand_during_lead = avg_daily * lead_time
        reorder_point = demand_during_lead + safety_stock
        recommended_order = max(0.0, forecasted_30 + safety_stock - current_stock)
        days_until_stockout = (current_stock / avg_daily) if avg_daily > 0 else float("inf")

        # --- expiry risk ---
        expiry = _parse_date(inv.get("expiry_date"))
        days_to_expiry = (expiry - today).days if expiry else None
        if days_to_expiry is None:
            expiry_risk = "unknown"
        elif days_to_expiry < 0:
            expiry_risk = "expired"
        elif days_to_expiry <= 30:
            expiry_risk = "high"
        elif days_to_expiry <= 90:
            expiry_risk = "medium"
        else:
            expiry_risk = "low"

        results.append({
            "medicine_id": mid,
            "medicine_name": med.get("medicine_name"),
            "category": med.get("category"),
            "current_stock": int(current_stock),
            "lead_time_days": int(lead_time),
            "forecasted_demand": round(forecasted_30, 2),
            "average_daily_demand": round(avg_daily, 3),
            "demand_std": round(demand_std, 3),
            "safety_stock": round(safety_stock, 2),
            "reorder_point": round(reorder_point, 2),
            "recommended_order_quantity": int(math.ceil(recommended_order)),
            "days_until_stockout": (round(days_until_stockout, 1)
                                    if math.isfinite(days_until_stockout) else None),
            "expiry_date": inv.get("expiry_date"),
            "days_to_expiry": days_to_expiry,
            "expiry_risk": expiry_risk,
        })
    return results
