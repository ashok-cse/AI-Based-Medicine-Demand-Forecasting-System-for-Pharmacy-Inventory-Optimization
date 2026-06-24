"""Ingest the REAL pharmacy sales dataset into the system schema.

Source: "Pharma Sales Data" by Milan Zdravkovic (Kaggle), a single pharmacy's
point-of-sale records, Jan 2014 - Oct 2019, grouped into 8 ATC drug categories.
A mirror of `salesdaily.csv` is bundled under data/real/.

What's real vs. derived (we are explicit about this):
  * SALES            -> 100% real (daily units sold per ATC group).
  * MEDICINES        -> real ATC groups mapped to readable names/categories.
  * INVENTORY        -> DERIVED. Public sales data contains no stock levels or
                        expiry dates (those are private operational data), so we
                        synthesize a plausible current snapshot from real recent
                        demand. The optimization math then runs on it.
  * EXTERNAL FACTORS -> calendar-derived only (month/weekday/season). We do NOT
                        fabricate temperature/disease/outbreak signals; those
                        columns are written as neutral placeholders so the demand
                        signal the models learn comes purely from the REAL series.
"""
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import Config
import db.mongo as mongo

random.seed(42)
np.random.seed(42)

REAL_CSV = os.path.join(Config.DATA_DIR, "real", "salesdaily.csv")

# ATC code -> (medicine_name, category, storage_type, unit_price)
# Descriptions follow the WHO ATC classification used by the dataset.
ATC_MAP = {
    "M01AB": ("Diclofenac (NSAID)", "Anti-inflammatory", "Room Temp", 6.50),
    "M01AE": ("Ibuprofen (NSAID)", "Anti-inflammatory", "Room Temp", 4.20),
    "N02BA": ("Aspirin / Salicylates", "Pain Relief", "Room Temp", 3.10),
    "N02BE": ("Paracetamol / Anilides", "Pain Relief", "Room Temp", 2.80),
    "N05B": ("Anxiolytics (Diazepam)", "Mental Health", "Cool Dry Place", 9.40),
    "N05C": ("Hypnotics & Sedatives", "Mental Health", "Cool Dry Place", 11.20),
    "R03": ("Asthma / COPD Inhalers", "Respiratory", "Room Temp", 18.75),
    "R06": ("Antihistamines (Allergy)", "Allergy", "Room Temp", 5.60),
}
ATC_CODES = list(ATC_MAP.keys())
SUPPLIERS = ["MedSupply Co", "PharmaDirect", "HealthWholesale", "GlobalMeds"]


def _season_for(month):
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8):
        return "Summer"
    return "Autumn"


def _load_real_csv():
    if not os.path.exists(REAL_CSV):
        raise FileNotFoundError(
            f"Real dataset not found at {REAL_CSV}. "
            "Run scripts/fetch_real_data.sh or place salesdaily.csv there."
        )
    df = pd.read_csv(REAL_CSV)
    df["datum"] = pd.to_datetime(df["datum"], format="%m/%d/%Y")
    return df.sort_values("datum")


def _build_medicines():
    meds = []
    for i, code in enumerate(ATC_CODES, start=1):
        name, category, storage, price = ATC_MAP[code]
        meds.append({
            "medicine_id": code,                 # ATC code doubles as the id
            "medicine_name": name,
            "category": category,
            "unit_price": price,
            "supplier": SUPPLIERS[i % len(SUPPLIERS)],
            "storage_type": storage,
        })
    return meds


def _build_sales(df):
    """Melt the wide real CSV into long (medicine_id, sale_date, quantity_sold)."""
    long = df.melt(id_vars=["datum"], value_vars=ATC_CODES,
                   var_name="medicine_id", value_name="quantity_sold")
    long = long.dropna(subset=["quantity_sold"])
    sales = []
    for i, row in enumerate(long.itertuples(index=False), start=1):
        qty = round(float(row.quantity_sold), 2)
        price = ATC_MAP[row.medicine_id][3]
        selling = round(price * 1.3, 2)
        sales.append({
            "sale_id": f"S{i:07d}",
            "medicine_id": row.medicine_id,
            "sale_date": row.datum.date().isoformat(),
            "quantity_sold": qty,
            "selling_price": selling,
            "total_amount": round(qty * selling, 2),
        })
    return sales


def _build_external(df):
    """Calendar-derived external factors (no fabricated weather/epidemiology)."""
    factors = []
    for d in df["datum"]:
        factors.append({
            "date": d.date().isoformat(),
            "temperature": 20.0,          # neutral placeholder (not in source data)
            "season": _season_for(d.month),
            "disease_index": 0.0,         # placeholder; real signal is in the series
            "outbreak_alert": 0,
            "local_event_flag": 0,
            "weather_condition": "Unknown",
        })
    return factors


def _build_inventory(df, medicines):
    """Derive a plausible current inventory snapshot from REAL recent demand."""
    today = datetime.utcnow().date()
    # Average daily demand over the last 60 real days per medicine.
    recent = df[df["datum"] >= df["datum"].max() - pd.Timedelta(days=60)]
    inventory = []
    for med in medicines:
        code = med["medicine_id"]
        avg_daily = float(recent[code].mean()) if code in recent else 1.0
        avg_daily = max(avg_daily, 0.5)
        scenario = random.choices(["healthy", "low", "overstock"],
                                  weights=[0.5, 0.3, 0.2])[0]
        if scenario == "low":
            stock = int(avg_daily * random.uniform(2, 6))
        elif scenario == "overstock":
            stock = int(avg_daily * random.uniform(70, 130) + 50)
        else:
            stock = int(avg_daily * random.uniform(20, 40) + 20)
        lead = random.choice([3, 5, 7, 10, 14])
        expiry_scn = random.choices(["near", "soon", "far"], weights=[0.25, 0.25, 0.5])[0]
        if expiry_scn == "near":
            expiry = today + timedelta(days=random.randint(5, 30))
        elif expiry_scn == "soon":
            expiry = today + timedelta(days=random.randint(31, 120))
        else:
            expiry = today + timedelta(days=random.randint(180, 720))
        inventory.append({
            "medicine_id": code,
            "current_stock": max(stock, 0),
            "batch_number": f"B{random.randint(10000, 99999)}",
            "expiry_date": expiry.isoformat(),
            "reorder_threshold": int(max(avg_daily * lead * 1.5, 3)),
            "lead_time_days": lead,
        })
    return inventory


def ingest(persist_csv=True):
    """Load the real dataset into Mongo (+ CSV fallback). Returns counts."""
    df = _load_real_csv()
    medicines = _build_medicines()
    sales = _build_sales(df)
    external = _build_external(df)
    inventory = _build_inventory(df, medicines)

    mongo.replace_collection(mongo.MEDICINES, medicines)
    mongo.replace_collection(mongo.SALES, sales)
    mongo.replace_collection(mongo.INVENTORY, inventory)
    mongo.replace_collection(mongo.EXTERNAL_FACTORS, external)

    if persist_csv:
        pd.DataFrame(medicines).to_csv(Config.MEDICINES_CSV, index=False)
        pd.DataFrame(sales).to_csv(Config.SALES_CSV, index=False)
        pd.DataFrame(inventory).to_csv(Config.INVENTORY_CSV, index=False)
        pd.DataFrame(external).to_csv(Config.EXTERNAL_CSV, index=False)

    return {
        "source": "real (Kaggle pharma-sales-data, daily)",
        "medicines": len(medicines),
        "sales": len(sales),
        "inventory": len(inventory),
        "external_factors": len(external),
        "date_range": f"{df['datum'].min().date()} to {df['datum'].max().date()}",
        "mongo": mongo.is_mongo_available(),
    }


if __name__ == "__main__":
    result = ingest()
    print("Real-data ingestion complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")
