"""Synthetic data generation for the pharmacy demand-forecasting prototype.

Generates realistic-but-fake data only. NO patient-level or personal data is
ever produced. Output is written both to MongoDB (preferred) and to the CSV
files under /data (fallback + reproducibility).

Run directly:  python seed_data.py
Or via API:    POST /api/seed
"""
import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import Config
import db.mongo as mongo

# Reproducible synthetic data
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

DAYS = 365
START_DATE = datetime.utcnow().date() - timedelta(days=DAYS)

CATEGORIES = [
    "Pain Relief",
    "Antibiotic",
    "Cold and Flu",
    "Allergy",
    "Diabetes",
    "Blood Pressure",
    "Digestive Health",
    "Vitamins",
    "Skin Care",
]

# A pool of plausible medicine names per category.
MEDICINE_NAMES = {
    "Pain Relief": ["Paracetamol 500mg", "Ibuprofen 400mg", "Aspirin 300mg", "Diclofenac Gel"],
    "Antibiotic": ["Amoxicillin 500mg", "Azithromycin 250mg", "Ciprofloxacin 500mg"],
    "Cold and Flu": ["Cough Syrup", "Decongestant Tablets", "Flu Relief Capsules", "Throat Lozenges"],
    "Allergy": ["Cetirizine 10mg", "Loratadine 10mg", "Antihistamine Nasal Spray"],
    "Diabetes": ["Metformin 500mg", "Glimepiride 2mg", "Insulin Pen"],
    "Blood Pressure": ["Amlodipine 5mg", "Losartan 50mg", "Atenolol 25mg"],
    "Digestive Health": ["Omeprazole 20mg", "Antacid Suspension", "Loperamide 2mg"],
    "Vitamins": ["Vitamin C 500mg", "Vitamin D3 1000IU", "Multivitamin Tablets", "Iron Supplement"],
    "Skin Care": ["Hydrocortisone Cream", "Antifungal Cream", "Moisturising Lotion"],
}

SUPPLIERS = ["MedSupply Co", "PharmaDirect", "HealthWholesale", "GlobalMeds", "CareDistributors"]
STORAGE_TYPES = ["Room Temp", "Refrigerated", "Cool Dry Place"]
WEATHER = ["Sunny", "Rainy", "Cloudy", "Cold", "Hot", "Humid"]


def _season_for(date):
    """Northern-hemisphere season from month."""
    m = date.month
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Spring"
    if m in (6, 7, 8):
        return "Summer"
    return "Autumn"


def generate_medicines():
    """Create 30 medicines spread across all categories."""
    medicines = []
    mid = 1
    # Ensure every category is represented, then fill to >= 30.
    pool = []
    for cat, names in MEDICINE_NAMES.items():
        for name in names:
            pool.append((cat, name))
    # pool has ~30 entries; cap/extend to exactly 30+.
    for cat, name in pool:
        # Assign a demand "profile": fast, medium, or slow moving.
        profile = random.choices(["fast", "medium", "slow"], weights=[0.3, 0.45, 0.25])[0]
        base_demand = {"fast": random.uniform(40, 70),
                       "medium": random.uniform(15, 35),
                       "slow": random.uniform(2, 10)}[profile]
        medicines.append({
            "medicine_id": f"M{mid:03d}",
            "medicine_name": name,
            "category": cat,
            "unit_price": round(random.uniform(2, 60), 2),
            "supplier": random.choice(SUPPLIERS),
            "storage_type": ("Refrigerated" if name in ("Insulin Pen",) else random.choice(STORAGE_TYPES)),
            # internal-only fields used by the generator (not part of schema)
            "_base_demand": round(base_demand, 2),
            "_profile": profile,
        })
        mid += 1
    return medicines


def generate_external_factors():
    """Daily external factors for the whole period."""
    factors = []
    outbreak_window = None  # tuple(start_idx, end_idx) for a simulated outbreak
    # Randomly schedule one outbreak lasting ~3 weeks.
    outbreak_start = random.randint(60, DAYS - 60)
    outbreak_window = (outbreak_start, outbreak_start + 21)

    for i in range(DAYS):
        date = START_DATE + timedelta(days=i)
        season = _season_for(date)
        # Temperature: seasonal sinusoid + noise (Celsius).
        day_of_year = date.timetuple().tm_yday
        temp = 15 + 12 * np.sin(2 * np.pi * (day_of_year - 100) / 365) + np.random.normal(0, 2)
        # Disease index: higher in winter, spikes during outbreak.
        disease = 0.3 + 0.4 * (season in ("Winter", "Autumn")) + np.random.uniform(0, 0.2)
        outbreak = 0
        if outbreak_window[0] <= i <= outbreak_window[1]:
            outbreak = 1
            disease += 0.4
        disease = round(float(min(disease, 1.0)), 3)
        local_event = 1 if random.random() < 0.05 else 0
        factors.append({
            "date": date.isoformat(),
            "temperature": round(float(temp), 1),
            "season": season,
            "disease_index": disease,
            "outbreak_alert": int(outbreak),
            "local_event_flag": int(local_event),
            "weather_condition": random.choice(WEATHER),
        })
    return factors


def generate_sales(medicines, factors):
    """Generate daily sales rows per medicine using seasonal + external effects."""
    factor_by_date = {f["date"]: f for f in factors}
    sales = []
    sale_id = 1
    for med in medicines:
        base = med["_base_demand"]
        cat = med["category"]
        unit_price = med["unit_price"]
        # small per-medicine multiplier for variety
        med_mult = random.uniform(0.85, 1.15)
        for f in factors:
            date = datetime.fromisoformat(f["date"]).date()
            season = f["season"]
            mult = 1.0
            # --- category-specific seasonality ---
            if cat == "Cold and Flu":
                mult *= 1.8 if season == "Winter" else (1.2 if season == "Autumn" else 0.7)
            elif cat == "Allergy":
                mult *= 1.9 if season == "Spring" else (1.1 if season == "Summer" else 0.7)
            elif cat == "Pain Relief":
                mult *= 1.0  # steady
            elif cat == "Skin Care":
                mult *= 1.3 if season == "Summer" else 0.9
            elif cat == "Vitamins":
                mult *= 1.2 if season in ("Winter", "Autumn") else 1.0
            # --- outbreak boost for respiratory/immunity related categories ---
            if f["outbreak_alert"] and cat in ("Cold and Flu", "Antibiotic", "Vitamins", "Pain Relief"):
                mult *= 1.6
            # --- disease index general lift ---
            mult *= (1 + 0.3 * f["disease_index"])
            # --- local event small bump ---
            if f["local_event_flag"]:
                mult *= 1.1
            # weekday effect: slightly busier mid-week
            dow = date.weekday()
            mult *= 1.1 if dow in (0, 1, 2) else (0.85 if dow >= 5 else 1.0)

            expected = base * med_mult * mult
            qty = int(max(0, np.random.poisson(max(expected, 0.1))))
            if qty <= 0:
                # still record zero-sales days occasionally so the series is continuous
                qty = 0
            selling_price = round(unit_price * random.uniform(1.15, 1.4), 2)
            sales.append({
                "sale_id": f"S{sale_id:06d}",
                "medicine_id": med["medicine_id"],
                "sale_date": date.isoformat(),
                "quantity_sold": qty,
                "selling_price": selling_price,
                "total_amount": round(qty * selling_price, 2),
            })
            sale_id += 1
    return sales


def generate_inventory(medicines, sales):
    """Create one current inventory record per medicine.

    Stock levels are set relative to recent demand so the demo shows a realistic
    mix of healthy stock, low stock, and overstock (with some near-expiry).
    """
    sales_df = pd.DataFrame(sales)
    recent_cutoff = (datetime.utcnow().date() - timedelta(days=30)).isoformat()
    inventory = []
    today = datetime.utcnow().date()
    for med in medicines:
        mid = med["medicine_id"]
        recent = sales_df[(sales_df["medicine_id"] == mid) &
                          (sales_df["sale_date"] >= recent_cutoff)]
        avg_daily = recent["quantity_sold"].mean() if len(recent) else med["_base_demand"]
        avg_daily = float(avg_daily) if not pd.isna(avg_daily) else med["_base_demand"]

        # Deliberately vary stock scenarios for a richer demo.
        scenario = random.choices(
            ["healthy", "low", "overstock"], weights=[0.5, 0.3, 0.2]
        )[0]
        if scenario == "low":
            current_stock = int(avg_daily * random.uniform(2, 6))
        elif scenario == "overstock":
            current_stock = int(avg_daily * random.uniform(60, 120) + 50)
        else:
            current_stock = int(avg_daily * random.uniform(20, 40) + 20)

        lead_time = random.choice([3, 5, 7, 10, 14])
        reorder_threshold = int(max(avg_daily * lead_time * 1.5, 5))

        # Expiry: mix of near-expiry and long-dated batches.
        expiry_scenario = random.choices(
            ["near", "soon", "far"], weights=[0.15, 0.25, 0.6]
        )[0]
        if expiry_scenario == "near":
            expiry = today + timedelta(days=random.randint(5, 30))
        elif expiry_scenario == "soon":
            expiry = today + timedelta(days=random.randint(31, 120))
        else:
            expiry = today + timedelta(days=random.randint(180, 720))

        inventory.append({
            "medicine_id": mid,
            "current_stock": int(max(current_stock, 0)),
            "batch_number": f"B{random.randint(10000, 99999)}",
            "expiry_date": expiry.isoformat(),
            "reorder_threshold": int(reorder_threshold),
            "lead_time_days": int(lead_time),
        })
    return inventory


def _strip_internal(medicines):
    """Remove generator-only fields (prefixed with _) before persisting."""
    cleaned = []
    for m in medicines:
        cleaned.append({k: v for k, v in m.items() if not k.startswith("_")})
    return cleaned


def seed(persist_csv=True):
    """Generate everything and write to Mongo (+ CSV fallback). Returns counts."""
    medicines = generate_medicines()
    factors = generate_external_factors()
    sales = generate_sales(medicines, factors)
    inventory = generate_inventory(medicines, sales)

    clean_medicines = _strip_internal(medicines)

    # Persist to Mongo (or CSV fallback inside replace_collection).
    mongo.replace_collection(mongo.MEDICINES, clean_medicines)
    mongo.replace_collection(mongo.SALES, sales)
    mongo.replace_collection(mongo.INVENTORY, inventory)
    mongo.replace_collection(mongo.EXTERNAL_FACTORS, factors)

    # Always also write CSVs for reproducibility / dataset explanation.
    if persist_csv:
        os.makedirs(Config.DATA_DIR, exist_ok=True)
        pd.DataFrame(clean_medicines).to_csv(Config.MEDICINES_CSV, index=False)
        pd.DataFrame(sales).to_csv(Config.SALES_CSV, index=False)
        pd.DataFrame(inventory).to_csv(Config.INVENTORY_CSV, index=False)
        pd.DataFrame(factors).to_csv(Config.EXTERNAL_CSV, index=False)

    return {
        "medicines": len(clean_medicines),
        "sales": len(sales),
        "inventory": len(inventory),
        "external_factors": len(factors),
        "mongo": mongo.is_mongo_available(),
    }


if __name__ == "__main__":
    result = seed()
    print("Seed complete:")
    for k, v in result.items():
        print(f"  {k}: {v}")
