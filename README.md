# 💊 AI-Based Medicine Demand Forecasting System for Pharmacy Inventory Optimization

A university software-engineering **prototype** that helps pharmacy staff forecast
medicine demand, monitor inventory, detect expiry risk, and get restocking
recommendations. It trains and compares **Linear Regression**, **Random Forest**,
and **LSTM** models, automatically selects the best one (lowest MAE), and presents
everything in a clean Bootstrap + Plotly dashboard.

> ⚠️ **Disclaimer:** This is an educational prototype, **not a clinical or medical
> decision-making product**. It uses only **synthetic, non-personal** pharmacy
> sales/inventory data and does **not** provide medical advice.

---

## ✨ Features

- **Synthetic data generator** — 30+ medicines across 9 categories, 365 days of sales,
  inventory with stock & expiry, and daily external factors (temperature, season,
  disease index, outbreak alerts, weather).
- **Realistic demand patterns** — cold & flu rise in winter, allergy in spring, pain
  relief steady, outbreak spikes, plus fast- and slow-moving medicines.
- **Feature engineering** — calendar features, lags, rolling means, seasonality,
  external factors, category encoding.
- **Three forecasting models** compared on the same chronological test period.
- **Automatic model selection** by lowest MAE, persisted to `selected_model.json` & MongoDB.
- **Inventory optimization** — safety stock, reorder point, recommended order qty,
  days-until-stockout, expiry risk.
- **Alert engine** — LOW_STOCK, EXPIRY_RISK, RESTOCK_NEEDED, OVERSTOCK_WARNING,
  MODEL_LOW_CONFIDENCE with critical/high/medium/low severity.
- **Dashboard** — summary cards, demand/severity/category/model charts, restock &
  expiry & alert tables.
- **CSV upload** — bring your own sales / inventory / external-factor data.
- **MongoDB-first** with automatic **CSV fallback** when no database is reachable.
- **Docker + Easypanel ready** with Gunicorn.

---

## 🏗️ Architecture

```
Browser (Bootstrap 5 + Plotly.js)
        │  fetch() JSON
        ▼
Flask app (app.py)  ── pages + /api/* endpoints
        │
        ├── ml/           preprocessing → models → forecasting → evaluation
        ├── inventory/    optimizer → alerts
        └── db/mongo.py   MongoDB (preferred) ⇄ CSV fallback (/data)
        │
        ▼
MongoDB collections: medicines, sales, inventory, external_factors,
                     forecasts, alerts, model_metrics
```

Model artifacts are stored in `/models`
(`linear_regression.pkl`, `random_forest.pkl`, `lstm_model.keras`, `selected_model.json`).

---

## 🧰 Tech Stack

| Layer       | Technology |
|-------------|------------|
| Backend     | Python 3.11, Flask, Gunicorn |
| Frontend    | HTML, CSS, JavaScript, Bootstrap 5, Plotly.js |
| Database    | MongoDB (`MONGO_URI`) with CSV fallback |
| ML          | pandas, numpy, scikit-learn, tensorflow-cpu |
| Packaging   | pip, Docker |
| Deployment  | Dockerfile + docker-compose, Easypanel |

---

## 📊 Dataset Explanation

The system uses a **real-world pharmacy sales dataset** by default, with a synthetic
generator as a fallback.

**Primary (real):** [*Pharma Sales Data* by Milan Zdravkovic](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data)
— a single pharmacy's point-of-sale records, **Jan 2014 – Oct 2019** (2,106 daily
records), grouped into **8 ATC drug categories** (e.g. `R06` antihistamines/allergy,
`R03` respiratory, `N02BE` paracetamol, `N05C` sedatives). It contains genuine,
non-fabricated seasonality — antihistamine sales peak in spring, respiratory drugs in
winter — which is what the models actually learn. Fetch it with
`bash scripts/fetch_real_data.sh` (no Kaggle login required; pulls a public mirror).

**What is real vs. derived** (we are explicit about this):

| Entity | Source |
|--------|--------|
| **Sales** | 100% real daily units sold per ATC group |
| **Medicines** | real ATC groups mapped to readable names/categories |
| **Inventory** (stock, expiry, lead time) | **derived** — public sales data has no stock/expiry; a plausible current snapshot is synthesized from *real* recent demand so the optimization math has something to run on |
| **External factors** (temperature, disease index) | **neutral placeholders** — not in the source data; we do *not* fabricate epidemiology. The demand signal comes purely from the real series + real calendar (month/weekday/season) |

**Fallback (synthetic):** if the real CSV is absent, `seed_data.py` (seed = 42) generates
30 medicines × 365 days with modelled seasonality. Force it via `POST /api/seed {"synthetic": true}`.

Both paths populate the same schema and CSV files in `/data`:

| Entity | Key fields |
|--------|-----------|
| **Medicine** | medicine_id, medicine_name, category, unit_price, supplier, storage_type |
| **Sales** | sale_id, medicine_id, sale_date, quantity_sold, selling_price, total_amount |
| **Inventory** | medicine_id, current_stock, batch_number, expiry_date, reorder_threshold, lead_time_days |
| **External Factors** | date, temperature, season, disease_index, outbreak_alert, local_event_flag, weather_condition |
| **Forecast Output** | medicine_id, forecast_date, predicted_quantity, model_name, confidence_level |
| **Alert Output** | alert_id, medicine_id, alert_type, severity, message, created_at |

Demand is generated with category seasonality, an outbreak window, disease-index lift,
weekday effects, and per-medicine fast/medium/slow profiles.

---

## 🤖 ML Models

| Model | Library | Notes |
|-------|---------|-------|
| **Linear Regression** | scikit-learn (`StandardScaler` pipeline) | baseline |
| **Random Forest Regressor** | scikit-learn | 120 trees, depth 12 — VPS-friendly |
| **LSTM** | tensorflow-cpu | small (32-unit) **per-medicine** sequence model (seq len 14) |
| **Naive (seasonal)** | — | baseline: "same weekday last week" — the bar to beat |

### Honest, fair evaluation methodology

This is the part most student projects get wrong. Here every model is judged on a
level playing field:

- **Same task.** All models — including the LSTM — predict the **same target**:
  next-day demand for each `(medicine, date)`. They are scored on the **same
  held-out points**, so no model is secretly solving an easier problem.
- **Naive baseline + skill score.** A **seasonal-naive baseline** ("same weekday last
  week") is included. We report a **skill score** = `1 − MAE_model / MAE_naive`; a
  model is only useful if skill > 0 (it actually beats the baseline). On the real
  dataset, RF/LR/LSTM each beat the naive baseline by **~23–27%**.
- **Rolling-origin backtest.** Metrics are averaged over **3 walk-forward folds**
  (30-day test windows each, 720 pooled test points) instead of one lucky holdout.
- **Selection.** The lowest-MAE candidate model is selected (`selected_model.json`).
  When the LSTM wins, per-medicine 30-day forecasts still use the best tabular model
  (RF) for operational stability — the LSTM remains a first-class participant in the
  comparison.
- **Robustness.** If TensorFlow is missing or LSTM training fails, the system
  **continues with LR + RF + baseline** (no crash).

### Evaluation metrics
- **MAE** — Mean Absolute Error *(headline metric)*
- **RMSE** — Root Mean Squared Error
- **MAPE** — Mean Absolute Percentage Error (zero-actual points excluded). Note: MAPE
  runs high on this dataset because several drugs have many genuine zero-demand days
  (intermittent demand) — MAE/RMSE/skill are the meaningful metrics here.

---

## 📦 Inventory Optimization Formulas

```
safety_stock              = demand_std * sqrt(lead_time_days)
demand_during_lead        = average_daily_demand * lead_time_days
reorder_point             = demand_during_lead + safety_stock
recommended_order_qty     = max(0, forecasted_demand_next_30_days + safety_stock - current_stock)
days_until_stockout       = current_stock / average_daily_demand     (when avg > 0)
```

Expiry risk: `expired` (<0 days), `high` (≤30), `medium` (≤90), `low` (otherwise).

---

## 🚨 Alert Types

| Type | Condition | Typical severity |
|------|-----------|------------------|
| `LOW_STOCK` | current_stock ≤ reorder_point | critical/high/medium (by days-to-stockout) |
| `EXPIRY_RISK` | expiry within 30 days / expired | critical/high/medium |
| `RESTOCK_NEEDED` | recommended_order_quantity > 0 | high/low |
| `OVERSTOCK_WARNING` | stock ≫ forecast **and** expiry risk | high/medium |
| `MODEL_LOW_CONFIDENCE` | too little sales history | low |

---

## 🖥️ Local Setup (without Docker)

```bash
# 1. Create a virtual environment (Python 3.11 recommended)
python3.11 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment
cp .env.example .env                 # edit MONGO_URI if needed

# 4. Fetch the real dataset (public mirror; no Kaggle login needed)
bash scripts/fetch_real_data.sh

# 5a. (Optional) ingest + train from the CLI
python ingest_real.py                # or: python seed_data.py  (synthetic fallback)
python train_models.py --forecast

# 4b. Run the app
python app.py                        # dev server on http://localhost:5000
```

> No MongoDB? The app automatically falls back to CSV files in `/data`, so the demo
> still works. The status badge in the top bar shows "MongoDB connected" or
> "CSV fallback mode".

You can also drive everything from the **dashboard buttons**: Seed → Train → Forecast.

---

## 🧪 Tests

Fast unit tests cover the pure logic (evaluation metrics, skill score, LSTM
sequence windows) — no database or TensorFlow required:

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## 🐳 Docker Setup (local)

```bash
# Build and run app + MongoDB together
docker compose up --build

# Open the dashboard (host port 8080 -> container 5000)
open http://localhost:8080           # or visit in your browser

# Stop
docker compose down                  # add -v to also remove the mongo volume
```

> **Why 8080?** macOS uses port 5000 for AirPlay/Control Center (it returns HTTP 403),
> so `docker-compose.yml` maps host **8080** → container **5000**. On Linux/Easypanel
> the container simply listens on 5000. To use 5000 locally, disable *AirPlay Receiver*
> in System Settings and change the port mapping back.

Build/run the image on its own (pointing at an external MongoDB):

```bash
docker build -t pharma-forecast .
docker run -p 5000:5000 \
  -e MONGO_URI="mongodb://host.docker.internal:27017/medicine_forecasting" \
  -e SECRET_KEY="change-this-secret" \
  pharma-forecast
```

---

## ☁️ Easypanel Deployment

The project is Easypanel-ready: the **Dockerfile is at the project root**, the app
listens on the **`PORT`** environment variable (default 5000), and **Gunicorn** runs
in production.

### Option A — App + MongoDB Atlas (recommended)

1. **Push the project to GitHub.**
2. Open **Easypanel**.
3. Create a **new Project**.
4. Add an **App** service.
5. Select your **GitHub repository**.
6. Easypanel will **build using the Dockerfile** automatically.
7. Add **environment variables**:
   - `MONGO_URI` → your MongoDB Atlas connection string
   - `MONGO_DB_NAME` → `medicine_forecasting`
   - `SECRET_KEY` → a strong random secret
   - `FLASK_ENV` → `production`
8. Add a **domain**.
9. Enable **SSL**.
10. **Deploy.**

After deploy, open the domain and run **Seed → Train → Forecast** from the top bar.

### Option B — App + MongoDB container (no Atlas)

If you prefer a Mongo container instead of Atlas:

1. In your Easypanel project, add a **MongoDB** service (or a Docker service using the
   `mongo:7` image) with a persistent volume on `/data/db`.
2. Add the **App** service from your GitHub repo (Dockerfile build) as above.
3. Set the App's `MONGO_URI` to the internal Mongo service hostname, e.g.
   `mongodb://mongo:27017/medicine_forecasting`.
4. Set `MONGO_DB_NAME`, `SECRET_KEY`, `FLASK_ENV=production`.
5. Add a domain, enable SSL, and deploy.

> The included `docker-compose.yml` mirrors Option B and is handy for verifying the
> full stack locally before deploying.

---

## 🎬 Demo Flow

1. **Open the dashboard.**
2. Click **Seed** (generates synthetic data).
3. Click **Train** (trains LR, RF, LSTM; selects best model).
4. Click **Forecast** (30-day forecasts + inventory optimization + alerts).
5. Open **Model Comparison** to see metrics and the selected model.
6. Open **Forecasts** / **Medicines** for inventory recommendations.
7. Open **Alerts** for low-stock and expiry alerts.

---

## ⚠️ Limitations

- Real **sales** data is used, but **inventory** (stock/expiry) is derived — public
  datasets don't expose private operational stock data.
- The real dataset has only **8 drug categories** (ATC groups), not individual SKUs.
- External factors (temperature/disease) are placeholders for the real dataset; the
  signal comes from the real series + calendar features.
- Forecasts are point estimates (confidence is heuristic, not a statistical interval).
- The LSTM is kept tiny for VPS friendliness; on this dataset it is competitive with
  but does not beat Random Forest.
- No authentication/authorization (intentionally — this is a prototype).
- **Not medical advice; not for clinical use.**

---

## 🚀 Future Enhancements

- Probabilistic forecasts with prediction intervals.
- Per-medicine deep-learning models and hyperparameter tuning.
- Supplier lead-time variability and multi-echelon inventory.
- Role-based authentication and audit logging.
- Scheduled automatic retraining and alert notifications (email/Slack).
- Real dataset ingestion connectors and data-quality checks.

---

## 📁 Project Structure

```
.
├── app.py                  # Flask app: pages + API
├── config.py               # env-driven configuration
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── ingest_real.py          # load the REAL Kaggle pharma-sales dataset
├── seed_data.py            # synthetic data generation (fallback)
├── train_models.py         # CLI: train + (optional) forecast
├── scripts/
│   └── fetch_real_data.sh  # download the real dataset (public mirror)
├── ml/
│   ├── preprocessing.py    # feature engineering + train/test + LSTM sequences
│   ├── models.py           # model builders + (de)serialization
│   ├── forecasting.py      # training orchestration + forecast generation
│   └── evaluation.py       # MAE / RMSE / MAPE
├── inventory/
│   ├── optimizer.py        # safety stock, reorder point, order qty, expiry
│   └── alerts.py           # alert generation
├── db/
│   └── mongo.py            # MongoDB access + CSV fallback
├── templates/              # base, dashboard, medicines, forecasts, alerts,
│                           #   upload, model_comparison
├── static/
│   ├── css/style.css
│   └── js/{dashboard.js,charts.js}
├── data/                   # synthetic_*.csv (generated)
└── models/                 # *.pkl, *.keras, selected_model.json (generated)
```
