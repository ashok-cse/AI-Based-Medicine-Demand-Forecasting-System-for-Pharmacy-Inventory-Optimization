"""Command-line entry point to train models and generate forecasts.

Usage:
    python train_models.py            # train + evaluate + select best model
    python train_models.py --forecast # also generate forecasts, optimize, alert
"""
import sys
import logging

from ml.forecasting import train_all_models, generate_forecasts
from inventory.optimizer import optimize_inventory
from inventory.alerts import generate_alerts
import db.mongo as mongo

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def main():
    if mongo.database_is_empty():
        print("Database empty — run `python ingest_real.py` (real data) "
              "or `python seed_data.py` (synthetic) first.")
        sys.exit(1)

    print("Training models...")
    result = train_all_models()
    print(f"Best model: {result['best_model']}")
    for c in result["comparison"]:
        star = " *" if c["is_best"] else ""
        print(f"  {c['model_name']:<18} MAE={c['mae']}  RMSE={c['rmse']}  MAPE={c['mape']}{star}")

    if "--forecast" in sys.argv:
        print("\nGenerating forecasts...")
        fc = generate_forecasts()
        print(f"  {len(fc)} forecast rows generated.")
        optimized = optimize_inventory()
        alerts = generate_alerts(optimized)
        print(f"  {len(optimized)} inventory recommendations, {len(alerts)} alerts.")


if __name__ == "__main__":
    main()
