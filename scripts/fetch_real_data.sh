#!/usr/bin/env bash
# Fetch the real pharmacy sales dataset (daily granularity).
#
# Primary source: "Pharma Sales Data" by Milan Zdravkovic on Kaggle:
#   https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data
# Kaggle requires authentication, so by default we pull `salesdaily.csv` from a
# public GitHub mirror of the same dataset. If you have the Kaggle CLI configured
# you can instead run:  kaggle datasets download -d milanzdravkovic/pharma-sales-data
set -euo pipefail

DEST_DIR="$(cd "$(dirname "$0")/.." && pwd)/data/real"
mkdir -p "$DEST_DIR"

URL="https://raw.githubusercontent.com/vmtamburro/Pharma-Sales-Analysis/main/salesdaily.csv"
echo "Downloading salesdaily.csv ..."
curl -fsSL "$URL" -o "$DEST_DIR/salesdaily.csv"
echo "Saved to $DEST_DIR/salesdaily.csv"
wc -l "$DEST_DIR/salesdaily.csv"
