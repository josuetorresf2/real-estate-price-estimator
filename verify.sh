#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
export PYTHONPATH="${PYTHONPATH:-src}"

"$PYTHON_BIN" -m pytest
"$PYTHON_BIN" scripts/generate_sample_data.py
"$PYTHON_BIN" -m real_estate_price_estimator.cli train \
  --data data/generated_housing.csv \
  --model-out models/price_pipeline.joblib \
  --min-r2 0.80
"$PYTHON_BIN" -m real_estate_price_estimator.cli predict \
  --model models/price_pipeline.joblib \
  --city Austin \
  --neighborhood "North Loop" \
  --zip-code 78751 \
  --square-feet 1850 \
  --bedrooms 3 \
  --bathrooms 2 \
  --lot-size 0.18 \
  --year-built 1998 \
  --school-rating 8.6 \
  --distance-to-city-center-miles 4.2 \
  --crime-index 31
