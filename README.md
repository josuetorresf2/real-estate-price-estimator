# Real Estate Price Estimator

A regression pipeline for estimating home sale prices from property, location, and school-quality features.

The project uses a scikit-learn `Pipeline` so categorical encoding, numeric scaling, model fitting, evaluation, and prediction stay reproducible.

## Features

- Location fields: `city`, `neighborhood`, `zip_code`
- Property fields: `square_feet`, `bedrooms`, `bathrooms`, `lot_size`, `year_built`
- Local context: `school_rating`, `distance_to_city_center_miles`, `crime_index`
- Optional free Zillow Research ZIP-level ZHVI market calibration
- Premium 3D browser UI with a local bundled Three.js runtime
- Trains a `HistGradientBoostingRegressor`
- Saves the trained model artifact with preprocessing included
- CLI commands for training, evaluation, and single-property prediction

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest
```

Run the full local verification predicate:

```bash
./verify.sh
```

If you do not install the package in editable mode, run CLI commands with `PYTHONPATH=src`.

Run the browser app:

```bash
PYTHONPATH=src python -m real_estate_price_estimator.web_app --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

The first market-backed estimate may take longer because the app downloads and caches Zillow Research's free ZIP-level ZHVI CSV in `data/zillow_zhvi_zip.csv`.

Generate a larger deterministic training sample for smoke testing:

```bash
python scripts/generate_sample_data.py
```

Train with the included sample data:

```bash
python -m real_estate_price_estimator.cli train \
  --data data/sample_housing.csv \
  --model-out models/price_pipeline.joblib
```

Predict one listing:

```bash
python -m real_estate_price_estimator.cli predict \
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
```

## Use Your Own Data

Provide a CSV with these columns:

```text
city,neighborhood,zip_code,square_feet,bedrooms,bathrooms,lot_size,year_built,school_rating,distance_to_city_center_miles,crime_index,price
```

The `price` column is the training target and should be omitted only for prediction input.

To serve an app backed by your own model:

```bash
PYTHONPATH=src python -m real_estate_price_estimator.cli train \
  --data path/to/your_housing.csv \
  --model-out models/local_market.joblib

PYTHONPATH=src python -m real_estate_price_estimator.web_app \
  --model models/local_market.joblib \
  --port 8000
```

## Notes

The sample dataset is synthetic and intended for development only. For production use, retrain with current local MLS, assessor, school district, and neighborhood market data.

Zillow integration uses the public Zillow Research ZHVI ZIP-level dataset. It is a market signal for the typical home value in a ZIP code, not an address-level Zestimate. Proper attribution to Zillow is shown in the app.
