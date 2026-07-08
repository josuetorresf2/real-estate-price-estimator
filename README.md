# Real Estate Price Estimator

A premium 3D web app and regression pipeline for estimating U.S. home sale prices from user-entered property facts, ZIP-level market signals, and public government location data.

![Estimator hero](docs/images/estimator-hero.jpg)

## What It Does

- Predicts a sale-price estimate with a scikit-learn regression pipeline.
- Lets users enter `unknown` or leave most property facts blank.
- Uses model imputation when square footage, bedrooms, bathrooms, lot size, year built, school rating, distance, or crime index are not known.
- Uses a supplied address to look up city, state, ZIP, and map coordinates through the U.S. Census Geocoder.
- Calibrates estimates with Zillow Research ZIP-level ZHVI data when available.
- Optionally blends U.S. Census ACS median home value when `CENSUS_API_KEY` is configured.
- Shows an OpenStreetMap area preview when coordinates are available.
- Runs a premium animated 3D interface with a locally bundled Three.js module, so the scene does not depend on a CDN.

![Estimate result](docs/images/estimate-result.jpg)

## Data Sources

The app is intentionally transparent about source quality:

- **Zillow Research ZHVI**: free ZIP-level typical home value signal. This is not an address-level Zestimate.
- **U.S. Census Geocoder**: free address-to-location lookup for city, state, ZIP, and coordinates.
- **U.S. Census ACS 5-year B25077**: optional government median owner-occupied home value by ZIP Code Tabulation Area. Requires `CENSUS_API_KEY`.
- **Trained regression model**: fills gaps using the model pipeline's categorical and numeric imputers.
- **OpenStreetMap**: area map preview when the Census Geocoder returns coordinates.

Public/free data generally does not include exact address-level beds, baths, square footage, lot size, or year built. Those fields can be entered when known; otherwise the app estimates using imputation and market context.

Source references:

- [Zillow Research Data](https://www.zillow.com/research/data/)
- [Zillow Public Real Estate Metrics API](https://www.zillowgroup.com/developers/api/public-data/real-estate-metrics/)
- [U.S. Census Geocoder](https://geocoding.geo.census.gov/)
- [U.S. Census ACS 5-Year API](https://www.census.gov/data/developers/data-sets/acs-5year.html)
- [OpenStreetMap](https://www.openstreetmap.org/)

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./verify.sh
```

Run the browser app:

```bash
PYTHONPATH=src python -m real_estate_price_estimator.web_app --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

The first market-backed estimate may take longer because the app downloads and caches Zillow Research's free ZIP-level ZHVI CSV in `data/zillow_zhvi_zip.csv`.

## Optional Census API

The U.S. Census ACS API can add a government median-home-value signal. Set this only if you have a Census API key:

```bash
export CENSUS_API_KEY=your_key_here
PYTHONPATH=src python -m real_estate_price_estimator.web_app --port 8000
```

Without a key, the app still uses the Census Geocoder and Zillow Research ZHVI.

## Training

Generate deterministic sample data for local smoke tests:

```bash
python scripts/generate_sample_data.py
```

Train a model:

```bash
python -m real_estate_price_estimator.cli train \
  --data data/sample_housing.csv \
  --model-out models/price_pipeline.joblib
```

Predict from the CLI:

```bash
python -m real_estate_price_estimator.cli predict \
  --model models/price_pipeline.joblib \
  --city Austin \
  --neighborhood unknown \
  --zip-code 78751 \
  --square-feet 1850 \
  --bedrooms 3 \
  --bathrooms 2 \
  --lot-size 0.18 \
  --school-rating 8.6 \
  --distance-to-city-center-miles 4.2 \
  --crime-index 31
```

## Use Your Own Data

Training CSV columns:

```text
city,neighborhood,zip_code,square_feet,bedrooms,bathrooms,lot_size,year_built,school_rating,distance_to_city_center_miles,crime_index,price
```

`price` is the training target. For web predictions, most inputs can be blank or `unknown`; for training, provide complete rows where possible so the model learns a stronger local relationship.

## Verification

```bash
./verify.sh
```

Current verification covers:

- Pipeline training and prediction.
- Unknown/blank property field parsing.
- Zillow Research ZHVI parsing and blending.
- Optional Census ACS blending logic.
- Address lookup data application.
- Browser route rendering.

## Production Notes

This is not an appraisal product. It is a market-aware estimator. Production use should retrain on current MLS, assessor, parcel, tax, school, and neighborhood data, and should use an approved property-record API if exact address-level facts are required.
