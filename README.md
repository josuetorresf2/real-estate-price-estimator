# Property Valuation Workbench

A production-style real estate valuation workbench for verifying addresses, mapping properties, checking public data coverage, and producing transparent market-aware estimates.

## Mission

Housing price tools are often either too expensive, too opaque, or only useful in one market. This project demonstrates a practical alternative: an evidence-first valuation workflow that explains what data was found, what data is missing, and when a model should not pretend to know more than the sources support.

## Problem Being Solved

Most simple real estate estimators fail in two ways:

- They ask users for facts they may not know, such as square footage, lot size, or year built.
- They produce a confident-looking number even when the data is weak.

This app is built to show stronger software engineering judgment. It verifies addresses, enriches the form from real providers when possible, shows map context, uses public market signals when available, and clearly labels missing or low-confidence data.

## Try These Examples

Select the country first, then paste the matching address/search text into `Address`.

| Country | Try this address/search | What should happen |
| --- | --- | --- |
| United States | `1701 Wynkoop St Denver CO` | Verifies with Census, fills city/state/ZIP, maps the property, and can use Zillow ZIP data. |
| Ecuador | `La Carolina Quito` | Verifies with OpenStreetMap/Nominatim and shows Ecuador regional macro context. |
| Brazil | `Avenida Paulista Sao Paulo` | Verifies with OpenStreetMap/Nominatim and shows Brazil regional macro context. |
| Peru | `Miraflores Lima` | Verifies district/city context and shows Peru regional macro context. |
| Colombia | `El Poblado Medellin` | Verifies location context and shows Colombia regional macro context. |
| Chile | `Providencia Santiago` | Verifies location context and shows Chile regional macro context. |

Short fragments such as `5324` return broad address options. Add city/country to narrow the matches.

![Estimator hero](docs/images/estimator-hero.jpg)

## Current App Experience

The web app now opens as a dark glass valuation workbench with a 3D animated background, live loading state, source-status panels, map preview, and an evidence-first estimate card.

- The country control is a real dropdown, so users can click the arrow or use the keyboard to move between supported markets.
- Changing the country clears the previous address, city, region, postal code, property facts, suggestions, map context, and estimate result so stale U.S. data does not remain on a South America search.
- The address field supports autocomplete-style suggestions and accepts short fragments when the user does not know the exact address.
- The right-side panel refreshes into live address intelligence with map context, verified location fields, regional signals, and provider-backed property facts when available.
- Unknown fields are allowed, but the app labels missing data instead of hiding uncertainty.

## Tech Stack

- Python standard-library web server
- scikit-learn regression pipeline
- pandas and joblib for training and model persistence
- Zillow Research ZHVI CSV integration
- U.S. Census Geocoder and optional ACS data
- OpenStreetMap/Nominatim global geocoding
- Optional Geoapify autocomplete and neighborhood enrichment
- Optional ATTOM address-level property facts
- Optional Google Street View and Mapbox imagery
- World Bank public indicators for regional context
- Docker and Render deployment config
- Pytest verification suite

## Core Features

- Country-aware address verification for the United States, Ecuador, Brazil, Peru, Colombia, and Chile.
- Keyboard-friendly country dropdown that resets stale address data when the market changes.
- Live address suggestions, including short-fragment fallback through public geocoding.
- City, state/province, ZIP/postal code, latitude, and longitude enrichment.
- Map preview with map-click reverse geocoding for location refinement.
- U.S. market anchoring through Zillow Research ZIP-level ZHVI.
- Optional U.S. Census ACS median home value signal.
- Optional ATTOM property facts for square footage, beds, baths, lot size, and year built.
- Optional Geoapify neighborhood/suburb enrichment.
- Regional South America context from OpenStreetMap, Mercado Libre search when available, and World Bank indicators.
- Honest estimate policy that avoids fake precision when core property facts are missing.
- Docker-ready deployment with a Render blueprint.

## User Workflow

1. Select a country.
2. Type an address, place name, or short fragment.
3. Choose a verified address suggestion when options appear.
4. Review the live map and source status.
5. Leave unknown fields blank or type `unknown`.
6. Submit the estimate.
7. Review the estimate method, confidence, range, and source notes.

## Data Quality System

The app changes behavior based on data quality:

- If square footage, beds, baths, and lot size are known, the property model can run.
- If those core facts are unknown, the app does not invent a precise model estimate.
- If Zillow or Census data is available, it returns a public-data market baseline.
- If regional data is contextual rather than property-level, it labels it as context.
- If a source cannot provide a field, the UI says that clearly instead of hiding it.

## Data Sources

- **Zillow Research ZHVI**: U.S. ZIP-level typical home value signal. This is not an address-level Zestimate.
- **U.S. Census Geocoder**: U.S. address verification and coordinates.
- **U.S. Census ACS 5-year B25077**: optional median owner-occupied home value by ZIP Code Tabulation Area.
- **OpenStreetMap Nominatim**: global geocoding and reverse geocoding.
- **Geoapify**: optional autocomplete and neighborhood/suburb enrichment.
- **ATTOM**: optional U.S. address-level property records.
- **Mercado Libre public search**: regional listing context when the public endpoint returns data.
- **World Bank Indicators API**: country-level macro context for supported South American markets.
- **Mapbox / Google Street View**: optional visual map or exterior context.

Outside the United States, the app does not claim Zillow-style valuation coverage. It verifies the location, maps the property area, and adds regional context when public APIs return it.

## Screenshots

![Estimate result](docs/images/estimate-result.jpg)

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./verify.sh
```

Run the app:

```bash
PYTHONPATH=src python -m real_estate_price_estimator.web_app --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Optional API Keys

The app works without keys, but optional providers improve enrichment:

```bash
export CENSUS_API_KEY=your_census_key_here
export ATTOM_API_KEY=your_attom_key_here
export GEOAPIFY_API_KEY=your_geoapify_key_here
export GOOGLE_STREET_VIEW_API_KEY=your_google_maps_key_here
export MAPBOX_ACCESS_TOKEN=your_mapbox_token_here
PYTHONPATH=src python -m real_estate_price_estimator.web_app --port 8000
```

## Docker

```bash
docker build -t property-valuation-workbench .
docker run --rm -p 8000:8000 property-valuation-workbench
```

## Free Deployment

This repo includes `render.yaml` and a `Dockerfile` for Render.

Deploy steps:

1. Push this repository to GitHub.
2. In Render, create a new Blueprint from the repository.
3. Select the free web service plan.
4. Add optional secrets: `CENSUS_API_KEY`, `GEOAPIFY_API_KEY`, `ATTOM_API_KEY`, `GOOGLE_STREET_VIEW_API_KEY`, and `MAPBOX_ACCESS_TOKEN`.
5. Deploy. Render will build the Docker image and publish the app on its generated `onrender.com` URL.

## Training

Generate deterministic sample data:

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

## Project Structure

```text
real-estate-price-estimator/
  src/real_estate_price_estimator/
    cli.py
    market_data.py
    pipeline.py
    web_app.py
  scripts/
    generate_sample_data.py
  tests/
  data/
  docs/images/
  static/vendor/
  Dockerfile
  render.yaml
  verify.sh
```

## Tests

```bash
./verify.sh
```

Current verification covers:

- Pipeline training and prediction.
- Unknown/blank property field parsing.
- Zillow Research ZHVI parsing and blending.
- Optional Census ACS blending logic.
- Address lookup and reverse geocoding.
- Country-aware suggestion routes.
- Data-quality-aware estimate decisions.
- Browser route rendering.

## Roadmap

- Add a real production database for cached geocoding and provider responses.
- Add first-class provider adapters for country-specific property record APIs.
- Add comparable listing normalization by country and currency.
- Add confidence scoring by source freshness and geographic precision.
- Add a hosted demo with configured provider keys.
- Add end-to-end browser tests for country examples.

## GitHub Setup

```bash
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/real-estate-price-estimator.git
git add .
git commit -m "Initial property valuation workbench"
git push -u origin main
```

Do not commit real API keys. Use local environment variables or deployment secrets.
