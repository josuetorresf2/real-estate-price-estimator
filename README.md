# Real Estate Price Estimator

A production-style valuation workbench and regression pipeline for estimating home prices from verified addresses, U.S. ZIP-level market signals, optional property-record APIs, regional listing context, and public government/global location data.

## Try These Examples

Select the country in the app, then paste the matching address/search text:

| Country | Try this address/search |
| --- | --- |
| United States | `1701 Wynkoop St Denver CO` |
| Ecuador | `La Carolina Quito` |
| Brazil | `Avenida Paulista Sao Paulo` |
| Peru | `Miraflores Lima` |
| Colombia | `El Poblado Medellin` |
| Chile | `Providencia Santiago` |

Short fragments such as `5324` also return broad address options; add city/country to narrow them.

![Estimator hero](docs/images/estimator-hero.jpg)

## What It Does

- Produces an estimate from real public market signals first, then uses the scikit-learn property model only when enough property facts are known.
- Lets users enter `unknown` or leave most property facts blank.
- Uses model imputation for partial property data, but avoids pretending the model is meaningful when core facts are missing.
- Uses a supplied address to look up city, state/province, ZIP/postal code, and map coordinates through the U.S. Census Geocoder or OpenStreetMap Nominatim for supported South American countries.
- Verifies addresses as users type, shows address candidates, and updates city, state, and ZIP when a match is found.
- Supports country-aware lookup for the United States, Ecuador, Brazil, Peru, Colombia, and Chile.
- Supports richer autocomplete through `GEOAPIFY_API_KEY`, with Census Geocoder as the no-key fallback.
- Optionally fills address-level square feet, bedrooms, bathrooms, lot size, and year built through `ATTOM_API_KEY`.
- Optionally fills neighborhood/suburb from Geoapify when `GEOAPIFY_API_KEY` returns that signal.
- Calculates miles to city center from Census Geocoder coordinates when available.
- Calibrates estimates with Zillow Research ZIP-level ZHVI data when available.
- Optionally blends U.S. Census ACS median home value when `CENSUS_API_KEY` is configured.
- Adds regional context for supported South American countries through OpenStreetMap/Nominatim, Mercado Libre public search when available, and World Bank country indicators.
- Optionally shows real exterior context through Google Street View Static API when `GOOGLE_STREET_VIEW_API_KEY` is configured.
- Shows an Engrain-inspired property map panel with a selected-property marker, market layers, source badges, Mapbox satellite-streets imagery when `MAPBOX_ACCESS_TOKEN` is configured, and OpenStreetMap fallback context.
- Presents the estimate as an evidence-first product workflow: address resolution, source coverage, model decision, map context, and deployment path.

![Estimate result](docs/images/estimate-result.jpg)

## Data Sources

The app is intentionally transparent about source quality:

- **Zillow Research ZHVI**: free ZIP-level typical home value signal. This is not an address-level Zestimate.
- **U.S. Census Geocoder**: free address-to-location lookup for city, state, ZIP, and coordinates.
- **U.S. Census ACS 5-year B25077**: optional government median owner-occupied home value by ZIP Code Tabulation Area. Requires `CENSUS_API_KEY`.
- **ATTOM Property API**: optional address-level property facts such as living area, rooms, lot size, and year built. Requires `ATTOM_API_KEY`.
- **Geoapify Address Autocomplete**: optional real address suggestions for partial user input. Requires `GEOAPIFY_API_KEY`.
- **Google Street View Static API**: optional real exterior/street-level imagery. Requires `GOOGLE_STREET_VIEW_API_KEY`.
- **Mapbox Static Images API**: optional satellite-streets map imagery. Requires `MAPBOX_ACCESS_TOKEN`.
- **OpenStreetMap Nominatim**: global geocoding fallback for Ecuador, Brazil, Peru, Colombia, and Chile.
- **Mercado Libre public search**: optional regional listing context where the public endpoint returns real estate listings.
- **World Bank public indicators**: country-level GDP per capita and urban population context for South American markets.
- **Trained regression model**: used only when core property facts are available; otherwise the app returns a public-data market baseline.
- **OpenStreetMap**: area map preview when the Census Geocoder returns coordinates.
- **Engrain/SightMap-inspired UX**: the app does not use Engrain's private platform, but borrows the product idea of an interactive property map with visual context and data layers.

Public/free Zillow Research and Census data generally do not include exact address-level beds, baths, square footage, lot size, or year built. Those fields can be entered when known, or auto-filled from ATTOM when an API key is configured. The app displays field-level source notes so users can see which facts were verified and which were unavailable.

Outside the United States, the app does not claim Zillow-style valuation coverage. It verifies the address, maps the property area, adds regional listing or macro context when public APIs return it, and clearly labels those signals as contextual rather than appraised values.

The app deliberately changes behavior based on data quality:

- If square footage, beds, baths, and lot size are known, it blends the property model with public market signals.
- If those core property facts are unknown, it does **not** run a fake precise model estimate. It returns a ZIP or Census market baseline with a wider range.
- If no public market anchor is found, it asks for a usable U.S. address or ZIP instead of inventing a number.

Source references:

- [Zillow Research Data](https://www.zillow.com/research/data/)
- [Zillow Public Real Estate Metrics API](https://www.zillowgroup.com/developers/api/public-data/real-estate-metrics/)
- [U.S. Census Geocoder](https://geocoding.geo.census.gov/)
- [U.S. Census ACS 5-Year API](https://www.census.gov/data/developers/data-sets/acs-5year.html)
- [ATTOM Property API](https://api.developer.attomdata.com/)
- [Geoapify Address Autocomplete API](https://apidocs.geoapify.com/docs/geocoding/address-autocomplete/)
- [Google Street View Static API](https://developers.google.com/maps/documentation/streetview)
- [Mapbox Static Images API](https://docs.mapbox.com/api/maps/static-images/)
- [OpenStreetMap](https://www.openstreetmap.org/)
- [Nominatim](https://nominatim.org/)
- [Nominatim Usage Policy](https://operations.osmfoundation.org/policies/nominatim/)
- [Mercado Libre Developers](https://developers.mercadolibre.com/)
- [World Bank API](https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation)

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

## Country Data Behavior

Use the `Country` field first, then type or paste one of these examples into `Address`. Short fragments such as `5324` will show broad options; adding a city or country narrows the results.

| Country | Example address/search | Expected data behavior |
| --- | --- | --- |
| United States | `1701 Wynkoop St Denver CO` | Census verifies the address, Zillow ZIP data can anchor the estimate, and optional ATTOM can fill property facts. |
| Ecuador | `La Carolina Quito` | OpenStreetMap/Nominatim verifies the area and World Bank context is shown for Ecuador. |
| Brazil | `Avenida Paulista Sao Paulo` | OpenStreetMap/Nominatim verifies the location and regional macro context is shown for Brazil. |
| Peru | `Miraflores Lima` | OpenStreetMap/Nominatim verifies the district/city and World Bank context is shown for Peru. |
| Colombia | `El Poblado Medellin` | OpenStreetMap/Nominatim verifies the location and World Bank context is shown for Colombia. |
| Chile | `Providencia Santiago` | OpenStreetMap/Nominatim verifies the location and World Bank context is shown for Chile. |

For South American markets, the app shows location, map, listing/macro context, and source availability. It does not claim to produce a Zillow-style address-level valuation unless country-specific property records or comparable sales are connected.

## Optional Census API

The U.S. Census ACS API can add a government median-home-value signal. Set this only if you have a Census API key:

```bash
export CENSUS_API_KEY=your_key_here
PYTHONPATH=src python -m real_estate_price_estimator.web_app --port 8000
```

Without a key, the app still uses the Census Geocoder and Zillow Research ZHVI.

## Optional Enrichment APIs

Address-level facts and imagery are provider-backed because free Zillow Research and Census endpoints do not include them:

```bash
export ATTOM_API_KEY=your_attom_key_here
export GEOAPIFY_API_KEY=your_geoapify_key_here
export GOOGLE_STREET_VIEW_API_KEY=your_google_maps_key_here
export MAPBOX_ACCESS_TOKEN=your_mapbox_token_here
PYTHONPATH=src python -m real_estate_price_estimator.web_app --port 8000
```

When `GEOAPIFY_API_KEY` is set, partial address input can return real autocomplete suggestions and may fill neighborhood/suburb when Geoapify returns that field. When `ATTOM_API_KEY` is set, the live address lookup can fill square feet, bedrooms, bathrooms, lot size, and year built from property records. When `GOOGLE_STREET_VIEW_API_KEY` is set, result pages can show real Street View exterior context for the verified address. When `MAPBOX_ACCESS_TOKEN` is set, live and result maps use Mapbox satellite-streets imagery; otherwise the app falls back to OpenStreetMap.

## Docker

Build and run locally:

```bash
docker build -t real-estate-price-estimator .
docker run --rm -p 8000:8000 real-estate-price-estimator
```

Open:

```text
http://127.0.0.1:8000
```

## Free Deployment

This repo includes `render.yaml` and a `Dockerfile` for Render. Render web services receive a free `*.onrender.com` subdomain, and the service is configured to bind to `0.0.0.0` through the `PORT` environment variable.

Deploy steps:

1. Push this repository to GitHub.
2. In Render, create a new Blueprint from the repository.
3. Select the free web service plan.
4. Add optional secret environment variables: `CENSUS_API_KEY`, `GEOAPIFY_API_KEY`, `ATTOM_API_KEY`, `GOOGLE_STREET_VIEW_API_KEY`, and `MAPBOX_ACCESS_TOKEN`.
5. Deploy. Render will build the Docker image and publish the app on its generated `onrender.com` URL.

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
- Data-quality-aware estimate decisions.
- Realtime address verification through the local `/api/geocode` endpoint.
- Browser route rendering.

## Production Notes

This is not an appraisal product. It is a market-aware estimator. Production use should retrain on current MLS, assessor, parcel, tax, school, and neighborhood data, and should use an approved property-record API if exact address-level facts are required.
