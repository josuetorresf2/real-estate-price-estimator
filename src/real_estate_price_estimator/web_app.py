from __future__ import annotations

import argparse
import html
import json
import mimetypes
import os
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

from .market_data import (
    census_home_value_for_zip,
    distance_to_city_center_miles,
    ensure_zhvi_csv,
    geocode_address,
    geocode_address_matches,
    geoapify_address_suggestions,
    latest_zhvi_for_zip,
    market_calibrated_estimate,
    property_facts_for_address,
)
from .pipeline import load_model, load_training_data, predict_price, save_model, train

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "price_pipeline.joblib"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sample_housing.csv"
DEFAULT_ZHVI_PATH = PROJECT_ROOT / "data" / "zillow_zhvi_zip.csv"
STATIC_ROOT = PROJECT_ROOT / "static"

DEFAULT_FORM_VALUES = {
    "address": "",
    "city": "",
    "state": "",
    "neighborhood": "",
    "zip_code": "78751",
    "square_feet": "",
    "bedrooms": "",
    "bathrooms": "",
    "lot_size": "",
    "year_built": "",
    "school_rating": "",
    "distance_to_city_center_miles": "",
    "crime_index": "",
}

FIELD_LABELS = {
    "address": "Address",
    "city": "City",
    "state": "State",
    "neighborhood": "Neighborhood",
    "zip_code": "ZIP code",
    "square_feet": "Square feet",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "lot_size": "Lot size (acres)",
    "year_built": "Year built (optional)",
    "school_rating": "School rating",
    "distance_to_city_center_miles": "Miles to city center",
    "crime_index": "Crime index",
}

NUMERIC_FIELDS = {
    "square_feet": float,
    "bedrooms": float,
    "bathrooms": float,
    "lot_size": float,
    "year_built": int,
    "school_rating": float,
    "distance_to_city_center_miles": float,
    "crime_index": float,
}
MODEL_REQUIRED_FACTS = {"square_feet", "bedrooms", "bathrooms", "lot_size"}
ADDRESS_ENRICHMENT_FIELDS = {
    "neighborhood",
    "square_feet",
    "bedrooms",
    "bathrooms",
    "lot_size",
    "year_built",
    "distance_to_city_center_miles",
}
FEATURE_FIELDS = {"city", "neighborhood", "zip_code", *NUMERIC_FIELDS}
OPTIONAL_FIELDS = {
    "address",
    "city",
    "state",
    "neighborhood",
    "zip_code",
    "square_feet",
    "bedrooms",
    "bathrooms",
    "lot_size",
    "year_built",
    "school_rating",
    "distance_to_city_center_miles",
    "crime_index",
}


def ensure_model(model_path: Path = DEFAULT_MODEL_PATH, data_path: Path = DEFAULT_DATA_PATH):
    if model_path.exists():
        return load_model(model_path)
    data = load_training_data(data_path)
    model, _ = train(data)
    save_model(model, model_path)
    return model


def parse_form(body: str) -> tuple[dict[str, object], dict[str, str], list[str]]:
    parsed = parse_qs(body, keep_blank_values=True)
    values = {key: parsed.get(key, [""])[0].strip() for key in DEFAULT_FORM_VALUES}
    errors = []
    features: dict[str, object] = {}

    for field, value in values.items():
        if field in {"address", "state"}:
            continue
        if field in OPTIONAL_FIELDS and value.lower() in {"", "unknown", "unkown", "uknown", "n/a", "na"}:
            features[field] = None if field in NUMERIC_FIELDS else ""
            continue
        if not value:
            errors.append(f"{FIELD_LABELS[field]} is required.")
            continue
        if field in NUMERIC_FIELDS:
            try:
                features[field] = NUMERIC_FIELDS[field](value)
            except ValueError:
                errors.append(f"{FIELD_LABELS[field]} must be a number.")
        elif field in FEATURE_FIELDS:
            features[field] = value

    return features, values, errors


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


@dataclass(frozen=True)
class EstimateDecision:
    estimate: float | None
    low: float | None
    high: float | None
    method: str
    confidence: str
    known_fact_count: int
    used_model: bool


def apply_address_location(values: dict[str, str], features: dict[str, object], location) -> None:
    if location is None:
        return
    if not values.get("city") and location.city:
        values["city"] = location.city
        features["city"] = location.city
    if not values.get("state") and location.state:
        values["state"] = location.state
    if not values.get("zip_code") and location.zip_code:
        values["zip_code"] = location.zip_code
        features["zip_code"] = location.zip_code


def apply_property_facts(values: dict[str, str], features: dict[str, object], facts) -> None:
    if facts is None:
        return
    for field, value in facts.as_form_values().items():
        if values.get(field, "").strip().lower() not in {"", "unknown", "unkown", "uknown", "n/a", "na"}:
            continue
        values[field] = value
        if field in NUMERIC_FIELDS:
            features[field] = NUMERIC_FIELDS[field](value)


def finalize_prediction_features(features: dict[str, object]) -> dict[str, object]:
    finalized = features.copy()
    for field in ("city", "neighborhood", "zip_code"):
        if not finalized.get(field):
            finalized[field] = "Unknown"
    for field in NUMERIC_FIELDS:
        finalized.setdefault(field, None)
    return finalized


def known_property_fact_count(features: dict[str, object]) -> int:
    return sum(1 for field in NUMERIC_FIELDS if features.get(field) is not None)


def should_use_model(features: dict[str, object]) -> bool:
    return all(features.get(field) is not None for field in MODEL_REQUIRED_FACTS)


def decide_estimate(
    *,
    model_prediction: float | None,
    market_signal: object | None,
    census_signal: object | None,
    known_fact_count: int,
) -> EstimateDecision:
    has_zillow = market_signal is not None
    has_census = census_signal is not None

    if model_prediction is not None:
        estimate = market_calibrated_estimate(model_prediction, market_signal, census_signal)
        band = 0.08 if has_zillow else 0.14
        confidence = "High" if has_zillow and known_fact_count >= 6 else "Medium"
        method = "Property model calibrated with public market data" if has_zillow or has_census else "Property model only"
        return EstimateDecision(
            estimate=estimate,
            low=estimate * (1 - band),
            high=estimate * (1 + band),
            method=method,
            confidence=confidence,
            known_fact_count=known_fact_count,
            used_model=True,
        )

    if has_zillow and has_census:
        estimate = market_signal.typical_home_value * 0.75 + census_signal.median_home_value * 0.25
        return EstimateDecision(
            estimate=estimate,
            low=estimate * 0.86,
            high=estimate * 1.14,
            method="Public-data market baseline: Zillow ZHVI plus Census ACS",
            confidence="Medium",
            known_fact_count=known_fact_count,
            used_model=False,
        )

    if has_zillow:
        estimate = market_signal.typical_home_value
        return EstimateDecision(
            estimate=estimate,
            low=estimate * 0.84,
            high=estimate * 1.16,
            method="Public-data market baseline: Zillow Research ZHVI",
            confidence="Medium",
            known_fact_count=known_fact_count,
            used_model=False,
        )

    if has_census:
        estimate = census_signal.median_home_value
        return EstimateDecision(
            estimate=estimate,
            low=estimate * 0.78,
            high=estimate * 1.22,
            method="Government market baseline: U.S. Census ACS",
            confidence="Low",
            known_fact_count=known_fact_count,
            used_model=False,
        )

    return EstimateDecision(
        estimate=None,
        low=None,
        high=None,
        method="No public market signal found",
        confidence="Unavailable",
        known_fact_count=known_fact_count,
        used_model=False,
    )


def map_preview(
    address_location: object | None,
    *,
    market_signal: object | None = None,
    census_signal: object | None = None,
    decision: EstimateDecision | None = None,
) -> str:
    if address_location is None or address_location.latitude is None or address_location.longitude is None:
        return ""
    lat = address_location.latitude
    lon = address_location.longitude
    marker = f"{lat},{lon},pm2rdm"
    bbox = f"{lon - 0.018},{lat - 0.012},{lon + 0.018},{lat + 0.012}"
    map_url = f"https://www.openstreetmap.org/export/embed.html?bbox={bbox}&layer=mapnik&marker={marker}"
    image_url = mapbox_static_url(address_location)
    map_media = (
        f'<img src="{html.escape(image_url)}" alt="Satellite streets map near the verified address" loading="lazy">'
        if image_url
        else f'<iframe title="Mapped address area" src="{html.escape(map_url)}" loading="lazy"></iframe>'
    )
    source_items = ["Census geocoded address", "OpenStreetMap context"]
    if market_signal is not None:
        source_items.append("Zillow ZIP value layer")
    if census_signal is not None:
        source_items.append("Census ACS value layer")
    layers = "".join(f"<li>{html.escape(item)}</li>" for item in source_items)
    range_text = "Run estimate"
    if decision is not None and decision.low is not None and decision.high is not None:
        range_text = f"{format_currency(decision.low)} - {format_currency(decision.high)}"
    zillow_text = "No ZIP value loaded"
    if market_signal is not None:
        zillow_text = f"{format_currency(market_signal.typical_home_value)} ZIP typical value"
    census_text = "Optional ACS value"
    if census_signal is not None:
        census_text = f"{format_currency(census_signal.median_home_value)} ACS median value"
    return f"""
    <section class="property-map" aria-label="Property map intelligence">
      <div class="map-stage">
        {map_media}
        <div class="map-pin" aria-hidden="true"><span></span></div>
        <div class="map-radar" aria-hidden="true"></div>
      </div>
      <div class="map-dossier">
        <div>
          <span>Selected property</span>
          <strong>{html.escape(address_location.matched_address)}</strong>
        </div>
        <div class="map-metrics">
          <div><span>Estimate range</span><strong>{range_text}</strong></div>
          <div><span>Market layer</span><strong>{zillow_text}</strong></div>
          <div><span>Gov layer</span><strong>{census_text}</strong></div>
        </div>
        <ul>{layers}</ul>
      </div>
    </section>
    """


def street_view_preview(address_location: object | None) -> str:
    image_url = street_view_url(address_location)
    if not image_url:
        return ""
    return f"""
    <figure class="house-preview">
      <img src="{html.escape(image_url)}" alt="Street View image near the verified address" loading="lazy">
      <figcaption>Real exterior context from Google Street View Static API when imagery is available.</figcaption>
    </figure>
    """


def mapbox_static_url(address_location: object | None) -> str:
    token = os.getenv("MAPBOX_ACCESS_TOKEN")
    if not token or address_location is None or address_location.latitude is None or address_location.longitude is None:
        return ""
    lat = address_location.latitude
    lon = address_location.longitude
    marker = f"pin-s+be4c00({lon},{lat})"
    query = urlencode({"access_token": token, "attribution": "true", "logo": "true"})
    return f"https://api.mapbox.com/styles/v1/mapbox/satellite-streets-v12/static/{marker}/{lon},{lat},16,0,55/640x360@2x?{query}"


def street_view_url(address_location: object | None) -> str:
    key = os.getenv("GOOGLE_STREET_VIEW_API_KEY")
    if not key or address_location is None:
        return ""
    if address_location.latitude is not None and address_location.longitude is not None:
        location = f"{address_location.latitude},{address_location.longitude}"
    else:
        location = getattr(address_location, "matched_address", "")
    if not location:
        return ""
    query = urlencode(
        {
            "size": "640x360",
            "location": location,
            "fov": "82",
            "pitch": "4",
            "source": "outdoor",
            "key": key,
        }
    )
    return f"https://maps.googleapis.com/maps/api/streetview?{query}"


def property_fact_payload(facts, distance_miles: float | None) -> dict[str, object]:
    fields: dict[str, dict[str, str]] = {}
    if facts is not None:
        for field, value in facts.as_form_values().items():
            fields[field] = {"value": value, "source": facts.source}
    if distance_miles is not None:
        fields["distance_to_city_center_miles"] = {
            "value": f"{distance_miles:.1f}",
            "source": "Calculated from U.S. Census Geocoder coordinates",
        }
    missing = {
        "neighborhood": "Neighborhood is not returned by Zillow Research or Census Geocoder.",
        "square_feet": "Address-level square footage requires a property-record provider such as ATTOM.",
        "bedrooms": "Address-level bedrooms require a property-record provider such as ATTOM.",
        "bathrooms": "Address-level bathrooms require a property-record provider such as ATTOM.",
        "lot_size": "Address-level lot size requires a property-record provider such as ATTOM.",
        "year_built": "Address-level year built requires a property-record provider such as ATTOM.",
        "school_rating": "School rating is not provided by Zillow Research or Census Geocoder.",
        "crime_index": "Crime index is not provided by Zillow Research or Census Geocoder.",
    }
    for field in fields:
        missing.pop(field, None)
    return {
        "fields": fields,
        "missing": missing,
        "provider_configured": bool(os.getenv("ATTOM_API_KEY")),
    }


def location_payload(location) -> dict[str, str]:
    return {
        "city": location.city,
        "state": location.state,
        "zip_code": location.zip_code,
        "matched_address": location.matched_address,
        "latitude": "" if location.latitude is None else str(location.latitude),
        "longitude": "" if location.longitude is None else str(location.longitude),
        "source": location.source,
    }


def enriched_address_query(address: str, *, city: str = "", state: str = "", zip_code: str = "") -> str:
    parts = [address.strip()]
    for value in (city, state, zip_code):
        value = value.strip()
        if value and value.lower() not in address.lower():
            parts.append(value)
    return " ".join(parts).strip()


def render_page(
    values: dict[str, str] | None = None,
    *,
    prediction: float | None = None,
    model_prediction: float | None = None,
    market_signal: object | None = None,
    census_signal: object | None = None,
    address_location: object | None = None,
    decision: EstimateDecision | None = None,
    errors: list[str] | None = None,
) -> str:
    form_values = values or DEFAULT_FORM_VALUES
    error_items = "".join(f"<li>{html.escape(error)}</li>" for error in errors or [])
    result = ""
    if prediction is not None:
        decision_note = ""
        if decision is not None:
            decision_note = f"""
            <dl class="market-card">
              <div><dt>Method</dt><dd>{html.escape(decision.method)}</dd></div>
              <div><dt>Confidence</dt><dd>{html.escape(decision.confidence)}</dd></div>
              <div><dt>Estimated range</dt><dd>{format_currency(decision.low)} - {format_currency(decision.high)}</dd></div>
              <div><dt>Known property facts</dt><dd>{decision.known_fact_count} of {len(NUMERIC_FIELDS)}</dd></div>
            </dl>
            """
        market_note = ""
        if market_signal is not None:
            market_note = f"""
            <dl class="market-card">
              <div><dt>Zillow ZIP signal</dt><dd>{format_currency(market_signal.typical_home_value)}</dd></div>
              <div><dt>Market</dt><dd>{html.escape(market_signal.city)}, {html.escape(market_signal.state)} {html.escape(market_signal.zip_code)}</dd></div>
              <div><dt>Data month</dt><dd>{html.escape(market_signal.date)}</dd></div>
            </dl>
            <p class="source-note">Market signal: Zillow Research ZHVI. This is ZIP-level typical home value data, not an address-level Zestimate.</p>
            """
        if census_signal is not None:
            market_note += f"""
            <dl class="market-card">
              <div><dt>Government signal</dt><dd>{format_currency(census_signal.median_home_value)}</dd></div>
              <div><dt>Geography</dt><dd>{html.escape(census_signal.name)}</dd></div>
              <div><dt>Dataset</dt><dd>ACS {html.escape(census_signal.year)} 5-year</dd></div>
            </dl>
            <p class="source-note">Government signal: U.S. Census ACS B25077 median owner-occupied home value. Requires CENSUS_API_KEY when enabled.</p>
            """
        if address_location is not None:
            market_note += f"""
            <p class="source-note">Address lookup: {html.escape(address_location.matched_address)} via {html.escape(address_location.source)}.</p>
            """
        if not market_note and model_prediction is not None:
            market_note = """
            <p class="source-note">No Zillow or Census pricing signal was found for this ZIP, so this result uses the trained feature model only.</p>
            """
        result = f"""
        <section class="result" aria-live="polite">
          <span>Market-calibrated estimate</span>
          <strong>{format_currency(prediction)}</strong>
          {street_view_preview(address_location)}
          {map_preview(address_location, market_signal=market_signal, census_signal=census_signal, decision=decision)}
          {decision_note}
          {f'<p class="model-note">Feature model: {format_currency(model_prediction)}</p>' if model_prediction is not None else ''}
          {market_note}
        </section>
        """
    else:
        result = """
        <section class="empty-result" aria-live="polite">
          <strong>Live market estimate</strong>
          <p class="note">Enter an address or ZIP. Unknown property facts are allowed and handled by model imputation when public data does not include them.</p>
          <div class="live-context" data-live-context>
            <span>Address intelligence</span>
            <p>Verify an address to load the property map, ZIP, city/state, distance, and data-source status.</p>
          </div>
        </section>
        """

    fields = "\n".join(
        f"""
        <label class="{field}">
          <span>{html.escape(label)}</span>
          <input name="{field}" value="{html.escape(form_values.get(field, ''))}" {input_attributes(field)}>
          {field_help(field)}
          {address_status(field)}
          {address_suggestions(field)}
          {field_status(field)}
        </label>
        """
        for field, label in FIELD_LABELS.items()
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Property Valuation Workbench</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Kanit:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet">
  <style>
    :root {{
      color-scheme: dark;
      --ink: #d7e2ea;
      --muted: rgba(215, 226, 234, 0.64);
      --soft: rgba(215, 226, 234, 0.1);
      --line: rgba(215, 226, 234, 0.16);
      --field: rgba(255, 255, 255, 0.08);
      --accent: #b600a8;
      --accent-strong: #be4c00;
      --warn: #ffb36b;
      --bg: #0c0c0c;
      --panel: rgba(12, 12, 12, 0.72);
      --shadow: rgba(0, 0, 0, 0.46);
    }}
    * {{ box-sizing: border-box; }}
    html {{ min-height: 100%; background: var(--bg); }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Kanit", ui-sans-serif, system-ui, sans-serif;
      color: var(--ink);
      background:
        linear-gradient(123deg, rgba(24, 1, 31, 0.86) 7%, rgba(182, 0, 168, 0.22) 37%, rgba(118, 33, 176, 0.18) 72%, rgba(190, 76, 0, 0.14) 100%),
        #0c0c0c;
      overflow-x: hidden;
    }}
    body.loading main {{
      filter: blur(10px);
      opacity: 0.38;
    }}
    .loader {{
      position: fixed;
      inset: 0;
      z-index: 10;
      display: grid;
      place-items: center;
      background: rgba(12, 12, 12, 0.82);
      transition: opacity 450ms ease, visibility 450ms ease;
    }}
    body:not(.loading) .loader {{
      opacity: 0;
      visibility: hidden;
      pointer-events: none;
    }}
    .loader-orbit {{
      width: 190px;
      height: 190px;
      border: 1px solid rgba(215, 226, 234, 0.18);
      border-radius: 50%;
      display: grid;
      place-items: center;
      position: relative;
      animation: spin 1.8s linear infinite;
    }}
    .loader-orbit::before,
    .loader-orbit::after {{
      content: "";
      position: absolute;
      inset: 28px;
      border-radius: 50%;
      border: 1px solid rgba(182, 0, 168, 0.52);
      transform: rotate(48deg) scaleY(0.56);
    }}
    .loader-orbit::after {{
      border-color: rgba(190, 76, 0, 0.7);
      transform: rotate(-32deg) scaleY(0.48);
    }}
    .loader-core {{
      width: 56px;
      height: 56px;
      border-radius: 50%;
      background: linear-gradient(123deg, #18011f 7%, #b600a8 37%, #7621b0 72%, #be4c00 100%);
      box-shadow: 0 0 46px rgba(182, 0, 168, 0.7);
    }}
    @keyframes spin {{
      to {{ transform: rotate(360deg); }}
    }}
    #cityScene {{
      position: fixed;
      inset: 0;
      width: 100vw;
      height: 100vh;
      display: block;
      z-index: 0;
    }}
    .scene-vignette {{
      position: fixed;
      inset: 0;
      z-index: 1;
      pointer-events: none;
      background:
        linear-gradient(90deg, rgba(12, 12, 12, 0.92), rgba(12, 12, 12, 0.28) 48%, rgba(12, 12, 12, 0.74)),
        linear-gradient(180deg, rgba(12, 12, 12, 0.08), rgba(12, 12, 12, 0.92));
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      pointer-events: none;
      background-image:
        linear-gradient(rgba(255, 255, 255, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255, 255, 255, 0.028) 1px, transparent 1px);
      background-size: 72px 72px;
      mask-image: linear-gradient(180deg, black, transparent 78%);
      z-index: 2;
    }}
    .hero-heading {{
      color: #f7fafc;
    }}
    main {{
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      min-height: 100vh;
      padding: 34px 0;
      position: relative;
      z-index: 3;
      display: grid;
      align-content: center;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 24px;
      margin-bottom: 20px;
    }}
    nav {{
      display: flex;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 28px;
      color: var(--ink);
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.12em;
    }}
    nav a {{
      color: inherit;
      text-decoration: none;
      transition: opacity 200ms ease;
    }}
    nav a:hover {{ opacity: 0.7; }}
    .eyebrow {{
      display: inline-flex;
      align-items: center;
      min-height: 28px;
      margin-bottom: 14px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: rgba(255, 255, 255, 0.07);
      color: var(--muted);
      padding: 0 12px;
      font-size: 0.82rem;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.12em;
      backdrop-filter: blur(18px) saturate(150%);
    }}
    h1 {{
      margin: 0;
      font-size: clamp(3rem, 10vw, 8.9rem);
      line-height: 0.82;
      letter-spacing: 0;
      max-width: 940px;
      text-wrap: balance;
      text-transform: uppercase;
      font-weight: 900;
    }}
    header p {{
      margin: 0;
      max-width: 390px;
      color: var(--muted);
      line-height: 1.45;
      font-size: 1rem;
    }}
    .signal-strip {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 16px;
    }}
    .signal-strip div {{
      border: 1px solid rgba(215, 226, 234, 0.13);
      border-radius: 8px;
      background: rgba(11, 18, 32, 0.76);
      padding: 12px;
    }}
    .signal-strip span {{
      display: block;
      color: var(--muted);
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .signal-strip strong {{
      display: block;
      margin-top: 4px;
      color: white;
      font-size: 0.94rem;
      line-height: 1.2;
    }}
    form {{
      background: transparent;
    }}
    .surface {{
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(330px, 0.42fr);
      gap: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background:
        linear-gradient(180deg, rgba(215, 226, 234, 0.1), rgba(215, 226, 234, 0.035)),
        var(--panel);
      padding: 12px;
      box-shadow: 0 36px 120px var(--shadow);
      backdrop-filter: blur(32px) saturate(145%);
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      border-radius: 8px;
      background:
        linear-gradient(180deg, rgba(215, 226, 234, 0.085), rgba(215, 226, 234, 0.035));
      padding: 12px;
    }}
    label.address {{
      grid-column: span 3;
    }}
    label {{
      display: grid;
      gap: 8px;
      min-width: 0;
      border: 1px solid rgba(215, 226, 234, 0.11);
      border-radius: 8px;
      background: rgba(215, 226, 234, 0.055);
      padding: 12px;
    }}
    label:focus-within {{
      border-color: rgba(182, 0, 168, 0.58);
      background: rgba(215, 226, 234, 0.085);
    }}
    label span {{
      color: var(--muted);
      font-size: 0.76rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    input {{
      width: 100%;
      min-height: 34px;
      border: 0;
      border-bottom: 1px solid rgba(215, 226, 234, 0.22);
      border-radius: 0;
      background: transparent;
      color: var(--ink);
      padding: 2px 0 8px;
      font: inherit;
      font-size: 1.05rem;
      font-weight: 650;
    }}
    input:focus {{
      outline: 0;
      border-color: var(--accent);
    }}
    .actions {{
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      gap: 18px;
      border-radius: 8px;
      background:
        linear-gradient(180deg, rgba(215, 226, 234, 0.12), transparent 38%),
        linear-gradient(145deg, rgba(24, 1, 31, 0.86), rgba(12, 12, 12, 0.98));
      color: white;
      padding: 22px;
      min-height: 100%;
      border: 1px solid rgba(215, 226, 234, 0.12);
      position: relative;
      overflow: hidden;
    }}
    .actions::before {{
      content: "";
      position: absolute;
      inset: 14px 14px auto auto;
      width: 96px;
      height: 96px;
      border-top: 1px solid rgba(190, 76, 0, 0.58);
      border-right: 1px solid rgba(182, 0, 168, 0.42);
    }}
    .actions > * {{
      position: relative;
    }}
    .scene {{
      display: none;
    }}
    button {{
      min-height: 46px;
      border: 0;
      border-radius: 8px;
      background: linear-gradient(123deg, #18011f 7%, #b600a8 37%, #7621b0 72%, #be4c00 100%);
      color: white;
      padding: 0 24px;
      font: inherit;
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      cursor: pointer;
      outline: 2px solid white;
      outline-offset: -3px;
      box-shadow: 0 4px 4px rgba(181, 1, 167, 0.25), 4px 4px 12px #7721b1 inset;
    }}
    button:hover {{ filter: brightness(1.06); }}
    .result {{
      display: grid;
      gap: 12px;
      align-content: center;
      text-align: left;
      min-height: 300px;
    }}
    .market-card {{
      display: grid;
      gap: 10px;
      margin: 8px 0 0;
    }}
    .market-card div {{
      border-top: 1px solid rgba(215, 226, 234, 0.12);
      padding-top: 10px;
    }}
    .market-card dt,
    .source-note,
    .model-note {{
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.35;
    }}
    .market-card dd {{
      margin: 3px 0 0;
      font-weight: 700;
      color: white;
    }}
    .property-map {{
      display: grid;
      gap: 10px;
      margin: 8px 0 12px;
      border: 1px solid rgba(215, 226, 234, 0.14);
      border-radius: 8px;
      overflow: hidden;
      background: rgba(215, 226, 234, 0.055);
    }}
    .map-stage {{
      position: relative;
      min-height: 230px;
      overflow: hidden;
      background: #101216;
    }}
    .house-preview {{
      display: grid;
      gap: 8px;
      margin: 4px 0 8px;
    }}
    .house-preview img {{
      width: 100%;
      aspect-ratio: 16 / 9;
      object-fit: cover;
      border: 1px solid rgba(215, 226, 234, 0.14);
      border-radius: 8px;
      filter: contrast(1.04) saturate(0.92);
    }}
    .house-preview figcaption {{
      color: var(--muted);
      font-size: 0.75rem;
      line-height: 1.25;
    }}
    .map-stage iframe,
    .map-stage img {{
      width: 100%;
      height: 230px;
      border: 0;
      object-fit: cover;
      filter: grayscale(0.25) contrast(1.05) brightness(0.88);
    }}
    .map-pin {{
      position: absolute;
      left: 50%;
      top: 50%;
      width: 30px;
      height: 30px;
      transform: translate(-50%, -86%);
      border-radius: 50% 50% 50% 0;
      background: linear-gradient(123deg, #b600a8, #be4c00);
      rotate: -45deg;
      box-shadow: 0 0 34px rgba(182, 0, 168, 0.72);
      pointer-events: none;
    }}
    .map-pin span {{
      position: absolute;
      inset: 8px;
      border-radius: 50%;
      background: white;
    }}
    .map-radar {{
      position: absolute;
      left: 50%;
      top: 50%;
      width: 180px;
      height: 180px;
      transform: translate(-50%, -50%);
      border: 1px solid rgba(182, 0, 168, 0.55);
      border-radius: 50%;
      box-shadow: 0 0 0 34px rgba(182, 0, 168, 0.08), 0 0 0 74px rgba(190, 76, 0, 0.05);
      pointer-events: none;
    }}
    .map-dossier {{
      display: grid;
      gap: 10px;
      padding: 12px;
      background:
        linear-gradient(180deg, rgba(215, 226, 234, 0.1), rgba(215, 226, 234, 0.035)),
        rgba(12, 12, 12, 0.72);
    }}
    .map-dossier span {{
      color: var(--muted);
      font-size: 0.72rem;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .map-dossier strong {{
      display: block;
      margin-top: 3px;
      color: white;
      font-size: 0.9rem;
      line-height: 1.2;
    }}
    .map-metrics {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 8px;
    }}
    .map-metrics div {{
      border-top: 1px solid rgba(215, 226, 234, 0.12);
      padding-top: 8px;
    }}
    .map-dossier ul {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .map-dossier li {{
      border: 1px solid rgba(215, 226, 234, 0.13);
      border-radius: 999px;
      padding: 4px 8px;
      color: var(--muted);
      font-size: 0.7rem;
    }}
    .field-help,
    .field-status {{
      color: var(--muted);
      font-size: 0.75rem;
      line-height: 1.25;
    }}
    .field-status[data-state="filled"] {{
      color: #b8f7c5;
    }}
    .field-status[data-state="missing"] {{
      color: var(--warn);
    }}
    .address-status {{
      min-height: 1rem;
      color: var(--muted);
      font-size: 0.75rem;
      line-height: 1.25;
    }}
    .address-status[data-state="matched"] {{
      color: #b8f7c5;
    }}
    .address-status[data-state="error"] {{
      color: var(--warn);
    }}
    .suggestions {{
      display: none;
      gap: 6px;
      margin: 0;
      padding: 0;
      list-style: none;
    }}
    .suggestions[data-open="true"] {{
      display: grid;
    }}
    .suggestions button {{
      min-height: 34px;
      width: 100%;
      padding: 8px 10px;
      border: 1px solid rgba(215, 226, 234, 0.14);
      outline: 0;
      background: rgba(215, 226, 234, 0.07);
      box-shadow: none;
      color: var(--ink);
      text-align: left;
      text-transform: none;
      letter-spacing: 0;
      font-size: 0.78rem;
      line-height: 1.2;
    }}
    .easteregg {{
      position: fixed;
      right: 7px;
      bottom: 5px;
      z-index: 4;
      color: rgba(215, 226, 234, 0.16);
      font-size: 0.55rem;
      letter-spacing: 0;
      user-select: none;
    }}
    .result span {{
      color: var(--muted);
      font-weight: 700;
      font-size: 0.9rem;
    }}
    .result strong {{
      color: white;
      font-size: clamp(2.6rem, 5vw, 4.6rem);
      line-height: 0.88;
      letter-spacing: 0;
      font-weight: 900;
    }}
    .empty-result {{
      display: grid;
      gap: 10px;
      align-content: center;
      min-height: 260px;
      color: var(--muted);
    }}
    .empty-result strong {{
      color: white;
      font-size: clamp(2rem, 3vw, 3rem);
      line-height: 1.1;
      font-weight: 900;
      text-transform: uppercase;
    }}
    .live-context {{
      display: grid;
      gap: 8px;
      border: 1px solid rgba(215, 226, 234, 0.13);
      border-radius: 8px;
      background: rgba(215, 226, 234, 0.055);
      padding: 12px;
      min-height: 120px;
    }}
    .live-context span {{
      color: var(--muted);
      font-size: 0.75rem;
      font-weight: 700;
      text-transform: uppercase;
    }}
    .live-context p {{
      margin: 0;
      color: var(--muted);
      font-size: 0.82rem;
      line-height: 1.35;
    }}
    .live-map {{
      display: grid;
      gap: 8px;
    }}
    .live-map iframe,
    .live-map img {{
      width: 100%;
      height: 190px;
      border: 1px solid rgba(215, 226, 234, 0.14);
      border-radius: 8px;
      object-fit: cover;
      filter: grayscale(0.25) contrast(1.05) brightness(0.88);
    }}
    .live-map dl {{
      display: grid;
      gap: 7px;
      margin: 0;
    }}
    .live-map div {{
      border-top: 1px solid rgba(215, 226, 234, 0.1);
      padding-top: 7px;
    }}
    .live-map dt {{
      color: var(--muted);
      font-size: 0.72rem;
      text-transform: uppercase;
    }}
    .live-map dd {{
      margin: 2px 0 0;
      color: white;
      font-size: 0.86rem;
      font-weight: 700;
      line-height: 1.25;
    }}
    .errors {{
      margin: 0 0 10px;
      border: 1px solid #fed7aa;
      border-radius: 8px;
      background: #fff7ed;
      color: var(--warn);
      padding: 12px 16px;
    }}
    .note {{
      margin: 0;
      color: var(--muted);
      font-size: 0.9rem;
      line-height: 1.4;
    }}
    .scene {{
      height: 150px;
      margin-bottom: 12px;
      border: 1px solid rgba(215, 226, 234, 0.12);
      border-radius: 8px;
      background:
        linear-gradient(123deg, rgba(24, 1, 31, 0.95) 7%, rgba(182, 0, 168, 0.35) 37%, rgba(118, 33, 176, 0.26) 72%, rgba(190, 76, 0, 0.22) 100%),
        linear-gradient(180deg, rgba(215, 226, 234, 0.08), rgba(215, 226, 234, 0.02));
      overflow: hidden;
      position: relative;
    }}
    .scene::before {{
      content: "";
      position: absolute;
      inset: auto 8% 0;
      height: 78%;
      background:
        linear-gradient(90deg, transparent 0 6%, rgba(215, 226, 234, 0.22) 6% 7%, transparent 7% 14%, rgba(215, 226, 234, 0.16) 14% 15%, transparent 15% 24%, rgba(215, 226, 234, 0.2) 24% 25%, transparent 25% 100%),
        linear-gradient(180deg, transparent 0 18%, rgba(215, 226, 234, 0.18) 18% 19%, transparent 19% 42%, rgba(215, 226, 234, 0.12) 42% 43%, transparent 43% 100%);
      clip-path: polygon(0 72%, 8% 48%, 20% 58%, 30% 18%, 44% 42%, 58% 10%, 70% 46%, 82% 32%, 100% 62%, 100% 100%, 0 100%);
      opacity: 0.9;
    }}
    .scene::after {{
      content: "";
      position: absolute;
      left: 9%;
      right: 9%;
      bottom: 18px;
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(182, 0, 168, 0.85), rgba(190, 76, 0, 0.75), transparent);
    }}
    /* Product-workbench visual pass: keep the app focused on evidence, not spectacle. */
    body {{
      background:
        radial-gradient(circle at 18% 0%, rgba(45, 89, 134, 0.24), transparent 32%),
        linear-gradient(180deg, #0b1220 0%, #0f172a 55%, #10131a 100%);
    }}
    #cityScene {{
      opacity: 0.1;
    }}
    .scene-vignette {{
      background:
        linear-gradient(90deg, rgba(11, 18, 32, 0.96), rgba(15, 23, 42, 0.7) 52%, rgba(11, 18, 32, 0.92)),
        linear-gradient(180deg, rgba(11, 18, 32, 0.35), rgba(11, 18, 32, 0.96));
    }}
    body::before {{
      opacity: 0.4;
      background-size: 48px 48px;
    }}
    h1 {{
      max-width: 780px;
      font-size: clamp(2.8rem, 7vw, 6.2rem);
      line-height: 0.92;
    }}
    .surface {{
      background:
        linear-gradient(180deg, rgba(148, 163, 184, 0.08), rgba(15, 23, 42, 0.88)),
        rgba(15, 23, 42, 0.9);
      box-shadow: 0 26px 80px rgba(0, 0, 0, 0.38);
    }}
    .grid {{
      background: rgba(2, 6, 23, 0.24);
    }}
    label {{
      background: rgba(15, 23, 42, 0.74);
      border-color: rgba(148, 163, 184, 0.18);
    }}
    .actions {{
      background:
        linear-gradient(180deg, rgba(30, 41, 59, 0.86), rgba(15, 23, 42, 0.96));
    }}
    .actions::before {{
      border-color: rgba(56, 189, 248, 0.32);
    }}
    .scene {{
      display: none;
    }}
    button {{
      background: #f8fafc;
      color: #0f172a;
      outline: 0;
      box-shadow: none;
      font-weight: 800;
    }}
    button:hover {{
      background: #dbeafe;
      filter: none;
    }}
    @media (max-width: 820px) {{
      header {{ align-items: stretch; flex-direction: column; }}
      header p, .result {{ max-width: none; text-align: left; }}
      .surface {{ grid-template-columns: 1fr; }}
      .grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      label.address {{ grid-column: span 2; }}
      button {{ width: 100%; }}
      main {{ align-content: start; }}
    }}
    @media (max-width: 520px) {{
      main {{ width: min(100% - 20px, 1080px); padding: 18px 0; }}
      h1 {{ font-size: clamp(2.45rem, 15vw, 4.1rem); }}
      header p {{ font-size: 0.94rem; }}
      .grid {{ grid-template-columns: 1fr; }}
      label.address {{ grid-column: auto; }}
      .scene {{ height: 120px; }}
      .result, .empty-result {{ min-height: 190px; }}
    }}
  </style>
</head>
<body class="loading">
  <div class="loader" aria-hidden="true"><div class="loader-orbit"><div class="loader-core"></div></div></div>
  <canvas id="cityScene" aria-hidden="true"></canvas>
  <div class="scene-vignette" aria-hidden="true"></div>
  <span class="easteregg" aria-hidden="true">Joshue Torres</span>
  <main>
    <nav aria-label="Primary">
      <a href="#estimate">Workbench</a>
      <a href="https://www.zillow.com/research/data/" target="_blank" rel="noreferrer">Data</a>
      <a href="#market">Sources</a>
      <a href="#contact">Deploy</a>
    </nav>
    <header>
      <div>
        <span class="eyebrow">Evidence-first valuation system</span>
        <h1 class="hero-heading">Property Valuation Workbench</h1>
      </div>
      <p>Verify an address, inspect public data coverage, and generate a transparent estimate with model and market-source reasoning.</p>
    </header>
    <section class="signal-strip" aria-label="System capabilities">
      <div><span>Address resolution</span><strong>Census verified</strong></div>
      <div><span>Market anchor</span><strong>Zillow ZIP ZHVI</strong></div>
      <div><span>Model policy</span><strong>No fake precision</strong></div>
      <div><span>Deployment</span><strong>Docker + Render</strong></div>
    </section>
    <form id="estimate" method="post" action="/predict">
      {f'<ul class="errors">{error_items}</ul>' if error_items else ''}
      <div class="surface">
        <div class="grid">
          {fields}
        </div>
        <div class="actions">
          <div class="scene" aria-hidden="true"></div>
          {result}
          <button type="submit">Estimate price</button>
        </div>
      </div>
    </form>
  </main>
  <script type="module">
    import * as THREE from "/static/vendor/three.module.js";

    window.addEventListener("load", () => {{
      setTimeout(() => document.body.classList.remove("loading"), 450);
    }});
    document.body.classList.add("loading");
    document.querySelector("form").addEventListener("submit", (event) => {{
      const button = event.currentTarget.querySelector("button");
      button.disabled = true;
      button.textContent = "Loading market data";
      document.body.classList.add("loading");
    }});

    const addressInput = document.querySelector('input[name="address"]');
    const cityInput = document.querySelector('input[name="city"]');
    const stateInput = document.querySelector('input[name="state"]');
    const zipInput = document.querySelector('input[name="zip_code"]');
    const addressStatus = document.querySelector("[data-address-status]");
    const addressSuggestions = document.querySelector("[data-address-suggestions]");
    const liveContext = document.querySelector("[data-live-context]");
    let addressTimer;
    let addressController;

    function setAddressStatus(message, state = "") {{
      if (!addressStatus) return;
      addressStatus.textContent = message;
      addressStatus.dataset.state = state;
    }}

    function fieldInput(field) {{
      return document.querySelector(`input[name="${{field}}"]`);
    }}

    function setFieldStatus(field, message, state = "") {{
      const element = document.querySelector(`[data-field-status="${{field}}"]`);
      if (!element) return;
      element.textContent = message;
      element.dataset.state = state;
    }}

    function maybeSetUnknown(field) {{
      const input = fieldInput(field);
      if (!input) return;
      if (!input.value.trim()) input.value = "unknown";
    }}

    function escapeText(value) {{
      const element = document.createElement("span");
      element.textContent = value || "";
      return element.innerHTML;
    }}

    function applyEnrichment(payload) {{
      const fields = payload.property_facts?.fields || {{}};
      const missing = payload.property_facts?.missing || {{}};
      Object.entries(fields).forEach(([field, detail]) => {{
        const input = fieldInput(field);
        if (!input) return;
        const current = input.value.trim().toLowerCase();
        if (!current || ["unknown", "unkown", "uknown", "n/a", "na"].includes(current)) {{
          input.value = detail.value;
        }}
        setFieldStatus(field, detail.source, "filled");
      }});
      Object.entries(missing).forEach(([field, message]) => {{
        maybeSetUnknown(field);
        setFieldStatus(field, message, "missing");
      }});
    }}

    function renderLiveContext(payload) {{
      if (!liveContext || !payload.matched) return;
      const lat = Number(payload.latitude);
      const lon = Number(payload.longitude);
      const hasCoords = Number.isFinite(lat) && Number.isFinite(lon);
      const map = payload.mapbox_static_url
        ? `<img alt="Satellite streets map near the verified address" loading="lazy" src="${{escapeText(payload.mapbox_static_url)}}">`
        : hasCoords
        ? `<iframe title="Live mapped address area" loading="lazy" src="https://www.openstreetmap.org/export/embed.html?bbox=${{lon - 0.018}},${{lat - 0.012}},${{lon + 0.018}},${{lat + 0.012}}&layer=mapnik&marker=${{lat}},${{lon}},pm2rdm"></iframe>`
        : "";
      const fields = payload.property_facts?.fields || {{}};
      const distance = fields.distance_to_city_center_miles?.value || "Not available";
      const filled = Object.keys(fields).filter((field) => field !== "distance_to_city_center_miles").length;
      liveContext.innerHTML = `
        <span>Address intelligence</span>
        <div class="live-map">
          ${{map}}
          <dl>
            <div><dt>Verified address</dt><dd>${{escapeText(payload.matched_address)}}</dd></div>
            <div><dt>Location</dt><dd>${{escapeText(payload.city)}}, ${{escapeText(payload.state)}} ${{escapeText(payload.zip_code)}}</dd></div>
            <div><dt>Miles to city center</dt><dd>${{escapeText(distance)}}</dd></div>
            <div><dt>Property facts filled</dt><dd>${{filled}} provider-backed fields</dd></div>
          </dl>
        </div>
      `;
    }}

    function renderSuggestions(suggestions) {{
      if (!addressSuggestions) return;
      addressSuggestions.innerHTML = "";
      addressSuggestions.dataset.open = suggestions.length ? "true" : "false";
      suggestions.slice(0, 5).forEach((suggestion) => {{
        const item = document.createElement("li");
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = suggestion.matched_address;
        button.addEventListener("click", () => {{
          addressInput.value = suggestion.matched_address;
          addressSuggestions.dataset.open = "false";
          verifyAddress();
        }});
        item.appendChild(button);
        addressSuggestions.appendChild(item);
      }});
    }}

    async function loadSuggestions(address) {{
      if (address.length < 6) {{
        renderSuggestions([]);
        return;
      }}
      try {{
        const params = new URLSearchParams({{
          address,
          city: cityInput.value.trim(),
          state: stateInput.value.trim(),
          zip_code: zipInput.value.trim(),
        }});
        const response = await fetch(`/api/suggest?${{params.toString()}}`);
        const payload = await response.json();
        renderSuggestions(payload.suggestions || []);
      }} catch (error) {{
        renderSuggestions([]);
      }}
    }}

    async function verifyAddress() {{
      const address = addressInput.value.trim();
      if (address.length < 8) {{
        setAddressStatus("Type a full U.S. address to verify city, state, and ZIP.");
        return;
      }}
      if (addressController) addressController.abort();
      addressController = new AbortController();
      setAddressStatus("Checking Census Geocoder...");
      try {{
        const response = await fetch(`/api/geocode?address=${{encodeURIComponent(address)}}`, {{
          signal: addressController.signal,
        }});
        const payload = await response.json();
        if (!payload.matched) {{
          setAddressStatus("No verified Census match yet. Add street, city, and state.", "error");
          return;
        }}
        cityInput.value = payload.city || cityInput.value;
        stateInput.value = payload.state || stateInput.value;
        zipInput.value = payload.zip_code || zipInput.value;
        applyEnrichment(payload);
        renderLiveContext(payload);
        renderSuggestions(payload.suggestions || []);
        setAddressStatus(`Verified: ${{payload.matched_address}}`, "matched");
      }} catch (error) {{
        if (error.name === "AbortError") return;
        setAddressStatus("Address verification is unavailable right now.", "error");
      }}
    }}

    if (addressInput) {{
      addressInput.addEventListener("input", () => {{
        clearTimeout(addressTimer);
        addressTimer = setTimeout(() => {{
          loadSuggestions(addressInput.value.trim());
          verifyAddress();
        }}, 650);
      }});
      setAddressStatus("Type a full U.S. address to verify city, state, and ZIP.");
    }}

    const canvas = document.getElementById("cityScene");
    const renderer = new THREE.WebGLRenderer({{ canvas, antialias: true, alpha: true }});
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;

    const scene = new THREE.Scene();
    scene.fog = new THREE.Fog(0x0c0c0c, 10, 36);

    const camera = new THREE.PerspectiveCamera(44, window.innerWidth / window.innerHeight, 0.1, 100);
    camera.position.set(0, 4.6, 13);
    camera.lookAt(0, 0.4, 0);

    const ambient = new THREE.HemisphereLight(0xbbccd7, 0x18011f, 1.15);
    scene.add(ambient);
    const key = new THREE.DirectionalLight(0xbe4c00, 2.2);
    key.position.set(4, 9, 6);
    scene.add(key);
    const rim = new THREE.PointLight(0xb600a8, 26, 34);
    rim.position.set(-6, 3, -5);
    scene.add(rim);

    const ground = new THREE.Mesh(
      new THREE.PlaneGeometry(38, 28, 18, 18),
      new THREE.MeshStandardMaterial({{ color: 0x0c0c0c, metalness: 0.28, roughness: 0.78 }})
    );
    ground.rotation.x = -Math.PI / 2;
    ground.position.y = -1.3;
    scene.add(ground);

    const grid = new THREE.GridHelper(42, 42, 0xb600a8, 0x252733);
    grid.material.transparent = true;
    grid.material.opacity = 0.22;
    grid.position.y = -1.28;
    scene.add(grid);

    const group = new THREE.Group();
    const coreMaterial = new THREE.MeshStandardMaterial({{
      color: 0x15151a,
      metalness: 0.78,
      roughness: 0.18,
      emissive: 0x4d124f,
      emissiveIntensity: 0.38
    }});
    const glassMaterial = new THREE.MeshStandardMaterial({{
      color: 0xbbccd7,
      metalness: 0.24,
      roughness: 0.08,
      transparent: true,
      opacity: 0.34,
      emissive: 0x223348,
      emissiveIntensity: 0.28
    }});

    const core = new THREE.Mesh(new THREE.IcosahedronGeometry(1.36, 2), coreMaterial);
    core.position.set(2.7, 1.35, -2.2);
    group.add(core);

    const glassShell = new THREE.Mesh(new THREE.IcosahedronGeometry(1.78, 1), glassMaterial);
    glassShell.position.copy(core.position);
    group.add(glassShell);

    const ringMaterial = new THREE.MeshBasicMaterial({{ color: 0xbbccd7, transparent: true, opacity: 0.34 }});
    const hotRingMaterial = new THREE.MeshBasicMaterial({{ color: 0xb600a8, transparent: true, opacity: 0.62 }});
    const rings = [];
    [2.9, 4.2, 5.7].forEach((radius, index) => {{
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(radius, 0.012 + index * 0.004, 8, 160),
        index === 1 ? hotRingMaterial : ringMaterial
      );
      ring.position.copy(core.position);
      ring.rotation.x = Math.PI / 2.3 + index * 0.32;
      ring.rotation.y = index * 0.45;
      rings.push(ring);
      group.add(ring);
    }});

    const markerMaterial = new THREE.MeshStandardMaterial({{
      color: 0xbe4c00,
      metalness: 0.6,
      roughness: 0.24,
      emissive: 0xbe4c00,
      emissiveIntensity: 0.5
    }});
    const markers = [];
    for (let i = 0; i < 18; i += 1) {{
      const marker = new THREE.Mesh(new THREE.SphereGeometry(0.055 + (i % 3) * 0.025, 16, 16), markerMaterial);
      const angle = (i / 18) * Math.PI * 2;
      const radius = 2.9 + (i % 3) * 0.95;
      marker.userData = {{ angle, radius, speed: 0.22 + (i % 4) * 0.045, y: (i % 5) * 0.13 - 0.26 }};
      markers.push(marker);
      group.add(marker);
    }}

    const starGeometry = new THREE.BufferGeometry();
    const starPositions = [];
    for (let i = 0; i < 420; i += 1) {{
      starPositions.push(
        (Math.random() - 0.5) * 42,
        Math.random() * 18 - 2,
        (Math.random() - 0.5) * 34
      );
    }}
    starGeometry.setAttribute("position", new THREE.Float32BufferAttribute(starPositions, 3));
    const stars = new THREE.Points(
      starGeometry,
      new THREE.PointsMaterial({{ color: 0xbbccd7, size: 0.035, transparent: true, opacity: 0.7 }})
    );
    scene.add(stars);

    scene.add(group);

    const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    function resize() {{
      const width = window.innerWidth;
      const height = window.innerHeight;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }}
    window.addEventListener("resize", resize);

    function animate(time) {{
      const t = time * 0.001;
      if (!reducedMotion) {{
        group.rotation.y = Math.sin(t * 0.26) * 0.12 - 0.18;
        core.rotation.x = t * 0.28;
        core.rotation.y = t * 0.4;
        glassShell.rotation.y = -t * 0.22;
        rings.forEach((ring, index) => {{
          ring.rotation.z = t * (0.08 + index * 0.035);
          ring.rotation.y += 0.0015 + index * 0.0005;
        }});
        markers.forEach((marker) => {{
          const next = marker.userData.angle + t * marker.userData.speed;
          marker.position.set(
            core.position.x + Math.cos(next) * marker.userData.radius,
            core.position.y + marker.userData.y + Math.sin(next * 1.7) * 0.25,
            core.position.z + Math.sin(next) * marker.userData.radius
          );
        }});
        stars.rotation.y = t * 0.018;
        camera.position.x = Math.sin(t * 0.18) * 0.55;
        camera.lookAt(0.9, 0.6, -1.6);
      }}
      renderer.render(scene, camera);
      requestAnimationFrame(animate);
    }}
    animate(0);
  </script>
</body>
</html>"""


def input_attributes(field: str) -> str:
    if field == "year_built":
        return 'type="text" inputmode="numeric" placeholder="Leave blank unless known"'
    if field in {"address", "city", "state", "neighborhood", "zip_code"}:
        return 'type="text"'
    if field in OPTIONAL_FIELDS:
        return 'type="text" inputmode="decimal" placeholder="unknown"'
    return 'type="number" step="0.01"'


def field_help(field: str) -> str:
    if field == "year_built":
        return '<small class="field-help">Leave blank if unknown. Enter 2026 for a brand-new build scenario.</small>'
    if field == "address":
        return '<small class="field-help">Optional. Used to look up city, state, ZIP, and map area through the U.S. Census Geocoder.</small>'
    if field in {"neighborhood", "square_feet", "bedrooms", "bathrooms", "lot_size", "school_rating", "distance_to_city_center_miles", "crime_index"}:
        return '<small class="field-help">Type unknown or leave blank if public data does not have it.</small>'
    return ""


def address_status(field: str) -> str:
    if field == "address":
        return '<small class="address-status" data-address-status></small>'
    return ""


def address_suggestions(field: str) -> str:
    if field == "address":
        return '<ul class="suggestions" data-address-suggestions></ul>'
    return ""


def field_status(field: str) -> str:
    if field in ADDRESS_ENRICHMENT_FIELDS or field in {"school_rating", "crime_index"}:
        return f'<small class="field-status" data-field-status="{field}"></small>'
    return ""


class AppHandler(BaseHTTPRequestHandler):
    model = None
    zhvi_path = DEFAULT_ZHVI_PATH

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self.respond_json({"status": "ok"})
            return
        if parsed.path == "/api/geocode":
            params = parse_qs(parsed.query)
            self.respond_geocode(params.get("address", [""])[0])
            return
        if parsed.path == "/api/suggest":
            params = parse_qs(parsed.query)
            self.respond_suggestions(
                params.get("address", [""])[0],
                city=params.get("city", [""])[0],
                state=params.get("state", [""])[0],
                zip_code=params.get("zip_code", [""])[0],
            )
            return
        if parsed.path.startswith("/static/"):
            self.respond_static(parsed.path.removeprefix("/static/"))
            return
        if parsed.path not in {"/", "/predict"}:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self.respond_html(render_page())

    def do_POST(self) -> None:
        if self.path != "/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        features, values, errors = parse_form(body)
        prediction = None
        model_prediction = None
        market_signal = None
        census_signal = None
        address_location = None
        decision = None
        if not errors:
            try:
                if values.get("address") and (not values.get("city") or not values.get("state") or not values.get("zip_code")):
                    address_location = self.lookup_address(values["address"])
                    apply_address_location(values, features, address_location)
                    property_facts = self.lookup_property_facts(values["address"])
                    apply_property_facts(values, features, property_facts)
                    distance_miles = self.lookup_distance_to_city_center(address_location) if address_location is not None else None
                    if distance_miles is not None and values.get("distance_to_city_center_miles", "").lower() in {"", "unknown", "unkown", "uknown", "n/a", "na"}:
                        values["distance_to_city_center_miles"] = f"{distance_miles:.1f}"
                        features["distance_to_city_center_miles"] = distance_miles
                prediction_features = finalize_prediction_features(features)
                lookup_zip = str(values.get("zip_code", "") or prediction_features.get("zip_code", ""))
                market_signal = self.lookup_market_signal(lookup_zip)
                census_signal = self.lookup_census_signal(lookup_zip)
                known_fact_count = known_property_fact_count(features)
                if should_use_model(features):
                    model_prediction = predict_price(self.model, prediction_features)
                decision = decide_estimate(
                    model_prediction=model_prediction,
                    market_signal=market_signal,
                    census_signal=census_signal,
                    known_fact_count=known_fact_count,
                )
                prediction = decision.estimate
                if prediction is None:
                    errors.append("Enter a valid U.S. address or ZIP so public market data can anchor the estimate.")
            except ValueError as exc:
                errors.append(str(exc))
            except OSError:
                prediction_features = finalize_prediction_features(features)
                known_fact_count = known_property_fact_count(features)
                if should_use_model(features):
                    model_prediction = predict_price(self.model, prediction_features)
                    decision = decide_estimate(
                        model_prediction=model_prediction,
                        market_signal=None,
                        census_signal=None,
                        known_fact_count=known_fact_count,
                    )
                    prediction = decision.estimate
                else:
                    errors.append("Public data lookup failed, and there are not enough property facts to run a meaningful model estimate.")
                market_signal = None
                census_signal = None
        self.respond_html(
            render_page(
                values,
                prediction=prediction,
                model_prediction=model_prediction,
                market_signal=market_signal,
                census_signal=census_signal,
                address_location=address_location,
                decision=decision,
                errors=errors,
            )
        )

    def lookup_market_signal(self, zip_code: str):
        zhvi_path = ensure_zhvi_csv(self.zhvi_path)
        return latest_zhvi_for_zip(zip_code, zhvi_path)

    def lookup_census_signal(self, zip_code: str):
        return census_home_value_for_zip(zip_code)

    def lookup_address(self, address: str):
        return geocode_address(address)

    def lookup_address_matches(self, address: str):
        return geocode_address_matches(address)

    def lookup_autocomplete_matches(self, address: str):
        try:
            return geoapify_address_suggestions(address)
        except OSError:
            return []

    def lookup_property_facts(self, address: str):
        try:
            return property_facts_for_address(address)
        except OSError:
            return None

    def lookup_distance_to_city_center(self, location):
        try:
            return distance_to_city_center_miles(location)
        except OSError:
            return None

    def respond_suggestions(self, address: str, *, city: str = "", state: str = "", zip_code: str = "") -> None:
        query = enriched_address_query(address, city=city, state=state, zip_code=zip_code)
        try:
            matches = self.lookup_autocomplete_matches(query) or self.lookup_address_matches(query)
        except OSError:
            matches = []
        self.respond_json({"suggestions": [location_payload(match) for match in matches[:5]]})

    def respond_geocode(self, address: str) -> None:
        try:
            location = self.lookup_address(address)
            suggestions = self.lookup_address_matches(address) if location is not None else []
        except OSError:
            location = None
            suggestions = []
        if location is None:
            self.respond_json({"matched": False})
            return
        property_facts = self.lookup_property_facts(address)
        distance_miles = self.lookup_distance_to_city_center(location)
        self.respond_json(
            {
                "matched": True,
                **location_payload(location),
                "suggestions": [location_payload(match) for match in suggestions[:5]],
                "property_facts": property_fact_payload(property_facts, distance_miles),
                "street_view_url": street_view_url(location),
                "mapbox_static_url": mapbox_static_url(location),
            }
        )

    def respond_html(self, body: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def respond_json(self, payload: dict[str, str]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def respond_static(self, relative_path: str) -> None:
        requested = (STATIC_ROOT / relative_path).resolve()
        if not requested.is_file() or STATIC_ROOT.resolve() not in requested.parents:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = requested.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(requested.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real estate price estimator web app.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, type=Path)
    parser.add_argument("--data", default=DEFAULT_DATA_PATH, type=Path)
    parser.add_argument("--zhvi-cache", default=DEFAULT_ZHVI_PATH, type=Path)
    args = parser.parse_args()

    AppHandler.model = ensure_model(args.model, args.data)
    AppHandler.zhvi_path = args.zhvi_cache
    server = ThreadingHTTPServer((args.host, args.port), AppHandler)
    print(f"Serving real estate estimator at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
