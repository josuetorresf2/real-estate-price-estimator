from __future__ import annotations

import json
import mimetypes
from http import HTTPStatus
from pathlib import Path
from urllib.parse import parse_qs

from .market_data import (
    census_home_value_for_zip,
    distance_to_city_center_miles,
    ensure_zhvi_csv,
    geocode_address_matches,
    geocode_address_global,
    geoapify_address_suggestions,
    geoapify_neighborhood_for_address,
    latest_zhvi_for_zip,
    nominatim_address_matches,
    property_facts_for_address,
    regional_listing_signal,
    regional_macro_signal,
    reverse_geocode_global,
)
from .pipeline import predict_price
from .web_app import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_ZHVI_PATH,
    STATIC_ROOT,
    apply_address_location,
    apply_neighborhood,
    apply_property_facts,
    decide_estimate,
    enriched_address_query,
    enriched_location_payload,
    ensure_model,
    finalize_prediction_features,
    known_property_fact_count,
    location_payload,
    parse_form,
    render_page,
    should_use_model,
)


class EstimatorService:
    def __init__(
        self,
        *,
        model_path: Path = DEFAULT_MODEL_PATH,
        data_path: Path = DEFAULT_DATA_PATH,
        zhvi_path: Path = DEFAULT_ZHVI_PATH,
    ) -> None:
        self.model = ensure_model(model_path, data_path)
        self.zhvi_path = zhvi_path

    def lookup_market_signal(self, zip_code: str):
        zhvi_path = ensure_zhvi_csv(self.zhvi_path)
        return latest_zhvi_for_zip(zip_code, zhvi_path)

    def lookup_census_signal(self, zip_code: str):
        return census_home_value_for_zip(zip_code)

    def lookup_address(self, address: str, *, country: str = "United States"):
        return geocode_address_global(address, country=country)

    def lookup_address_matches(self, address: str, *, country: str = "United States"):
        if country == "United States":
            return geocode_address_matches(address)
        return nominatim_address_matches(address, country=country)

    def lookup_autocomplete_matches(self, address: str, *, country: str = "United States"):
        try:
            return geoapify_address_suggestions(address, country=country)
        except OSError:
            return []

    def lookup_property_facts(self, address: str, *, country: str = "United States"):
        if country != "United States":
            return None
        try:
            return property_facts_for_address(address)
        except OSError:
            return None

    def lookup_neighborhood(self, address: str, *, country: str = "United States"):
        try:
            return geoapify_neighborhood_for_address(address, country=country)
        except OSError:
            return None

    def lookup_regional_listing_signal(self, query: str, *, country: str):
        try:
            return regional_listing_signal(query, country=country)
        except OSError:
            return None

    def lookup_regional_macro_signal(self, *, country: str):
        try:
            return regional_macro_signal(country)
        except OSError:
            return None

    def lookup_distance_to_city_center(self, location):
        try:
            return distance_to_city_center_miles(location)
        except OSError:
            return None

    def lookup_reverse_geocode(self, latitude: float, longitude: float):
        try:
            return reverse_geocode_global(latitude, longitude)
        except OSError:
            return None


service = EstimatorService()


def app(environ, start_response):
    method = environ.get("REQUEST_METHOD", "GET").upper()
    path = environ.get("PATH_INFO", "/")
    query = parse_qs(environ.get("QUERY_STRING", ""))

    if method == "GET":
        if path == "/health":
            return _json(start_response, {"status": "ok"})
        if path == "/api/geocode":
            return _json(start_response, _geocode_payload(query))
        if path == "/api/suggest":
            return _json(start_response, _suggestions_payload(query))
        if path == "/api/reverse-geocode":
            return _json(start_response, _reverse_geocode_payload(query))
        if path.startswith("/static/"):
            return _static(start_response, path.removeprefix("/static/"))
        if path in {"/", "/predict"}:
            return _html(start_response, render_page())
        return _html(start_response, "Not found", status=HTTPStatus.NOT_FOUND)

    if method == "POST" and path == "/predict":
        length = int(environ.get("CONTENT_LENGTH") or "0")
        body = environ["wsgi.input"].read(length).decode("utf-8")
        return _html(start_response, _prediction_page(body))

    return _html(start_response, "Not found", status=HTTPStatus.NOT_FOUND)


def _prediction_page(body: str) -> str:
    features, values, errors = parse_form(body)
    prediction = None
    model_prediction = None
    market_signal = None
    census_signal = None
    regional_signal = None
    address_location = None
    decision = None
    if not errors:
        try:
            country = values.get("country") or "United States"
            if values.get("address"):
                address_location = service.lookup_address(values["address"], country=country)
                apply_address_location(values, features, address_location)
                property_facts = service.lookup_property_facts(values["address"], country=country)
                apply_property_facts(values, features, property_facts)
                neighborhood = service.lookup_neighborhood(values["address"], country=country)
                apply_neighborhood(values, features, neighborhood)
                distance_miles = service.lookup_distance_to_city_center(address_location) if address_location is not None else None
                if distance_miles is not None and values.get("distance_to_city_center_miles", "").lower() in {"", "unknown", "unkown", "uknown", "n/a", "na"}:
                    values["distance_to_city_center_miles"] = f"{distance_miles:.1f}"
                    features["distance_to_city_center_miles"] = distance_miles
                if address_location is not None and country != "United States":
                    regional_query = " ".join(part for part in (address_location.city, address_location.state, country) if part)
                    regional_signal = service.lookup_regional_listing_signal(regional_query, country=country)
            prediction_features = finalize_prediction_features(features)
            lookup_zip = str(values.get("zip_code", "") or prediction_features.get("zip_code", ""))
            if country == "United States":
                market_signal = service.lookup_market_signal(lookup_zip)
                census_signal = service.lookup_census_signal(lookup_zip)
            known_fact_count = known_property_fact_count(features)
            if should_use_model(features):
                model_prediction = predict_price(service.model, prediction_features)
            decision = decide_estimate(
                model_prediction=model_prediction,
                market_signal=market_signal,
                census_signal=census_signal,
                regional_signal=regional_signal,
                known_fact_count=known_fact_count,
            )
            prediction = decision.estimate
            if prediction is None:
                errors.append("No public pricing baseline was found for this location. Add square feet, bedrooms, bathrooms, and lot size, or choose a more specific verified address suggestion.")
        except ValueError as exc:
            errors.append(str(exc))
        except OSError:
            prediction_features = finalize_prediction_features(features)
            known_fact_count = known_property_fact_count(features)
            if should_use_model(features):
                model_prediction = predict_price(service.model, prediction_features)
                decision = decide_estimate(
                    model_prediction=model_prediction,
                    market_signal=None,
                    census_signal=None,
                    known_fact_count=known_fact_count,
                )
                prediction = decision.estimate
            else:
                errors.append("The live public data lookup is temporarily unavailable. Add square feet, bedrooms, bathrooms, and lot size to run the property model, or try again after selecting a verified address suggestion.")
            market_signal = None
            census_signal = None
    return render_page(
        values,
        prediction=prediction,
        model_prediction=model_prediction,
        market_signal=market_signal,
        census_signal=census_signal,
        address_location=address_location,
        decision=decision,
        errors=errors,
    )


def _geocode_payload(query: dict[str, list[str]]) -> dict[str, object]:
    address = query.get("address", [""])[0]
    country = query.get("country", ["United States"])[0]
    try:
        location = service.lookup_address(address, country=country)
        suggestions = service.lookup_address_matches(address, country=country) if location is not None else []
    except OSError:
        location = None
        suggestions = []
    if location is None:
        return {"matched": False}
    property_facts = service.lookup_property_facts(address, country=country)
    neighborhood = service.lookup_neighborhood(address, country=country)
    distance_miles = service.lookup_distance_to_city_center(location)
    regional_query = " ".join(part for part in (location.city, location.state, country) if part)
    return enriched_location_payload(
        location,
        country=country,
        suggestions=suggestions,
        property_facts=property_facts,
        neighborhood=neighborhood,
        distance_miles=distance_miles,
        regional_signal=service.lookup_regional_listing_signal(regional_query, country=country),
        macro_signal=service.lookup_regional_macro_signal(country=country),
    )


def _suggestions_payload(query: dict[str, list[str]]) -> dict[str, object]:
    address = query.get("address", [""])[0]
    country = query.get("country", ["United States"])[0]
    search = enriched_address_query(
        address,
        city=query.get("city", [""])[0],
        state=query.get("state", [""])[0],
        zip_code=query.get("zip_code", [""])[0],
    )
    try:
        matches = (
            service.lookup_autocomplete_matches(search, country=country)
            or nominatim_address_matches(search, country=country)
            or service.lookup_address_matches(search, country=country)
        )
    except OSError:
        matches = []
    return {"suggestions": [location_payload(match) for match in matches[:5]]}


def _reverse_geocode_payload(query: dict[str, list[str]]) -> dict[str, object]:
    country = query.get("country", ["United States"])[0]
    try:
        location = service.lookup_reverse_geocode(float(query.get("lat", [""])[0]), float(query.get("lon", [""])[0]))
    except ValueError:
        location = None
    if location is None:
        return {"matched": False}
    regional_query = " ".join(part for part in (location.city, location.state, country) if part)
    return enriched_location_payload(
        location,
        country=country,
        suggestions=[location],
        property_facts=None,
        neighborhood=service.lookup_neighborhood(location.matched_address, country=country),
        distance_miles=service.lookup_distance_to_city_center(location),
        regional_signal=service.lookup_regional_listing_signal(regional_query, country=country),
        macro_signal=service.lookup_regional_macro_signal(country=country),
    )


def _html(start_response, body: str, *, status: HTTPStatus = HTTPStatus.OK):
    return _response(start_response, body.encode("utf-8"), status=status, content_type="text/html; charset=utf-8")


def _json(start_response, payload: dict[str, object], *, status: HTTPStatus = HTTPStatus.OK):
    return _response(start_response, json.dumps(payload).encode("utf-8"), status=status, content_type="application/json")


def _static(start_response, relative_path: str):
    requested = (STATIC_ROOT / relative_path).resolve()
    if not requested.is_file() or STATIC_ROOT.resolve() not in requested.parents:
        return _html(start_response, "Not found", status=HTTPStatus.NOT_FOUND)
    return _response(
        start_response,
        requested.read_bytes(),
        status=HTTPStatus.OK,
        content_type=mimetypes.guess_type(requested.name)[0] or "application/octet-stream",
    )


def _response(start_response, body: bytes, *, status: HTTPStatus, content_type: str):
    start_response(
        f"{status.value} {status.phrase}",
        [
            ("Content-Type", content_type),
            ("Content-Length", str(len(body))),
        ],
    )
    return [body]
