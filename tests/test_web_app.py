from http import HTTPStatus

from real_estate_price_estimator.market_data import AddressLocation, PropertyFacts
from real_estate_price_estimator.web_app import (
    AppHandler,
    apply_address_location,
    apply_neighborhood,
    apply_property_facts,
    address_suggestions,
    address_status,
    decide_estimate,
    enriched_address_query,
    finalize_prediction_features,
    format_currency,
    known_property_fact_count,
    map_preview,
    parse_form,
    render_page,
    should_use_model,
)


def valid_body() -> str:
    return (
        "city=Austin&neighborhood=North+Loop&zip_code=78751&square_feet=1850"
        "&bedrooms=3&bathrooms=2&lot_size=0.18&year_built=1998"
        "&school_rating=8.6&distance_to_city_center_miles=4.2&crime_index=31"
    )


def test_parse_form_returns_typed_features():
    features, values, errors = parse_form(valid_body())

    assert errors == []
    assert values["city"] == "Austin"
    assert features["square_feet"] == 1850
    assert features["year_built"] == 1998
    assert features["school_rating"] == 8.6


def test_parse_form_reports_invalid_numbers():
    _, _, errors = parse_form(valid_body().replace("square_feet=1850", "square_feet=large"))

    assert "Square feet must be a number." in errors


def test_parse_form_allows_blank_year_built():
    features, _, errors = parse_form(valid_body().replace("year_built=1998", "year_built="))

    assert errors == []
    assert features["year_built"] is None


def test_parse_form_allows_unknown_property_fields():
    body = (
        "city=&neighborhood=unknown&zip_code=78751&square_feet=unknown"
        "&bedrooms=&bathrooms=unknown&lot_size=&year_built="
        "&school_rating=&distance_to_city_center_miles=unknown&crime_index="
    )
    features, _, errors = parse_form(body)

    assert errors == []
    finalized = finalize_prediction_features(features)
    assert finalized["city"] == "Unknown"
    assert finalized["neighborhood"] == "Unknown"
    assert finalized["square_feet"] is None
    assert finalized["bathrooms"] is None
    assert finalized["school_rating"] is None
    assert known_property_fact_count(features) == 0
    assert should_use_model(features) is False


def test_should_use_model_requires_core_property_facts():
    features = {
        "square_feet": 1850,
        "bedrooms": 3,
        "bathrooms": 2,
        "lot_size": 0.18,
        "year_built": None,
        "school_rating": None,
        "distance_to_city_center_miles": None,
        "crime_index": None,
    }

    assert should_use_model(features) is True


def test_decide_estimate_prefers_public_market_baseline_without_core_facts():
    class Signal:
        typical_home_value = 600000

    decision = decide_estimate(
        model_prediction=None,
        market_signal=Signal(),
        census_signal=None,
        known_fact_count=0,
    )

    assert decision.estimate == 600000
    assert decision.used_model is False
    assert decision.method == "Public-data market baseline: Zillow Research ZHVI"


def test_decide_estimate_uses_model_when_core_facts_are_known():
    class Signal:
        typical_home_value = 600000

    decision = decide_estimate(
        model_prediction=1000000,
        market_signal=Signal(),
        census_signal=None,
        known_fact_count=4,
    )

    assert decision.estimate == 780000
    assert decision.used_model is True


def test_render_page_escapes_values_and_shows_prediction():
    page = render_page({"city": "<Austin>", "neighborhood": "North Loop"}, prediction=515000)

    assert "&lt;Austin&gt;" in page
    assert "$515,000" in page


def test_format_currency_rounds_to_whole_dollars():
    assert format_currency(1211164.76) == "$1,211,165"


def test_apply_address_location_fills_missing_city_state_zip():
    values = {"address": "1600 Pennsylvania Ave NW", "city": "", "state": "", "zip_code": ""}
    features = {"neighborhood": "Unknown"}
    location = AddressLocation(
        city="Washington",
        state="DC",
        zip_code="20500",
        matched_address="1600 PENNSYLVANIA AVE NW, WASHINGTON, DC, 20500",
        latitude=38.8977,
        longitude=-77.0365,
    )

    apply_address_location(values, features, location)

    assert values["city"] == "Washington"
    assert values["state"] == "DC"
    assert values["zip_code"] == "20500"
    assert features["city"] == "Washington"
    assert features["zip_code"] == "20500"


def test_apply_property_facts_fills_unknown_inputs():
    values = {"square_feet": "unknown", "bedrooms": "", "bathrooms": "1", "lot_size": "", "year_built": ""}
    features = {}
    facts = PropertyFacts(square_feet=2200, bedrooms=4, bathrooms=2.5, lot_size=0.19, year_built=2001)

    apply_property_facts(values, features, facts)

    assert values["square_feet"] == "2200"
    assert values["bedrooms"] == "4"
    assert values["bathrooms"] == "1"
    assert values["lot_size"] == "0.19"
    assert values["year_built"] == "2001"
    assert features["square_feet"] == 2200
    assert features["lot_size"] == 0.19


def test_apply_neighborhood_fills_unknown_value():
    values = {"neighborhood": "unknown"}
    features = {}

    apply_neighborhood(values, features, "Old North End")

    assert values["neighborhood"] == "Old North End"
    assert features["neighborhood"] == "Old North End"


def test_map_preview_renders_openstreetmap_if_coordinates_exist():
    location = AddressLocation(
        city="Washington",
        state="DC",
        zip_code="20500",
        matched_address="1600 PENNSYLVANIA AVE NW, WASHINGTON, DC, 20500",
        latitude=38.8977,
        longitude=-77.0365,
    )

    preview = map_preview(location)

    assert "openstreetmap.org" in preview
    assert "Mapped address area" in preview


def test_address_status_markup_is_rendered_only_for_address():
    assert "data-address-status" in address_status("address")
    assert address_status("city") == ""
    assert "data-address-suggestions" in address_suggestions("address")
    assert address_suggestions("city") == ""


def test_enriched_address_query_uses_location_context():
    assert enriched_address_query("5324 Alt", city="Austin", state="TX", zip_code="78751") == "5324 Alt Austin TX 78751"


def test_get_predict_route_renders_app(monkeypatch):
    responses = []

    monkeypatch.setattr(AppHandler, "respond_html", lambda self, body: responses.append(body))
    monkeypatch.setattr(AppHandler, "send_error", lambda self, status: responses.append(status))

    handler = object.__new__(AppHandler)
    handler.path = "/predict"

    handler.do_GET()

    assert responses
    assert responses[0] != HTTPStatus.NOT_FOUND
    assert "Property Valuation Workbench" in responses[0]


def test_geocode_route_returns_verified_location(monkeypatch):
    responses = []
    location = AddressLocation(
        city="Denver",
        state="CO",
        zip_code="80202",
        matched_address="100 MAIN ST, DENVER, CO, 80202",
        latitude=39.75,
        longitude=-104.99,
    )

    monkeypatch.setattr(AppHandler, "lookup_address", lambda self, address, country="United States": location)
    queries = []
    monkeypatch.setattr(AppHandler, "lookup_address_matches", lambda self, address, country="United States": queries.append(address) or [location])
    monkeypatch.setattr(AppHandler, "lookup_property_facts", lambda self, address, country="United States": None)
    monkeypatch.setattr(AppHandler, "lookup_neighborhood", lambda self, address, country="United States": "Downtown")
    monkeypatch.setattr(AppHandler, "lookup_distance_to_city_center", lambda self, location: 1.4)
    monkeypatch.setattr(AppHandler, "lookup_regional_listing_signal", lambda self, query, country: None)
    monkeypatch.setattr(AppHandler, "lookup_regional_macro_signal", lambda self, country: None)
    monkeypatch.setattr(AppHandler, "respond_json", lambda self, payload: responses.append(payload))

    handler = object.__new__(AppHandler)
    handler.path = "/api/geocode?address=100%20Main%20St%20Denver%20CO"

    handler.do_GET()

    assert responses == [
        {
            "matched": True,
            "city": "Denver",
            "state": "CO",
            "zip_code": "80202",
            "matched_address": "100 MAIN ST, DENVER, CO, 80202",
            "latitude": "39.75",
            "longitude": "-104.99",
            "source": "U.S. Census Geocoder",
            "country": "United States",
            "suggestions": [
                {
                    "city": "Denver",
                    "state": "CO",
                    "zip_code": "80202",
                    "matched_address": "100 MAIN ST, DENVER, CO, 80202",
                    "latitude": "39.75",
                    "longitude": "-104.99",
                    "source": "U.S. Census Geocoder",
                }
            ],
            "property_facts": {
                "fields": {
                    "distance_to_city_center_miles": {
                        "value": "1.4",
                        "source": "Calculated from verified geocoded coordinates",
                    },
                    "neighborhood": {
                        "value": "Downtown",
                        "source": "Geoapify neighborhood/suburb signal",
                    }
                },
                "missing": {
                    "square_feet": "Square footage is unavailable from the current verified property-record sources.",
                    "bedrooms": "Bedroom count is unavailable from the current verified property-record sources.",
                    "bathrooms": "Bathroom count is unavailable from the current verified property-record sources.",
                    "lot_size": "Lot size is unavailable from the current verified property-record sources.",
                    "year_built": "Year built is unavailable from the current verified property-record sources.",
                    "school_rating": "School rating is unavailable from the current verified public sources.",
                    "crime_index": "Crime index is unavailable from the current verified public sources.",
                },
                "provider_configured": False,
            },
            "regional_listing": {
                "available": False,
                "message": "No regional listing context found for this address/city.",
            },
            "regional_macro": {
                "available": False,
                "message": "No regional macro context found.",
            },
            "street_view_url": "",
            "mapbox_static_url": "",
        }
    ]


def test_suggest_route_returns_address_options(monkeypatch):
    responses = []
    location = AddressLocation(
        city="Austin",
        state="TX",
        zip_code="78751",
        matched_address="5324 AVENUE F, AUSTIN, TX, 78751",
    )

    queries = []
    monkeypatch.setattr(AppHandler, "lookup_address_matches", lambda self, address, country="United States": queries.append(address) or [location])
    monkeypatch.setattr(AppHandler, "respond_json", lambda self, payload: responses.append(payload))

    handler = object.__new__(AppHandler)
    handler.path = "/api/suggest?address=5324%20Av&city=Austin&state=TX&zip_code=78751"

    handler.do_GET()

    assert queries == ["5324 Av Austin TX 78751"]
    assert responses[0]["suggestions"][0]["matched_address"] == "5324 AVENUE F, AUSTIN, TX, 78751"


def test_suggest_route_uses_public_fallback_for_short_fragments(monkeypatch):
    responses = []
    location = AddressLocation(
        city="Austin",
        state="TX",
        zip_code="78753",
        matched_address="5324 EXAMPLE RD, AUSTIN, TX, 78753",
    )

    monkeypatch.setattr(AppHandler, "lookup_autocomplete_matches", lambda self, address, country="United States": [])
    monkeypatch.setattr("real_estate_price_estimator.web_app.nominatim_address_matches", lambda address, country="United States": [location])
    monkeypatch.setattr(AppHandler, "respond_json", lambda self, payload: responses.append(payload))

    handler = object.__new__(AppHandler)
    handler.path = "/api/suggest?address=5324&country=United%20States"

    handler.do_GET()

    assert responses[0]["suggestions"][0]["matched_address"] == "5324 EXAMPLE RD, AUSTIN, TX, 78753"


def test_reverse_geocode_route_returns_selected_map_point(monkeypatch):
    responses = []
    location = AddressLocation(
        city="Quito",
        state="Pichincha",
        zip_code="170518",
        matched_address="La Carolina, Quito, Ecuador",
        latitude=-0.1934,
        longitude=-78.4824,
        source="OpenStreetMap Nominatim",
    )

    monkeypatch.setattr(AppHandler, "lookup_reverse_geocode", lambda self, latitude, longitude: location)
    monkeypatch.setattr(AppHandler, "lookup_neighborhood", lambda self, address, country="United States": None)
    monkeypatch.setattr(AppHandler, "lookup_distance_to_city_center", lambda self, location: None)
    monkeypatch.setattr(AppHandler, "lookup_regional_listing_signal", lambda self, query, country: None)
    monkeypatch.setattr(AppHandler, "lookup_regional_macro_signal", lambda self, country: None)
    monkeypatch.setattr(AppHandler, "respond_json", lambda self, payload: responses.append(payload))

    handler = object.__new__(AppHandler)
    handler.path = "/api/reverse-geocode?lat=-0.1934&lon=-78.4824&country=Ecuador"

    handler.do_GET()

    assert responses[0]["matched"] is True
    assert responses[0]["country"] == "Ecuador"
    assert responses[0]["matched_address"] == "La Carolina, Quito, Ecuador"
