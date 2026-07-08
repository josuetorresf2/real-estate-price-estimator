from http import HTTPStatus

from real_estate_price_estimator.market_data import AddressLocation
from real_estate_price_estimator.web_app import (
    AppHandler,
    apply_address_location,
    finalize_prediction_features,
    format_currency,
    map_preview,
    parse_form,
    render_page,
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


def test_get_predict_route_renders_app(monkeypatch):
    responses = []

    monkeypatch.setattr(AppHandler, "respond_html", lambda self, body: responses.append(body))
    monkeypatch.setattr(AppHandler, "send_error", lambda self, status: responses.append(status))

    handler = object.__new__(AppHandler)
    handler.path = "/predict"

    handler.do_GET()

    assert responses
    assert responses[0] != HTTPStatus.NOT_FOUND
    assert "Real Estate Price Estimator" in responses[0]
