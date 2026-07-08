from http import HTTPStatus

from real_estate_price_estimator.web_app import AppHandler, format_currency, parse_form, render_page


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


def test_render_page_escapes_values_and_shows_prediction():
    page = render_page({"city": "<Austin>", "neighborhood": "North Loop"}, prediction=515000)

    assert "&lt;Austin&gt;" in page
    assert "$515,000" in page


def test_format_currency_rounds_to_whole_dollars():
    assert format_currency(1211164.76) == "$1,211,165"


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
