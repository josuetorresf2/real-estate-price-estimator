from io import BytesIO

from real_estate_price_estimator.wsgi import app


def call_wsgi(path: str, *, method: str = "GET", body: bytes = b""):
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = dict(headers)

    response = b"".join(
        app(
            {
                "REQUEST_METHOD": method,
                "PATH_INFO": path,
                "QUERY_STRING": "",
                "CONTENT_LENGTH": str(len(body)),
                "wsgi.input": BytesIO(body),
            },
            start_response,
        )
    )
    return captured, response


def test_wsgi_health_endpoint():
    captured, response = call_wsgi("/health")

    assert captured["status"].startswith("200")
    assert response == b'{"status": "ok"}'


def test_wsgi_predict_page_renders():
    captured, response = call_wsgi("/predict")

    assert captured["status"].startswith("200")
    assert b"Property Valuation Workbench" in response


def test_wsgi_static_asset_renders():
    captured, response = call_wsgi("/static/vendor/three.module.js")

    assert captured["status"].startswith("200")
    assert b"three" in response.lower()
