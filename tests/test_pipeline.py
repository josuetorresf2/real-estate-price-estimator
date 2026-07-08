import pandas as pd
import pytest

from real_estate_price_estimator.pipeline import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_pipeline,
    predict_price,
    train,
)


def training_frame() -> pd.DataFrame:
    rows = []
    cities = [
        ("Austin", "North Loop", "78751", 450_000),
        ("Denver", "Highland", "80211", 540_000),
        ("Seattle", "Ballard", "98107", 650_000),
    ]
    for index in range(30):
        city, neighborhood, zip_code, base_price = cities[index % len(cities)]
        square_feet = 1200 + index * 55
        school_rating = 6.8 + (index % 5) * 0.4
        rows.append(
            {
                "city": city,
                "neighborhood": neighborhood,
                "zip_code": zip_code,
                "square_feet": square_feet,
                "bedrooms": 2 + index % 4,
                "bathrooms": 1.5 + (index % 3) * 0.5,
                "lot_size": 0.1 + (index % 6) * 0.025,
                "year_built": 1975 + index,
                "school_rating": school_rating,
                "distance_to_city_center_miles": 2.5 + (index % 8),
                "crime_index": 20 + (index % 9) * 3,
                "price": base_price + square_feet * 185 + school_rating * 18_000,
            }
        )
    return pd.DataFrame(rows)


def test_build_pipeline_contains_preprocessing_and_model_steps():
    pipeline = build_pipeline()

    assert list(pipeline.named_steps) == ["preprocessor", "model"]


def test_train_returns_pipeline_and_metrics():
    model, metrics = train(training_frame(), test_size=0.25)

    assert hasattr(model, "predict")
    assert metrics.mae >= 0
    assert metrics.rmse >= 0
    assert -1 <= metrics.r2 <= 1


def test_predict_price_accepts_unseen_category():
    model, _ = train(training_frame(), test_size=0.25)
    features = {
        "city": "Portland",
        "neighborhood": "New Neighborhood",
        "zip_code": "97214",
        "square_feet": 1800,
        "bedrooms": 3,
        "bathrooms": 2,
        "lot_size": 0.12,
        "year_built": 2001,
        "school_rating": 8.2,
        "distance_to_city_center_miles": 4.5,
        "crime_index": 28,
    }

    prediction = predict_price(model, features)

    assert prediction > 0


def test_train_rejects_missing_required_columns():
    data = training_frame().drop(columns=[TARGET_COLUMN])

    with pytest.raises(ValueError, match=TARGET_COLUMN):
        train(data)


def test_predict_rejects_missing_required_columns():
    model, _ = train(training_frame(), test_size=0.25)
    features = {column: 1 for column in FEATURE_COLUMNS}
    features.pop("school_rating")

    with pytest.raises(ValueError, match="school_rating"):
        predict_price(model, features)
