from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "price"

CATEGORICAL_FEATURES = ["city", "neighborhood", "zip_code"]
NUMERIC_FEATURES = [
    "square_feet",
    "bedrooms",
    "bathrooms",
    "lot_size",
    "year_built",
    "school_rating",
    "distance_to_city_center_miles",
    "crime_index",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass(frozen=True)
class EvaluationMetrics:
    mae: float
    rmse: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2}


def load_training_data(path: str | Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in data.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {', '.join(missing)}")
    return data[FEATURE_COLUMNS + [TARGET_COLUMN]]


def build_pipeline() -> Pipeline:
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.08,
        max_iter=350,
        l2_regularization=0.05,
        random_state=42,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def train(
    data: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[Pipeline, EvaluationMetrics]:
    missing = [column for column in FEATURE_COLUMNS + [TARGET_COLUMN] if column not in data.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {', '.join(missing)}")

    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)
    metrics = evaluate(pipeline, x_test, y_test)
    return pipeline, metrics


def evaluate(model: Pipeline, x: pd.DataFrame, y: pd.Series) -> EvaluationMetrics:
    predictions = model.predict(x)
    mse = mean_squared_error(y, predictions)
    return EvaluationMetrics(
        mae=float(mean_absolute_error(y, predictions)),
        rmse=float(mse**0.5),
        r2=float(r2_score(y, predictions)),
    )


def save_model(model: Pipeline, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def load_model(path: str | Path) -> Pipeline:
    return joblib.load(path)


def predict_price(model: Pipeline, features: dict[str, object] | pd.DataFrame) -> float:
    if isinstance(features, dict):
        data = pd.DataFrame([features])
    else:
        data = features.copy()

    missing = [column for column in FEATURE_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Prediction input is missing required columns: {', '.join(missing)}")

    return float(model.predict(data[FEATURE_COLUMNS])[0])
