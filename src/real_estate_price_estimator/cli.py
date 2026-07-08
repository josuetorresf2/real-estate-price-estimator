from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import FEATURE_COLUMNS, load_model, load_training_data, predict_price, save_model, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and use a real estate price estimator.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train a price regression pipeline.")
    train_parser.add_argument("--data", required=True, type=Path, help="CSV file containing training data.")
    train_parser.add_argument(
        "--model-out",
        default=Path("models/price_pipeline.joblib"),
        type=Path,
        help="Path where the trained model will be saved.",
    )
    train_parser.add_argument("--test-size", default=0.2, type=float, help="Fraction of data used for evaluation.")
    train_parser.add_argument(
        "--min-r2",
        default=None,
        type=float,
        help="Fail training if holdout R2 is below this value.",
    )

    predict_parser = subparsers.add_parser("predict", help="Predict the price of a single property.")
    predict_parser.add_argument("--model", required=True, type=Path, help="Path to a trained model artifact.")
    predict_parser.add_argument("--city", required=True)
    predict_parser.add_argument("--neighborhood", required=True)
    predict_parser.add_argument("--zip-code", required=True, dest="zip_code")
    predict_parser.add_argument("--square-feet", required=True, type=float, dest="square_feet")
    predict_parser.add_argument("--bedrooms", required=True, type=float)
    predict_parser.add_argument("--bathrooms", required=True, type=float)
    predict_parser.add_argument("--lot-size", required=True, type=float, dest="lot_size")
    predict_parser.add_argument("--year-built", required=True, type=int, dest="year_built")
    predict_parser.add_argument("--school-rating", required=True, type=float, dest="school_rating")
    predict_parser.add_argument(
        "--distance-to-city-center-miles",
        required=True,
        type=float,
        dest="distance_to_city_center_miles",
    )
    predict_parser.add_argument("--crime-index", required=True, type=float, dest="crime_index")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.command == "train":
        data = load_training_data(args.data)
        model, metrics = train(data, test_size=args.test_size)
        if args.min_r2 is not None and metrics.r2 < args.min_r2:
            raise SystemExit(f"Holdout R2 {metrics.r2:.3f} is below required minimum {args.min_r2:.3f}.")
        save_model(model, args.model_out)
        print(json.dumps({"model": str(args.model_out), "metrics": metrics.as_dict()}, indent=2))
        return

    if args.command == "predict":
        model = load_model(args.model)
        features = {column: getattr(args, column) for column in FEATURE_COLUMNS}
        prediction = predict_price(model, features)
        print(json.dumps({"predicted_price": round(prediction, 2)}, indent=2))


if __name__ == "__main__":
    main()
