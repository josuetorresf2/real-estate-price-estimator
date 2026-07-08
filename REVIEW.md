# Review Notes

## Done When

- `python -m pytest` passes.
- `python scripts/generate_sample_data.py` writes a deterministic generated training CSV.
- `python -m real_estate_price_estimator.cli train --data data/sample_housing.csv --model-out models/price_pipeline.joblib` trains and writes a model artifact.
- `python -m real_estate_price_estimator.cli train --data data/generated_housing.csv --model-out models/price_pipeline.joblib --min-r2 0.80` meets the holdout quality gate.
- `python -m real_estate_price_estimator.cli predict --model models/price_pipeline.joblib ...` returns a positive predicted price.
- `python -m real_estate_price_estimator.web_app --port 8000` serves a browser form for non-technical users.
- `/predict` blends the trained model with Zillow Research ZIP-level ZHVI when the ZIP appears in the public dataset.
- `year_built` is optional for prediction; missing values are handled by the model pipeline imputer.
- `/static/vendor/three.module.js` serves the local Three.js module so the 3D scene does not rely on `unpkg.com`.
- `./verify.sh` passes from the project root.

## Dependency Rationale

- `pandas`: CSV loading and tabular feature handling.
- `scikit-learn`: preprocessing, regression model, train/test split, and metrics.
- `joblib`: model artifact serialization compatible with scikit-learn.
- `pytest`: machine-checkable test verification.

## Verification Adjustment

The first verification run passed unit tests but failed the CLI smoke test because the `src/` package was not on `PYTHONPATH`. `verify.sh` now exports `PYTHONPATH=src` before running the train and predict commands.

The second verification pass exposed that the tiny hand-written sample CSV produced a negative holdout R2, so `verify.sh` now trains against a deterministic generated dataset and enforces `--min-r2 0.80`.

## Scope

This is a standalone estimator project. It does not touch auth, billing, migrations, secrets, deployment credentials, or the existing `ai-resume-job-matcher` application.

The app uses Python's standard `http.server` stack, so no web framework dependency was added.

Zillow access uses the official free Zillow Research CSV feed. Address-level Zestimates and public records require approved Bridge/Zillow data access and are not self-serve free APIs.
