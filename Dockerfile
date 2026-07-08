FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY data ./data
COPY src ./src
COPY static ./static

EXPOSE 8000

CMD ["sh", "-c", "python -m real_estate_price_estimator.web_app --host 0.0.0.0 --port ${PORT:-8000}"]
