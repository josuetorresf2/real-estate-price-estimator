from __future__ import annotations

import csv
import random
from pathlib import Path

HEADER = [
    "city",
    "neighborhood",
    "zip_code",
    "square_feet",
    "bedrooms",
    "bathrooms",
    "lot_size",
    "year_built",
    "school_rating",
    "distance_to_city_center_miles",
    "crime_index",
    "price",
]

MARKETS = [
    ("Austin", "North Loop", "78751", 345_000, 72_000),
    ("Austin", "Tarrytown", "78703", 570_000, 245_000),
    ("Austin", "Mueller", "78723", 430_000, 125_000),
    ("Denver", "Highland", "80211", 385_000, 95_000),
    ("Denver", "Cherry Creek", "80206", 560_000, 230_000),
    ("Denver", "Montbello", "80239", 285_000, -35_000),
    ("Seattle", "Ballard", "98107", 520_000, 115_000),
    ("Seattle", "Queen Anne", "98109", 670_000, 255_000),
    ("Seattle", "Rainier Valley", "98118", 430_000, 25_000),
    ("Phoenix", "Arcadia", "85018", 320_000, 105_000),
    ("Phoenix", "Roosevelt Row", "85004", 310_000, 75_000),
    ("Phoenix", "Ahwatukee", "85044", 290_000, 35_000),
    ("Atlanta", "Midtown", "30309", 330_000, 85_000),
    ("Atlanta", "Buckhead", "30305", 455_000, 180_000),
    ("Atlanta", "East Point", "30344", 210_000, -25_000),
]


def generate_rows(count: int, seed: int) -> list[dict[str, object]]:
    rng = random.Random(seed)
    rows = []
    for _ in range(count):
        city, neighborhood, zip_code, city_base, neighborhood_adjustment = rng.choice(MARKETS)
        square_feet = rng.randint(950, 4200)
        bedrooms = min(6, max(1, round(square_feet / 650 + rng.uniform(-0.7, 0.8))))
        bathrooms = round(min(5.5, max(1, bedrooms * 0.65 + rng.uniform(-0.2, 0.8))) * 2) / 2
        lot_size = round(rng.uniform(0.04, 0.35), 3)
        year_built = rng.randint(1950, 2025)
        school_rating = round(rng.uniform(5.6, 9.8), 1)
        distance = round(rng.uniform(0.8, 13.5), 1)
        crime_index = round(rng.uniform(12, 62), 1)

        age_penalty = max(0, 2026 - year_built) * 850
        price = (
            city_base
            + neighborhood_adjustment
            + square_feet * 215
            + bedrooms * 18_500
            + bathrooms * 31_000
            + lot_size * 310_000
            + school_rating * 44_000
            - distance * 16_000
            - crime_index * 4_200
            - age_penalty
            + rng.gauss(0, 22_000)
        )
        rows.append(
            {
                "city": city,
                "neighborhood": neighborhood,
                "zip_code": zip_code,
                "square_feet": square_feet,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "lot_size": lot_size,
                "year_built": year_built,
                "school_rating": school_rating,
                "distance_to_city_center_miles": distance,
                "crime_index": crime_index,
                "price": round(max(150_000, price), 2),
            }
        )
    return rows


def main() -> None:
    output_path = Path("data/generated_housing.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = generate_rows(count=750, seed=42)
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)


if __name__ == "__main__":
    main()
