from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

ZHVI_ZIP_URL = "https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"


@dataclass(frozen=True)
class ZillowMarketSignal:
    zip_code: str
    city: str
    state: str
    county: str
    date: str
    typical_home_value: float
    source: str = "Zillow Research ZHVI, ZIP, all homes, smoothed and seasonally adjusted"


def ensure_zhvi_csv(cache_path: Path, *, source_url: str = ZHVI_ZIP_URL) -> Path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists() or cache_path.stat().st_size == 0:
        urlretrieve(source_url, cache_path)
    return cache_path


def latest_zhvi_for_zip(zip_code: str, csv_path: Path) -> ZillowMarketSignal | None:
    normalized_zip = str(zip_code).zfill(5)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        date_columns = [column for column in reader.fieldnames or [] if _is_date_column(column)]
        for row in reader:
            if str(row.get("RegionName", "")).zfill(5) != normalized_zip:
                continue
            for date in reversed(date_columns):
                raw_value = row.get(date, "")
                if raw_value:
                    return ZillowMarketSignal(
                        zip_code=normalized_zip,
                        city=row.get("City", ""),
                        state=row.get("State", ""),
                        county=row.get("CountyName", ""),
                        date=date,
                        typical_home_value=float(raw_value),
                    )
    return None


def market_calibrated_estimate(model_estimate: float, signal: ZillowMarketSignal | None) -> float:
    if signal is None:
        return model_estimate
    return model_estimate * 0.45 + signal.typical_home_value * 0.55


def _is_date_column(column: str) -> bool:
    parts = column.split("-")
    return len(parts) == 3 and all(part.isdigit() for part in parts)
