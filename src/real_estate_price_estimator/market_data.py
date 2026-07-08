from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen, urlretrieve

ZHVI_ZIP_URL = "https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
CENSUS_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
CENSUS_ACS_URL = "https://api.census.gov/data/{year}/acs/acs5"
CENSUS_ACS_YEAR = "2023"


@dataclass(frozen=True)
class ZillowMarketSignal:
    zip_code: str
    city: str
    state: str
    county: str
    date: str
    typical_home_value: float
    source: str = "Zillow Research ZHVI, ZIP, all homes, smoothed and seasonally adjusted"


@dataclass(frozen=True)
class AddressLocation:
    city: str
    state: str
    zip_code: str
    matched_address: str
    latitude: float | None = None
    longitude: float | None = None
    source: str = "U.S. Census Geocoder"


@dataclass(frozen=True)
class CensusHomeValueSignal:
    zip_code: str
    name: str
    year: str
    median_home_value: float
    source: str = "U.S. Census ACS 5-year B25077 median owner-occupied home value"


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


def market_calibrated_estimate(
    model_estimate: float,
    signal: ZillowMarketSignal | None,
    census_signal: CensusHomeValueSignal | None = None,
) -> float:
    if signal is None and census_signal is None:
        return model_estimate
    if signal is not None and census_signal is not None:
        return model_estimate * 0.4 + signal.typical_home_value * 0.45 + census_signal.median_home_value * 0.15
    if signal is not None:
        return model_estimate * 0.45 + signal.typical_home_value * 0.55
    return model_estimate * 0.55 + census_signal.median_home_value * 0.45


def geocode_address(address: str, *, timeout: float = 8) -> AddressLocation | None:
    if not address.strip():
        return None
    query = urlencode(
        {
            "address": address,
            "benchmark": "Public_AR_Current",
            "format": "json",
        }
    )
    request = Request(
        f"{CENSUS_GEOCODER_URL}?{query}",
        headers={"User-Agent": "real-estate-price-estimator/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    matches = payload.get("result", {}).get("addressMatches", [])
    if not matches:
        return None
    match = matches[0]
    components = match.get("addressComponents", {})
    coordinates = match.get("coordinates", {})
    return AddressLocation(
        city=components.get("city", ""),
        state=components.get("state", ""),
        zip_code=components.get("zip", ""),
        matched_address=match.get("matchedAddress", ""),
        latitude=_as_float(coordinates.get("y")),
        longitude=_as_float(coordinates.get("x")),
    )


def census_home_value_for_zip(
    zip_code: str,
    *,
    api_key: str | None = None,
    year: str = CENSUS_ACS_YEAR,
    timeout: float = 8,
) -> CensusHomeValueSignal | None:
    key = api_key or os.getenv("CENSUS_API_KEY")
    if not key:
        return None

    normalized_zip = str(zip_code).zfill(5)
    query = urlencode(
        {
            "get": "NAME,B25077_001E",
            "for": f"zip code tabulation area:{normalized_zip}",
            "key": key,
        }
    )
    request = Request(
        f"{CENSUS_ACS_URL.format(year=year)}?{query}",
        headers={"User-Agent": "real-estate-price-estimator/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if len(payload) < 2:
        return None
    header, row = payload[0], payload[1]
    data = dict(zip(header, row))
    value = data.get("B25077_001E")
    if not value or value in {"-666666666", "-222222222"}:
        return None
    return CensusHomeValueSignal(
        zip_code=normalized_zip,
        name=data.get("NAME", ""),
        year=year,
        median_home_value=float(value),
    )


def _is_date_column(column: str) -> bool:
    parts = column.split("-")
    return len(parts) == 3 and all(part.isdigit() for part in parts)


def _as_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
