from __future__ import annotations

import csv
import json
import os
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen, urlretrieve

ZHVI_ZIP_URL = "https://files.zillowstatic.com/research/public_csvs/zhvi/Zip_zhvi_uc_sfrcondo_tier_0.33_0.67_sm_sa_month.csv"
CENSUS_GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/locations/onelineaddress"
CENSUS_ACS_URL = "https://api.census.gov/data/{year}/acs/acs5"
CENSUS_ACS_YEAR = "2023"
ATTOM_PROPERTY_URL = "https://api.gateway.attomdata.com/propertyapi/v1.0.0/property/expandedprofile"
GEOAPIFY_AUTOCOMPLETE_URL = "https://api.geoapify.com/v1/geocode/autocomplete"
NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
MERCADOLIBRE_SEARCH_URL = "https://api.mercadolibre.com/sites/{site_id}/search"
WORLD_BANK_URL = "https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator}"
COUNTRY_CODES = {
    "United States": "us",
    "Ecuador": "ec",
    "Brazil": "br",
    "Peru": "pe",
    "Colombia": "co",
    "Chile": "cl",
}
MERCADOLIBRE_SITES = {
    "Ecuador": "MEC",
    "Brazil": "MLB",
    "Peru": "MPE",
    "Colombia": "MCO",
    "Chile": "MLC",
}
WORLD_BANK_COUNTRIES = {
    "Ecuador": "ECU",
    "Brazil": "BRA",
    "Peru": "PER",
    "Colombia": "COL",
    "Chile": "CHL",
}
WORLD_BANK_INDICATORS = {
    "gdp_per_capita_usd": "NY.GDP.PCAP.CD",
    "urban_population_pct": "SP.URB.TOTL.IN.ZS",
}
CITY_CENTERS = {
    ("AUSTIN", "TX"): (30.2672, -97.7431),
    ("DENVER", "CO"): (39.7392, -104.9903),
    ("COLORADO SPRINGS", "CO"): (38.8339, -104.8214),
    ("ATLANTA", "GA"): (33.7490, -84.3880),
    ("WASHINGTON", "DC"): (38.9072, -77.0369),
    ("QUITO", "PICHINCHA"): (-0.1807, -78.4678),
    ("GUAYAQUIL", "GUAYAS"): (-2.1894, -79.8891),
    ("SAO PAULO", "SP"): (-23.5558, -46.6396),
    ("SAO PAULO", "SÃO PAULO"): (-23.5558, -46.6396),
    ("RIO DE JANEIRO", "RJ"): (-22.9068, -43.1729),
    ("LIMA", "LIMA"): (-12.0464, -77.0428),
    ("BOGOTA", "BOGOTA"): (4.7110, -74.0721),
    ("BOGOTÁ", "BOGOTÁ D.C."): (4.7110, -74.0721),
    ("MEDELLIN", "ANTIOQUIA"): (6.2442, -75.5812),
    ("MEDELLÍN", "ANTIOQUIA"): (6.2442, -75.5812),
    ("SANTIAGO", "REGION METROPOLITANA"): (-33.4489, -70.6693),
    ("SANTIAGO", "REGIÓN METROPOLITANA"): (-33.4489, -70.6693),
}


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


@dataclass(frozen=True)
class PropertyFacts:
    square_feet: float | None = None
    bedrooms: float | None = None
    bathrooms: float | None = None
    lot_size: float | None = None
    year_built: int | None = None
    source: str = "ATTOM Property API"

    def as_form_values(self) -> dict[str, str]:
        values = {
            "square_feet": _format_number(self.square_feet),
            "bedrooms": _format_number(self.bedrooms),
            "bathrooms": _format_number(self.bathrooms),
            "lot_size": _format_decimal(self.lot_size),
            "year_built": "" if self.year_built is None else str(self.year_built),
        }
        return {key: value for key, value in values.items() if value}


@dataclass(frozen=True)
class RegionalListingSignal:
    country: str
    source: str
    listing_count: int
    average_price: float | None
    currency: str
    sample_titles: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegionalMacroSignal:
    country: str
    source: str
    values: dict[str, float]
    years: dict[str, str]


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
    matches = geocode_address_matches(address, timeout=timeout)
    return matches[0] if matches else None


def geocode_address_global(address: str, *, country: str = "United States", timeout: float = 8) -> AddressLocation | None:
    if country == "United States":
        return geocode_address(address, timeout=timeout)
    matches = nominatim_address_matches(address, country=country, timeout=timeout)
    return matches[0] if matches else None


def geocode_address_matches(address: str, *, timeout: float = 8) -> list[AddressLocation]:
    if not address.strip():
        return []
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
        return []
    return [_address_location_from_match(match) for match in matches]


def nominatim_address_matches(address: str, *, country: str = "United States", timeout: float = 8) -> list[AddressLocation]:
    if not address.strip():
        return []
    query = urlencode(
        {
            "q": f"{address}, {country}",
            "format": "jsonv2",
            "addressdetails": "1",
            "limit": "5",
            "countrycodes": COUNTRY_CODES.get(country, ""),
        }
    )
    request = Request(
        f"{NOMINATIM_SEARCH_URL}?{query}",
        headers={"User-Agent": "real-estate-price-estimator/0.1 (portfolio project)"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return [_address_location_from_nominatim(item) for item in payload]


def geoapify_address_suggestions(
    query_text: str,
    *,
    country: str = "United States",
    api_key: str | None = None,
    limit: int = 5,
    timeout: float = 8,
) -> list[AddressLocation]:
    key = api_key or os.getenv("GEOAPIFY_API_KEY")
    if not key or not query_text.strip():
        return []
    query = urlencode(
        {
            "text": query_text,
            "filter": f"countrycode:{COUNTRY_CODES.get(country, 'us')}",
            "limit": str(limit),
            "format": "json",
            "apiKey": key,
        }
    )
    request = Request(
        f"{GEOAPIFY_AUTOCOMPLETE_URL}?{query}",
        headers={"User-Agent": "real-estate-price-estimator/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = payload.get("results", [])
    return [_address_location_from_geoapify(result) for result in results]


def geoapify_neighborhood_for_address(address: str, *, country: str = "United States", api_key: str | None = None, timeout: float = 8) -> str | None:
    key = api_key or os.getenv("GEOAPIFY_API_KEY")
    if not key or not address.strip():
        return None
    query = urlencode(
        {
            "text": address,
            "filter": f"countrycode:{COUNTRY_CODES.get(country, 'us')}",
            "limit": "1",
            "format": "json",
            "apiKey": key,
        }
    )
    request = Request(
        f"{GEOAPIFY_AUTOCOMPLETE_URL}?{query}",
        headers={"User-Agent": "real-estate-price-estimator/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = payload.get("results", [])
    if not results:
        return None
    return _neighborhood_from_geoapify(results[0])


def regional_listing_signal(query_text: str, *, country: str, timeout: float = 8) -> RegionalListingSignal | None:
    site_id = MERCADOLIBRE_SITES.get(country)
    if not site_id or not query_text.strip():
        return None
    query = urlencode({"q": f"inmueble {query_text}", "limit": "12"})
    request = Request(
        f"{MERCADOLIBRE_SEARCH_URL.format(site_id=site_id)}?{query}",
        headers={"User-Agent": "real-estate-price-estimator/0.1"},
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    results = payload.get("results", [])
    prices = [_as_float(item.get("price")) for item in results]
    prices = [price for price in prices if price is not None and price > 0]
    currency = ""
    for item in results:
        currency = item.get("currency_id") or currency
        if currency:
            break
    average_price = sum(prices) / len(prices) if prices else None
    sample_titles = tuple(str(item.get("title", "")) for item in results[:3] if item.get("title"))
    return RegionalListingSignal(
        country=country,
        source=f"Mercado Libre {site_id} public search",
        listing_count=len(results),
        average_price=average_price,
        currency=currency,
        sample_titles=sample_titles,
    )


def regional_macro_signal(country: str, *, timeout: float = 8) -> RegionalMacroSignal | None:
    country_code = WORLD_BANK_COUNTRIES.get(country)
    if not country_code:
        return None
    values: dict[str, float] = {}
    years: dict[str, str] = {}
    for name, indicator in WORLD_BANK_INDICATORS.items():
        query = urlencode({"format": "json", "per_page": "6"})
        request = Request(
            f"{WORLD_BANK_URL.format(country_code=country_code, indicator=indicator)}?{query}",
            headers={"User-Agent": "real-estate-price-estimator/0.1"},
        )
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        for row in rows:
            value = _as_float(row.get("value"))
            if value is not None:
                values[name] = value
                years[name] = str(row.get("date", ""))
                break
    if not values:
        return None
    return RegionalMacroSignal(
        country=country,
        source="World Bank public indicators",
        values=values,
        years=years,
    )


def property_facts_for_address(address: str, *, api_key: str | None = None, timeout: float = 8) -> PropertyFacts | None:
    key = api_key or os.getenv("ATTOM_API_KEY")
    if not key or not address.strip():
        return None

    query = urlencode({"address1": address})
    request = Request(
        f"{ATTOM_PROPERTY_URL}?{query}",
        headers={
            "Accept": "application/json",
            "User-Agent": "real-estate-price-estimator/0.1",
            "apikey": key,
        },
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))

    properties = payload.get("property") or []
    if not properties:
        return None
    record = properties[0]
    facts = PropertyFacts(
        square_feet=_first_number(
            record,
            (
                ("building", "size", "livingsize"),
                ("building", "size", "universalsize"),
                ("building", "size", "bldgsize"),
            ),
        ),
        bedrooms=_first_number(record, (("building", "rooms", "beds"), ("building", "rooms", "bedrooms"))),
        bathrooms=_first_number(
            record,
            (
                ("building", "rooms", "bathstotal"),
                ("building", "rooms", "bathsfull"),
                ("building", "rooms", "bathrooms"),
            ),
        ),
        lot_size=_lot_size_acres(record),
        year_built=_first_int(record, (("summary", "yearbuilt"), ("building", "summary", "yearbuilt"))),
    )
    return facts if facts.as_form_values() else None


def distance_to_city_center_miles(location: AddressLocation) -> float | None:
    if location.latitude is None or location.longitude is None or not location.city or not location.state:
        return None
    center = CITY_CENTERS.get((location.city.upper(), location.state.upper()))
    if center is None:
        center_location = geocode_address(f"{location.city}, {location.state}")
        if center_location is None or center_location.latitude is None or center_location.longitude is None:
            return None
        center = (center_location.latitude, center_location.longitude)
    return _haversine_miles(location.latitude, location.longitude, center[0], center[1])


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


def _address_location_from_match(match: dict[str, object]) -> AddressLocation:
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


def _address_location_from_geoapify(result: dict[str, object]) -> AddressLocation:
    return AddressLocation(
        city=str(result.get("city") or result.get("county") or ""),
        state=str(result.get("state_code") or result.get("state") or ""),
        zip_code=str(result.get("postcode") or ""),
        matched_address=str(result.get("formatted") or ""),
        latitude=_as_float(result.get("lat")),
        longitude=_as_float(result.get("lon")),
        source="Geoapify Address Autocomplete",
    )


def _address_location_from_nominatim(result: dict[str, object]) -> AddressLocation:
    address = result.get("address", {})
    city = (
        address.get("city")
        or address.get("town")
        or address.get("village")
        or address.get("municipality")
        or address.get("county")
        or ""
    )
    state = address.get("state") or address.get("region") or address.get("province") or ""
    zip_code = address.get("postcode") or ""
    return AddressLocation(
        city=str(city),
        state=str(state),
        zip_code=str(zip_code),
        matched_address=str(result.get("display_name") or ""),
        latitude=_as_float(result.get("lat")),
        longitude=_as_float(result.get("lon")),
        source="OpenStreetMap Nominatim",
    )


def _neighborhood_from_geoapify(result: dict[str, object]) -> str | None:
    for key in ("neighbourhood", "neighborhood", "suburb", "district", "quarter"):
        value = result.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _get_path(payload: dict[str, object], path: tuple[str, ...]) -> object | None:
    current: object = payload
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _first_number(payload: dict[str, object], paths: tuple[tuple[str, ...], ...]) -> float | None:
    for path in paths:
        value = _as_float(_get_path(payload, path))
        if value is not None and value > 0:
            return value
    return None


def _first_int(payload: dict[str, object], paths: tuple[tuple[str, ...], ...]) -> int | None:
    value = _first_number(payload, paths)
    if value is None:
        return None
    return int(value)


def _lot_size_acres(record: dict[str, object]) -> float | None:
    acres = _first_number(record, (("lot", "lotsize2"), ("lot", "lotSizeAcres")))
    if acres is not None:
        return acres
    square_feet = _first_number(record, (("lot", "lotsize1"), ("lot", "lotSizeSquareFeet")))
    if square_feet is None:
        return None
    return square_feet / 43560


def _format_number(value: float | None) -> str:
    if value is None:
        return ""
    return str(int(value)) if float(value).is_integer() else str(value)


def _format_decimal(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_miles = 3958.7613
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    value = math.sin(delta_lat / 2) ** 2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2) ** 2
    return radius_miles * 2 * math.atan2(math.sqrt(value), math.sqrt(1 - value))
