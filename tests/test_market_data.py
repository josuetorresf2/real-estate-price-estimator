from pathlib import Path

from real_estate_price_estimator.market_data import (
    CensusHomeValueSignal,
    latest_zhvi_for_zip,
    market_calibrated_estimate,
)


def test_latest_zhvi_for_zip_reads_latest_non_empty_value(tmp_path: Path):
    csv_path = tmp_path / "zhvi.csv"
    csv_path.write_text(
        "RegionID,SizeRank,RegionName,RegionType,StateName,State,City,Metro,CountyName,2026-01-31,2026-02-28\n"
        "1,1,78751,zip,TX,TX,Austin,Austin-Round Rock,Travis,515000,525000\n"
        "2,2,30309,zip,GA,GA,Atlanta,Atlanta-Sandy Springs,Fulton,410000,\n",
        encoding="utf-8",
    )

    signal = latest_zhvi_for_zip("78751", csv_path)

    assert signal is not None
    assert signal.zip_code == "78751"
    assert signal.city == "Austin"
    assert signal.date == "2026-02-28"
    assert signal.typical_home_value == 525000


def test_market_calibrated_estimate_blends_model_with_zillow_signal(tmp_path: Path):
    csv_path = tmp_path / "zhvi.csv"
    csv_path.write_text(
        "RegionID,SizeRank,RegionName,RegionType,StateName,State,City,Metro,CountyName,2026-01-31\n"
        "1,1,78751,zip,TX,TX,Austin,Austin-Round Rock,Travis,500000\n",
        encoding="utf-8",
    )
    signal = latest_zhvi_for_zip("78751", csv_path)

    assert market_calibrated_estimate(1000000, signal) == 725000


def test_market_calibrated_estimate_blends_model_zillow_and_census(tmp_path: Path):
    csv_path = tmp_path / "zhvi.csv"
    csv_path.write_text(
        "RegionID,SizeRank,RegionName,RegionType,StateName,State,City,Metro,CountyName,2026-01-31\n"
        "1,1,78751,zip,TX,TX,Austin,Austin-Round Rock,Travis,600000\n",
        encoding="utf-8",
    )
    zillow_signal = latest_zhvi_for_zip("78751", csv_path)
    census_signal = CensusHomeValueSignal(
        zip_code="78751",
        name="ZCTA5 78751",
        year="2023",
        median_home_value=500000,
    )

    assert market_calibrated_estimate(1000000, zillow_signal, census_signal) == 745000
