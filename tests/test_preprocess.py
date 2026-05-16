"""Tests voor de preprocess module."""

import pandas as pd
import pytest

from scripts.data.preprocess import (
    add_time_features,
    load_production,
    load_sun,
    load_wind,
)


def create_production_csv_sample(tmp_path):
    csv = tmp_path / "productie.csv"
    csv.write_text(
        "tijd,vlaanderen zon kwh,vlaanderen wind kwh,elia zon kwh,elia wind kwh\n"
        "2025-03-01 00:00:00+00:00,0.0,314185.0,0.0,112000.0\n"
        "2025-03-01 01:00:00+00:00,0.0,308580.0,500.0,91000.0\n"
        "2025-03-01 02:00:00+00:00,100.0,259495.0,1000.0,79000.0\n"
    )
    return csv


def create_wind_csv_sample(tmp_path):
    csv = tmp_path / "wind.csv"
    csv.write_text(
        '"tijdstip","wind_ecmwf_2026","wind_kmi_2002"\n'
        '"2025-03-01 00:00:00+00","12.5",NULL\n'
        '"2025-03-01 01:00:00+00","13.1",NULL\n'
        '"2025-03-01 02:00:00+00","11.8",NULL\n'
    )
    return csv


def create_wind_csv_fallback_sample(tmp_path):
    """Winddata waarbij ecmwf_2026 NULL is maar kmi_2002 een waarde heeft."""
    csv = tmp_path / "wind_fallback.csv"
    csv.write_text(
        '"tijdstip","wind_ecmwf_2026","wind_kmi_2002"\n'
        '"2025-03-01 00:00:00+00",NULL,"8.4"\n'
        '"2025-03-01 01:00:00+00",NULL,"9.1"\n'
        '"2025-03-01 02:00:00+00",NULL,"7.3"\n'
    )
    return csv


def create_sun_csv_sample(tmp_path):
    csv = tmp_path / "sun.csv"
    csv.write_text(
        "id,datum,open_meteo_radiation,kmi_radiation_avg,kaggle_radiation_avg\n"
        "1,2025-03-01 00:00:00,150.0,,\n"
        "2,2025-03-02 00:00:00,160.0,,\n"
    )
    return csv


def create_sun_csv_fallback_sample(tmp_path):
    """Zondata waarbij open_meteo_radiation NULL is maar kaggle_radiation_avg een waarde heeft."""
    csv = tmp_path / "sun_fallback.csv"
    csv.write_text(
        "id,datum,open_meteo_radiation,kmi_radiation_avg,kaggle_radiation_avg\n"
        "1,2025-03-01 00:00:00,,, 200.0\n"
        "2,2025-03-02 00:00:00,,,210.0\n"
    )
    return csv


def test_load_production(tmp_path):
    csv = create_production_csv_sample(tmp_path)
    df = load_production(csv)
    assert "solar_mw" in df.columns
    assert "wind_mw" in df.columns
    # kWh / 1000 = MW
    assert df["wind_mw"].iloc[0] == pytest.approx(112.0)
    # Hiermee check ik of de tijdzone juist ingesteld
    assert df.index.tz is not None


def test_load_wind(tmp_path):
    csv = create_wind_csv_sample(tmp_path)
    start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2025-03-01 02:00:00", tz="UTC")
    df = load_wind(csv, start, end)
    assert "wind_speed_kmh" in df.columns
    assert len(df) == 3
    assert df["wind_speed_kmh"].iloc[0] == pytest.approx(12.5)


def test_load_wind_coalesce_fallback(tmp_path):
    """Hier test ik of de fallback werkt: als ecmwf_2026 NULL is moet kmi_2002 gebruikt worden"""
    csv = create_wind_csv_fallback_sample(tmp_path)
    start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2025-03-01 02:00:00", tz="UTC")
    df = load_wind(csv, start, end)
    assert df["wind_speed_kmh"].isna().sum() == 0
    assert df["wind_speed_kmh"].iloc[0] == pytest.approx(8.4)


def test_load_sun_expands_to_hourly(tmp_path):
    csv = create_sun_csv_sample(tmp_path)
    start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2025-03-01 05:00:00", tz="UTC")
    df = load_sun(csv, start, end)
    assert len(df) == 6
    assert (df["solar_radiation_wm2"] == 150.0).all()


def test_load_sun_coalesce_fallback(tmp_path):
    """Hier test ik of de fallback werkt: als open_meteo_radiation NULL is moet kaggle_radiation_avg gebruikt worden"""
    csv = create_sun_csv_fallback_sample(tmp_path)
    start = pd.Timestamp("2025-03-01 00:00:00", tz="UTC")
    end = pd.Timestamp("2025-03-01 05:00:00", tz="UTC")
    df = load_sun(csv, start, end)
    assert df["solar_radiation_wm2"].isna().sum() == 0
    assert df["solar_radiation_wm2"].iloc[0] == pytest.approx(200.0)


def test_add_time_features():
    idx = pd.date_range("2025-03-01", periods=3, freq="h", tz="UTC")
    df = pd.DataFrame({"solar_mw": [0.0, 1.0, 2.0]}, index=idx)
    df = add_time_features(df)
    assert "hour" in df.columns
    assert "month" in df.columns
    assert "dayofweek" in df.columns
    assert "is_weekend" in df.columns
    assert df["month"].iloc[0] == 3
    assert df["hour"].iloc[0] == 0
    assert df["is_weekend"].iloc[0] == 1
