"""Module voor het preprocessen van data."""

from pathlib import Path

import pandas as pd

DATASETS_DIR = Path(__file__).parent.parent.parent / "datasets"
OUTPUT_DIR = Path(__file__).parent.parent.parent / "datasets/exports"


def load_production(path):
    """Production dataset inladen."""
    df = pd.read_csv(path)
    # Een nieuwe kolom/feature maken van tijd data die omgezet is naar een datetime formaat
    df["tijd"] = pd.to_datetime(df["tijd"], utc=True)
    df = df.set_index("tijd").sort_index()
    # kWh converteren naar MW door het te delen door 1000
    df["solar_mw"] = df["elia zon kwh"] / 1000
    df["wind_mw"] = df["elia wind kwh"] / 1000
    return df[["solar_mw", "wind_mw"]]


def load_wind(path, start, end):
    """Wind dataset inladen met een start en end parameter."""
    df = pd.read_csv(path, na_values=["NULL"])
    df["tijdstip"] = pd.to_datetime(df["tijdstip"], utc=True)
    df = df.set_index("tijdstip").sort_index()
    df = df.loc[start:end]

    wind_cols = [
        "wind_ecmwf_2026",
        "wind_ukkel_2024",
        "wind_antwerpen_archive",
        "wind_kmi_2002",
    ]
    available = [c for c in wind_cols if c in df.columns]
    combined = df[available[0]]
    # Hier combineer ik de windkolommen op volgorde van voorkeur.
    for col in available[1:]:
        combined = combined.combine_first(df[col])
    # Een algemene kolom van wind snelheid
    df["wind_speed_kmh"] = combined
    return df[["wind_speed_kmh"]]


def load_sun(path, start, end):
    """Sun dataset inladen."""
    df = pd.read_csv(path)
    df["datum"] = pd.to_datetime(df["datum"]).dt.tz_localize("UTC")
    df = df.set_index("datum").sort_index()
    df = df.loc[start:end]

    rad_cols = ["open_meteo_radiation", "kaggle_radiation_avg", "kmi_radiation_avg"]
    available = [c for c in rad_cols if c in df.columns]
    combined = df[available[0]]
    for col in available[1:]:
        combined = combined.combine_first(df[col])
    df["solar_radiation_wm2"] = combined

    hourly_idx = pd.date_range(start=start, end=end, freq="h", tz="UTC")
    df = df[["solar_radiation_wm2"]].reindex(hourly_idx, method="ffill")
    return df


def add_time_features(df):
    """Deze functie voegt 4 kolommen toe aan de dataframe. Namelijk uur, maand, dag van de week en of het weekend is."""
    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)
    return df


def preprocess(output_path):
    production = load_production(DATASETS_DIR / "productie_combined.csv")

    # Hier definieer ik een begin en eind datum zodat ik het als filter kan gebruiken
    start = production.index.min()
    end = production.index.max()

    # Hier laad ik de datasets in
    wind = load_wind(DATASETS_DIR / "v_wind_alles_compleet.csv", start, end)
    sun = load_sun(DATASETS_DIR / "sun_combined.csv", start, end)

    df = production.join(wind, how="left").join(sun, how="left")
    # Hier voeg ik de 4 kolommen zie functie beschrijving boven
    df = add_time_features(df)
    # Hier verwijder ik alle NA waardes
    df = df.dropna(subset=["solar_mw", "wind_mw"])

    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df.index.min()} - {df.index.max()}")
    print(f"Missing wind_speed_kmh: {df['wind_speed_kmh'].isna().sum()}")
    print(f"Missing solar_radiation_wm2: {df['solar_radiation_wm2'].isna().sum()}")

    # Exporteer ik de gepreprocessed dataset
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path)
        print(f"Saved to {output_path}")

    return df


if __name__ == "__main__":
    result = preprocess(output_path=OUTPUT_DIR / "processed.parquet")
    print(result.describe().to_string())
