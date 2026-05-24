"""Batch pipeline"""

import threading
import time
from pathlib import Path

import pandas as pd
import requests
from prefect import flow, task
from prefect.deployments import run_deployment
from sklearn.metrics import root_mean_squared_error

from scripts.models.load import load_model
from scripts.monitoring import controleer_hertraining as check_rmse
from scripts.monitoring import (
    genereer_evidently_rapport,
    sla_metrics_op,
    sla_vergelijking_op,
)
from scripts.train import FEATURES
from scripts.utils import haal_records_op

# Coördinaten Antwerpen
LAT = 51.2194
LON = 4.4025

OUTPUT_DIR = Path("/batch-data/predictions")


@task(
    name="laad-ecmwf-voorspelling", log_prints=True, retries=3, retry_delay_seconds=30
)
def laad_ecmwf_voorspelling():
    """Haalt de volgende 24 uur weersdata op via de Open-Meteo ECMWF forecast API."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "wind_speed_10m,shortwave_radiation",
        "forecast_days": 2,
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(
        {
            "wind_speed_kmh": data["hourly"]["wind_speed_10m"],
            "solar_radiation_wm2": data["hourly"]["shortwave_radiation"],
        },
        index=pd.to_datetime(data["hourly"]["time"], utc=True),
    )
    df.index.name = "tijd"

    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    nu = pd.Timestamp.now(tz="UTC").floor("h")
    df = df[df.index >= nu].head(24)

    print(f"ECMWF forecast opgehaald: {len(df)} uur vanaf {df.index.min()}")
    return df


@task(name="laad-ecmwf-gisteren", log_prints=True, retries=3, retry_delay_seconds=30)
def laad_ecmwf_gisteren():
    """Haalt de weersdata van gisteren op via de Open-Meteo forecast API met past_days=1."""
    gisteren = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).date()

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": "wind_speed_10m,shortwave_radiation",
        "past_days": 1,
        "forecast_days": 1,
        "timezone": "UTC",
        "wind_speed_unit": "kmh",
    }

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    df = pd.DataFrame(
        {
            "wind_speed_kmh": data["hourly"]["wind_speed_10m"],
            "solar_radiation_wm2": data["hourly"]["shortwave_radiation"],
        },
        index=pd.to_datetime(data["hourly"]["time"], utc=True),
    )
    df.index.name = "tijd"

    # Filter enkel gisteren
    df = df[df.index.date == gisteren]

    df["hour"] = df.index.hour
    df["month"] = df.index.month
    df["dayofweek"] = df.index.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    print(f"Historische weersdata opgehaald: {len(df)} uur voor {gisteren}")
    return df


@task(name="laad-elia-actuals", log_prints=True, retries=3, retry_delay_seconds=30)
def laad_elia_actuals():
    """Haalt de werkelijke productiedata van gisteren op via de Elia Open Data API."""
    gisteren = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).date()
    start = f"{gisteren}T00:00:00Z"
    einde = f"{gisteren}T23:59:00Z"

    # Zonne energie filteren op regio Flanders
    records_zon = haal_records_op(
        "ods032", start, einde, extra_filter=' AND region="Flanders"'
    )
    df_zon = pd.DataFrame(
        [
            {
                "tijd": pd.Timestamp(r["datetime"], tz="UTC"),
                "solar_mw": r.get("measured", 0) or 0,
            }
            for r in records_zon
        ]
    )
    df_zon = (
        df_zon.groupby("tijd")["solar_mw"]
        .sum()
        .resample("h")
        .sum()
        .rename("solar_mw")
        .to_frame()
    )

    # Windenergie filteren op regio flanders
    records_wind = haal_records_op(
        "ods031",
        start,
        einde,
        extra_filter=' AND region="Flanders" AND offshoreonshore="Onshore"',
    )
    df_wind = pd.DataFrame(
        [
            {
                "tijd": pd.Timestamp(r["datetime"], tz="UTC"),
                "wind_mw": r.get("measured", 0) or 0,
            }
            for r in records_wind
        ]
    )
    df_wind = (
        df_wind.groupby("tijd")["wind_mw"]
        .sum()
        .resample("h")
        .sum()
        .rename("wind_mw")
        .to_frame()
    )

    df_actuals = df_zon.join(df_wind, how="inner")
    df_actuals.index.name = "tijd"

    print(f"Elia actuals opgehaald: {len(df_actuals)} uur voor {gisteren}")
    return df_actuals


@task(name="run-predictie", log_prints=True)
def run_predictie(model, df_forecast):
    """Voert model predictie uit op de weersdata."""
    x = df_forecast[FEATURES].values
    predictions = model.predict(x)

    df_result = df_forecast.copy()
    df_result["solar_mw_pred"] = predictions[:, 0]
    df_result["wind_mw_pred"] = predictions[:, 1]

    print(f"Predictie voltooid: {len(df_result)} voorspellingen gegenereerd")
    return df_result


@task(name="sla-voorspellingen-op", log_prints=True)
def sla_voorspellingen_op(df_result):
    """Sla voorspellingen op als parquet bestand."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_date = df_result.index.normalize().min().strftime("%Y-%m-%d")
    output_path = OUTPUT_DIR / f"{run_date}.parquet"
    df_result.to_parquet(output_path)
    print(f"Voorspellingen opgeslagen: {output_path}")
    return output_path


@task(name="vergelijk-met-actuals", log_prints=True)
def vergelijk_met_actuals(df_pred, df_actuals):
    """Vergelijkt de voorspellingen met de werkelijke Elia-productiedata en berekent RMSE."""
    overlap = df_pred.index.intersection(df_actuals.index)

    if len(overlap) == 0:
        print("Geen overlap tussen voorspellingen en actuals.")
        return {}

    df_p = df_pred.loc[overlap]
    df_a = df_actuals.loc[overlap]

    rmse_solar = root_mean_squared_error(df_a["solar_mw"], df_p["solar_mw_pred"])
    rmse_wind = root_mean_squared_error(df_a["wind_mw"], df_p["wind_mw_pred"])
    rmse_mean = (rmse_solar + rmse_wind) / 2

    metrics = {
        "rmse_solar_mw": round(rmse_solar, 4),
        "rmse_wind_mw": round(rmse_wind, 4),
        "rmse": round(rmse_mean, 4),
        "n_uur": len(overlap),
    }

    print(f"RMSE solar:{rmse_solar:.4f} MW")
    print(f"RMSE wind: {rmse_wind:.4f} MW")
    print(f"RMSE gemiddeld: {rmse_mean:.4f} MW")
    print(f"Geëvalueerd op: {metrics['n_uur']} uur")

    return metrics


@task(name="rapporteer-monitoring", log_prints=True)
def rapporteer_monitoring(metrics, df_pred, df_actuals):
    """Slaagt metrics op in PostgreSQL en genereert een Evidently rapport."""
    if not metrics:
        print("Geen metrics beschikbaar dus monitoring overgeslagen.")
        return
    run_date = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).date()
    sla_metrics_op(metrics, run_date)
    sla_vergelijking_op(df_pred, df_actuals, run_date)
    genereer_evidently_rapport(df_pred, df_actuals, run_date)


@task(name="controleer-hertraining", log_prints=True)
def controleer_hertraining(metrics):
    """Triggert hertraining als RMSE de drempelwaarde overschrijdt."""
    if not metrics:
        return
    check_rmse(metrics)


@flow(name="batch-pipeline", log_prints=True)
def batch_pipeline():
    """Dagelijkse batch pipeline."""
    model = load_model()

    # Voorspelling voor de volgende 24 uur (Open-Meteo forecast)
    df_forecast = laad_ecmwf_voorspelling()
    df_pred_toekomst = run_predictie(model, df_forecast)
    sla_voorspellingen_op(df_pred_toekomst)

    # Evaluatie op gisteren (Open-Meteo archive vs Elia actuals)
    df_gisteren = laad_ecmwf_gisteren()
    df_pred_gisteren = run_predictie(model, df_gisteren)
    df_actuals = laad_elia_actuals()
    metrics = vergelijk_met_actuals(df_pred_gisteren, df_actuals)

    # Monitoring + hertraining check
    rapporteer_monitoring(metrics, df_pred_gisteren, df_actuals)
    controleer_hertraining(metrics)


def trigger_eerste_run():
    """Triggert een directe Prefect run zodra de deployment geregistreerd is."""
    time.sleep(10)
    try:
        run_deployment("batch-pipeline/dagelijkse-batch", timeout=0)
        print("Initiële batch run getriggerd via Prefect.")
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"Kon initiële run niet triggeren: {exc}")


if __name__ == "__main__":
    thread = threading.Thread(target=trigger_eerste_run, daemon=True)
    thread.start()
    batch_pipeline.serve(
        name="dagelijkse-batch",
        cron="0 6 * * *",
    )
