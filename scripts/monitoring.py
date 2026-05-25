"""Monitoring module: sla metrics op in PostgreSQL en genereer Evidently rapporten."""

import os
from pathlib import Path

import pandas as pd
import sqlalchemy
from evidently import ColumnMapping
from evidently.metric_preset import RegressionPreset
from evidently.report import Report
from prefect.deployments import run_deployment

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@database:5432/postgres"
)
REPORTS_DIR = Path("/batch-data/evidently-reports")

# RMSE drempelwaarde in MW. Boven deze waarde wordt hertraining getriggerd
RMSE_DREMPEL = 50.0


def maak_tabel_aan(engine):
    """Maakt de monitoring_metrics tabel aan als die nog niet bestaat."""
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS monitoring_metrics (
                id SERIAL PRIMARY KEY,
                run_date DATE UNIQUE NOT NULL,
                rmse_solar_mw FLOAT,
                rmse_wind_mw FLOAT,
                rmse FLOAT,
                n_uur INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            )"""))


def sla_metrics_op(metrics, run_date):
    """Slaagt RMSE metrics op in PostgreSQL voor Grafana."""
    engine = sqlalchemy.create_engine(DATABASE_URL)
    maak_tabel_aan(engine)
    with engine.begin() as conn:
        conn.execute(
            sqlalchemy.text("""
                INSERT INTO monitoring_metrics
                    (run_date, rmse_solar_mw, rmse_wind_mw, rmse, n_uur)
                VALUES
                    (:run_date, :rmse_solar_mw, :rmse_wind_mw, :rmse, :n_uur)
                ON CONFLICT (run_date) DO UPDATE SET
                    rmse_solar_mw = EXCLUDED.rmse_solar_mw,
                    rmse_wind_mw = EXCLUDED.rmse_wind_mw,
                    rmse = EXCLUDED.rmse,
                    n_uur = EXCLUDED.n_uur"""),
            {
                "run_date": run_date,
                "rmse_solar_mw": metrics["rmse_solar_mw"],
                "rmse_wind_mw": metrics["rmse_wind_mw"],
                "rmse": metrics["rmse"],
                "n_uur": metrics["n_uur"],
            },
        )
    print(f"Metrics opgeslagen in PostgreSQL voor {run_date}")


def genereer_evidently_rapport(df_pred, df_actuals, run_date):
    """Genereert Evidently regressie rapporten als HTML en slaat ze op per doelvariabele."""
    overlap = df_pred.index.intersection(df_actuals.index)
    if len(overlap) == 0:
        print("Geen overlap voor Evidently rapport overgeslagen.")
        return

    df_p = df_pred.loc[overlap]
    df_a = df_actuals.loc[overlap]

    df_combined = pd.DataFrame(
        {
            "solar_mw": df_a["solar_mw"].values,
            "solar_mw_pred": df_p["solar_mw_pred"].values,
            "wind_mw": df_a["wind_mw"].values,
            "wind_mw_pred": df_p["wind_mw_pred"].values,
        }
    )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for target, prediction in [
        ("solar_mw", "solar_mw_pred"),
        ("wind_mw", "wind_mw_pred"),
    ]:
        column_mapping = ColumnMapping(target=target, prediction=prediction)
        report = Report(metrics=[RegressionPreset()])
        report.run(
            reference_data=None, current_data=df_combined, column_mapping=column_mapping
        )
        report_path = REPORTS_DIR / f"{run_date}_{target}.html"
        report.save_html(str(report_path))

    print(f"Evidently rapporten opgeslagen voor {run_date}")


def sla_vergelijking_op(df_pred, df_actuals, run_date):
    """Sla uurlijkse voorspellingen vs actuals op in PostgreSQL voor Grafana."""
    overlap = df_pred.index.intersection(df_actuals.index)
    if len(overlap) == 0:
        return

    engine = sqlalchemy.create_engine(DATABASE_URL)
    with engine.begin() as conn:
        conn.execute(sqlalchemy.text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                tijd TIMESTAMPTZ NOT NULL,
                run_date DATE NOT NULL,
                solar_mw_pred FLOAT,
                wind_mw_pred FLOAT,
                solar_mw_actual FLOAT,
                wind_mw_actual FLOAT,
                UNIQUE (tijd, run_date)
            )
        """))
        for tijd in overlap:
            conn.execute(
                sqlalchemy.text("""
                    INSERT INTO predictions
                        (tijd, run_date, solar_mw_pred, wind_mw_pred, solar_mw_actual, wind_mw_actual)
                    VALUES
                        (:tijd, :run_date, :solar_mw_pred, :wind_mw_pred, :solar_mw_actual, :wind_mw_actual)
                    ON CONFLICT (tijd, run_date) DO UPDATE SET
                        solar_mw_pred = EXCLUDED.solar_mw_pred,
                        wind_mw_pred = EXCLUDED.wind_mw_pred,
                        solar_mw_actual = EXCLUDED.solar_mw_actual,
                        wind_mw_actual = EXCLUDED.wind_mw_actual
                """),
                {
                    "tijd": tijd.isoformat(),
                    "run_date": str(run_date),
                    "solar_mw_pred": float(df_pred.loc[tijd, "solar_mw_pred"]),
                    "wind_mw_pred": float(df_pred.loc[tijd, "wind_mw_pred"]),
                    "solar_mw_actual": float(df_actuals.loc[tijd, "solar_mw"]),
                    "wind_mw_actual": float(df_actuals.loc[tijd, "wind_mw"]),
                },
            )
    print(f"Vergelijking opgeslagen in PostgreSQL voor {run_date}: {len(overlap)} uur")


def controleer_hertraining(metrics):
    """Trigger hertraining als RMSE de drempelwaarde overschrijdt."""
    if not metrics:
        return

    rmse = metrics.get("rmse", 0)
    if rmse > RMSE_DREMPEL:
        print(f"RMSE {rmse:.4f} MW > drempel {RMSE_DREMPEL} MW hertraining starten")
        run_deployment("training-pipeline/training-pipeline", timeout=0)
        print("Hertraining getriggerd via Prefect")
    else:
        print(f"RMSE {rmse:.4f} MW <= drempel {RMSE_DREMPEL} MW geen hertraining nodig")
