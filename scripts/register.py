"""Registreer het beste model uit de HPO-runs in de MLFlow Model Registry."""

import os
from pathlib import Path

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestRegressor

from scripts.train import load_splits
from scripts.utils import log_rmse_metrics

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI", "http://experiment-tracking:5000"
)
HPO_EXPERIMENT_NAME = "energy-forecast-hyperopt"
REGISTER_EXPERIMENT_NAME = "energy-forecast-best-models"
MODEL_NAME = "energy-forecast-model"
RF_PARAMS = [
    "n_estimators",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "random_state",
    "n_jobs",
]

DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "exports"


def train_and_log_model(params, x_train, y_train, x_val, y_val):
    """Hertrainen met gegeven params en model opslaan als MLFlow artifact."""
    mlflow.set_experiment(REGISTER_EXPERIMENT_NAME)

    with mlflow.start_run():
        # Zet params om naar int (want MLFlow slaat ze als strings op)
        clean_params = {k: int(v) for k, v in params.items() if k in RF_PARAMS}
        clean_params["n_jobs"] = -1
        clean_params["random_state"] = 42

        mlflow.log_params(clean_params)

        rf = RandomForestRegressor(**clean_params)
        rf.fit(x_train, y_train)
        y_pred = rf.predict(x_val)

        log_rmse_metrics(y_val, y_pred)

        # Model opslaan als artifact zodat we het later kunnen registreren
        mlflow.sklearn.log_model(rf, artifact_path="model")


def run_register_model(data_path, top_n=5):
    """Selecteer de top N HPO-runs, hertrainen, en registreer het beste model."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient()

    x_train, y_train, x_val, y_val = load_splits(data_path)

    # Haalt de top N runs op uit het HPO-experiment (gesorteerd op laagste RMSE)
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(
            f"Experiment '{HPO_EXPERIMENT_NAME}' niet gevonden. Voer eerst hpo.py uit."
        )

    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.rmse ASC"],
    )

    print(f"{len(runs)} runs geselecteerd uit '{HPO_EXPERIMENT_NAME}'")

    # Hertrainen van elk geselecteerd model en opslaan in het register-experiment
    for run in runs:
        train_and_log_model(run.data.params, x_train, y_train, x_val, y_val)

    # Hier kiest het de run met de laagste RMSE uit het register-experiment
    register_experiment = client.get_experiment_by_name(REGISTER_EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=register_experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.rmse ASC"],
    )[0]

    # Registreer het beste model in de MLFlow Model Registry
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)

    print(f"Model geregistreerd: '{MODEL_NAME}' versie {registered.version}")
    print(f"\tRun ID: {run_id}")
    print(f"\tRMSE: {best_run.data.metrics['rmse']:.4f} MW")


if __name__ == "__main__":
    print("...beste model registreren")
    run_register_model(DATASETS_DIR / "processed.parquet", top_n=5)
