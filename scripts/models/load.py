"""Hulpfuncties voor het laden van het geregistreerde model uit MLFlow."""

import mlflow
import requests

MLFLOW_TRACKING_URI = "http://experiment-tracking:5000"
MODEL_NAME = "energy-forecast-model"


def get_latest_version(model_name):
    """Haal het versienummer van het meest recente model op uit de registry."""
    response = requests.post(
        f"{MLFLOW_TRACKING_URI}/api/2.0/mlflow/registered-models/get-latest-versions",
        json={"name": model_name, "stages": ["None"]},
    )
    latest_versions = response.json().get("model_versions", [])
    return latest_versions[-1]["version"]


def load_model():
    """Laad het meest recente model uit de MLFlow registry."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    latest_version = get_latest_version(MODEL_NAME)
    print(f"Model: {MODEL_NAME} versie {latest_version}")
    return mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{latest_version}")
