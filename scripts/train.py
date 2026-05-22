"""Basis training module met MLFlow experiment tracking."""

from pathlib import Path

import mlflow
import pandas as pd

from scripts.utils import fit_and_log

MLFLOW_TRACKING_URI = "http://experiment-tracking:5000"
EXPERIMENT_NAME = "energy-forecast-train"

DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "exports"

# Inputfeatures en doelvariabelen
FEATURES = [
    "wind_speed_kmh",
    "solar_radiation_wm2",
    "hour",
    "month",
    "dayofweek",
    "is_weekend",
]
TARGETS = ["solar_mw", "wind_mw"]

# 80% van de data voor training, 20% voor validatie
TRAIN_RATIO = 0.8


def make_splits(df):
    """Pure functie: splits een DataFrame in train- en validatie-arrays."""
    # Splitpunt op basis van de werkelijke datumrange geen hardgecodeerde datums
    split_idx = int(len(df) * TRAIN_RATIO)
    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx:]

    print(
        f"Train: {len(df_train)} rijen  ({df_train.index.min()} → {df_train.index.max()})"
    )
    print(
        f"Validatie: {len(df_val)} rijen  ({df_val.index.min()} → {df_val.index.max()})"
    )

    x_train = df_train[FEATURES].values
    y_train = df_train[TARGETS].values
    x_val = df_val[FEATURES].values
    y_val = df_val[TARGETS].values

    return x_train, y_train, x_val, y_val


def load_splits(path):
    """Laad de preprocessed parquet en geef de gesplitste arrays terug."""
    df = pd.read_parquet(path)
    return make_splits(df)


def run_train(data_path):
    """Train een RandomForestRegressor en log alles via MLFlow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    x_train, y_train, x_val, y_val = load_splits(data_path)

    params = {"max_depth": 10, "n_estimators": 100, "random_state": 42, "n_jobs": -1}

    with mlflow.start_run():
        rf, rmse_mean = fit_and_log(params, x_train, y_train, x_val, y_val)
        # Model opslaan als artifact in MLFlow
        mlflow.sklearn.log_model(rf, artifact_path="model")
        print(f"RMSE gemiddeld: {rmse_mean:.4f} MW")


if __name__ == "__main__":
    print("...model trainen")
    run_train(DATASETS_DIR / "processed.parquet")
