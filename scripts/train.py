"""Basis training module met MLFlow experiment tracking."""

from pathlib import Path

import mlflow
import pandas as pd

from scripts.utils import fit_and_log, log_rmse_metrics

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

# 70% training, 10% validatie, 20% test
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1


def make_splits(df):
    """Splits een DataFrame in train-, validatie- en testarrays."""
    n = len(df)
    train_end = int(n * TRAIN_RATIO)
    val_end = train_end + round(n * VAL_RATIO)

    df_train = df.iloc[:train_end]
    df_val = df.iloc[train_end:val_end]
    df_test = df.iloc[val_end:]

    print(
        f"Train: {len(df_train)} rijen ({df_train.index.min()} -> {df_train.index.max()})"
    )
    print(
        f"Validatie: {len(df_val)} rijen ({df_val.index.min()} -> {df_val.index.max()})"
    )
    print(
        f"Test: {len(df_test)} rijen ({df_test.index.min()} -> {df_test.index.max()})"
    )

    x_train = df_train[FEATURES].values
    y_train = df_train[TARGETS].values
    x_val = df_val[FEATURES].values
    y_val = df_val[TARGETS].values
    x_test = df_test[FEATURES].values
    y_test = df_test[TARGETS].values

    return x_train, y_train, x_val, y_val, x_test, y_test


def load_splits(path):
    """Laad de preprocessed parquet en geef de gesplitste arrays terug."""
    df = pd.read_parquet(path)
    return make_splits(df)


def run_train(data_path):
    """Traint een RandomForestRegressor pipeline en log alles via MLFlow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    x_train, y_train, x_val, y_val, x_test, y_test = load_splits(data_path)

    params = {"max_depth": 10, "n_estimators": 100, "random_state": 42, "n_jobs": -1}

    with mlflow.start_run():
        pipeline, rmse_val = fit_and_log(params, x_train, y_train, x_val, y_val)
        y_test_pred = pipeline.predict(x_test)
        log_rmse_metrics(y_test, y_test_pred, prefix="test_")
        mlflow.sklearn.log_model(pipeline, artifact_path="model")
        print(f"RMSE validatie: {rmse_val:.4f} MW")


if __name__ == "__main__":
    print("...model trainen")
    run_train(DATASETS_DIR / "processed.parquet")
