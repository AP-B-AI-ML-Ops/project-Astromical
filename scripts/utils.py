"""Gedeelde hulpfuncties voor de training scripts."""

import mlflow
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error


def log_rmse_metrics(y_val, y_pred):
    """Bereken RMSE per doelvariabele en log naar de actieve MLFlow run."""
    rmse_solar = root_mean_squared_error(y_val[:, 0], y_pred[:, 0])
    rmse_wind = root_mean_squared_error(y_val[:, 1], y_pred[:, 1])
    rmse_mean = (rmse_solar + rmse_wind) / 2

    mlflow.log_metric("rmse_solar_mw", rmse_solar)
    mlflow.log_metric("rmse_wind_mw", rmse_wind)
    mlflow.log_metric("rmse", rmse_mean)

    return rmse_mean


def fit_and_log(params, x_train, y_train, x_val, y_val):
    """Train een RandomForest, log params en metrics, en retourneer model + RMSE."""
    mlflow.log_params(params)

    rf = RandomForestRegressor(**params)
    rf.fit(x_train, y_train)
    y_pred = rf.predict(x_val)

    rmse_mean = log_rmse_metrics(y_val, y_pred)

    return rf, rmse_mean


def haal_records_op(dataset_id, start, einde, extra_filter=""):
    """Haal alle records op voor een dataset"""
    url = f"https://opendata.elia.be/api/explore/v2.1/catalog/datasets/{dataset_id}/records"
    params = {
        "where": f'datetime >= "{start}" AND datetime <= "{einde}"{extra_filter}',
        "limit": 100,
        "order_by": "datetime ASC",
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("results", [])
