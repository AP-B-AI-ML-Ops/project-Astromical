"""Hyperparameter optimalisatie met Optuna en MLFlow tracking."""

from pathlib import Path

import mlflow
import optuna
from optuna.samplers import TPESampler

from scripts.train import load_splits
from scripts.utils import fit_and_log

MLFLOW_TRACKING_URI = "http://experiment-tracking:5000"
HPO_EXPERIMENT_NAME = "energy-forecast-hyperopt"

DATASETS_DIR = Path(__file__).parent.parent / "datasets" / "exports"


def run_optimization(data_path, num_trials=10):
    """Voer hyperparameter optimalisatie uit en log elke trial naar MLFlow."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(HPO_EXPERIMENT_NAME)
    # Dit zet autolog uit zodat ik zelf params en metrics log per trial
    mlflow.sklearn.autolog(disable=True)

    x_train, y_train, x_val, y_val, *_ = load_splits(data_path)

    def objective(trial):
        """Train model met gesuggereerde params en retourneer RMSE."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 20),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 4),
            "random_state": 42,
            "n_jobs": -1,
        }

        with mlflow.start_run():
            _, rmse_mean = fit_and_log(params, x_train, y_train, x_val, y_val)

        return rmse_mean

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=num_trials)

    print(f"Beste RMSE:   {study.best_value:.4f} MW")
    print(f"Beste params: {study.best_params}")


if __name__ == "__main__":
    print("...hyperparameters optimaliseren")
    run_optimization(DATASETS_DIR / "processed.parquet", num_trials=10)
