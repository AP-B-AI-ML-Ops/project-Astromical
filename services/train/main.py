"""Voert de volledige training pipeline uit met Prefect orchestratie."""

from pathlib import Path

from prefect import flow, task

from scripts.data.preprocess import preprocess
from scripts.hpo import run_optimization
from scripts.register import run_register_model
from scripts.train import run_train

PARQUET_PATH = Path("datasets/exports/processed.parquet")


@task(name="preprocess", log_prints=True)
def stap_preprocess(parquet_path):
    """Laad CSV-bestanden en exporteer de verwerkte parquet."""
    preprocess(parquet_path)


@task(name="basistraining", log_prints=True)
def stap_train(parquet_path):
    """Traint een basismodel met vaste parameters en log ze naar MLFlow."""
    run_train(parquet_path)


@task(name="hyperparameter-optimalisatie", log_prints=True)
def stap_hpo(parquet_path, num_trials=10):
    """Voert Optuna HPO uit en logt elke trial naar MLFlow."""
    run_optimization(parquet_path, num_trials=num_trials)


@task(name="model-registreren", log_prints=True)
def stap_registreer(parquet_path, top_n=5):
    """Selecteerd het beste model uit HPO en registreert het in de MLFlow registry."""
    run_register_model(parquet_path, top_n=top_n)


@flow(name="training-pipeline", log_prints=True)
def training_pipeline(parquet_path=PARQUET_PATH, num_trials=10, top_n=5):
    """Volledige training pipeline"""
    stap_preprocess(parquet_path)
    stap_train(parquet_path)
    stap_hpo(parquet_path, num_trials=num_trials)
    stap_registreer(parquet_path, top_n=top_n)


if __name__ == "__main__":
    training_pipeline()
