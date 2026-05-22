"""Voert de volledige training pipeline uit dus preprocess, train, hpo en registreeren."""

from pathlib import Path

from scripts.data.preprocess import preprocess
from scripts.hpo import run_optimization
from scripts.register import run_register_model
from scripts.train import run_train

PARQUET_PATH = Path("datasets/exports/processed.parquet")


print("Data preprocessen...")
preprocess(PARQUET_PATH)

print("Basistraining...")
run_train(PARQUET_PATH)

print("Hyperparameter optimalisatie...")
run_optimization(PARQUET_PATH, num_trials=10)

print("Beste model registreren...")
run_register_model(PARQUET_PATH, top_n=5)

print("Training pipeline voltooid.")
