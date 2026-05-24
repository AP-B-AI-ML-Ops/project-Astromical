FROM python:3.11-slim

RUN pip install -U pip && pip install psycopg2-binary mlflow uvicorn

# --host 0.0.0.0 zodat MLFlow bereikbaar is vanuit andere containers
# --allowed-hosts "*" om de nieuwe security middleware te omzeilen
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "postgresql+psycopg2://postgres:postgres@database/postgres", \
     "--serve-artifacts", \
     "--artifacts-destination", "/mlruns", \
     "--allowed-hosts", "*"]
