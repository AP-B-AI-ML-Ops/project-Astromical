FROM python:3.11-slim

RUN pip install -U pip setuptools wheel

RUN pip install prefect

RUN prefect config set PREFECT_API_URL="http://127.0.0.1:4200/api"

CMD ["prefect", "server", "start", "--host", "0.0.0.0", "--port", "4200"]
