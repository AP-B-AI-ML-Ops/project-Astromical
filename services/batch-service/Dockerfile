FROM python:3.11-slim

RUN pip install -U pip

WORKDIR /app

# Requirements apart kopiëren zodat Docker de laag kan cachen
COPY services/batch-service/requirements.txt .
RUN pip install -r requirements.txt

# Projectcode en datasets kopiëren
COPY scripts/ ./scripts/
COPY datasets/ ./datasets/

# PYTHONPATH instellen zodat 'from scripts.xxx import ...' werkt
ENV PYTHONPATH=/app

CMD ["python", "scripts/batch.py"]
