FROM python:3.11-slim

RUN pip install -U pip

WORKDIR /app

# Requirements apart kopiëren zodat Docker de laag kan cachen
COPY services/web-service/requirements.txt .
RUN pip install -r requirements.txt

# Projectcode kopiëren
COPY scripts/ ./scripts/

# PYTHONPATH instellen zodat 'from scripts.x import ...' werkt
ENV PYTHONPATH=/app

EXPOSE 9696

CMD ["python", "scripts/api/predict.py"]
