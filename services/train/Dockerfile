FROM python:3.11-slim

RUN pip install -U pip

WORKDIR /app

# Requirements apart kopiëren zodat Docker de laag kan cachen
COPY services/train/requirements.txt .
RUN pip install -r requirements.txt

# Projectcode en datasets kopiëren
COPY scripts/ ./scripts/
COPY datasets/ ./datasets/
COPY services/train/main.py .

# PYTHONPATH instellen zodat 'from scripts.x import ...' werkt
ENV PYTHONPATH=/app

CMD ["python", "main.py"]
