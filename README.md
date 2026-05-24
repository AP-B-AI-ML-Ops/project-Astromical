# Renewable Energy Forecasting

## Dataset(s)

The training data comes from the Data Engineering course and combines three sources:

| File | Sources | Content |
|---|---|---|
| `productie_combined.csv` | Energie Vlaanderen, Elia | Hourly solar and wind production (kWh, converted to MW) |
| `v_wind_alles_compleet.csv` | Open-Meteo ECMWF, Geo.be, Kaggle | Hourly wind speed (km/h) |
| `sun_combined.csv` | Open-Meteo ECMWF, Geo.be, Kaggle | Daily solar radiation (W/m2), expanded to hourly |

The three files are joined on the `tijd` column and saved as `datasets/exports/processed.parquet`.

The data is split chronologically: 70% training, 10% validation, 20% test. This ensures the model is always evaluated on future data.

For new predictions, the service fetches live ECMWF weather data from the Open-Meteo API. The batch service also compares predictions against real production data from the [Elia Open Data API](https://opendata.elia.be).

## Project Explanation

This project predicts **solar production (MW)** and **wind production (MW)** for the Antwerp region for the next 24 hours. Grid operators and energy traders need reliable short-term forecasts to keep the electricity grid balanced.

**Inputs** (per hour):

| Feature | Description |
|---|---|
| `wind_speed_kmh` | Wind speed at 10 m height (km/h) |
| `solar_radiation_wm2` | Solar radiation (W/m2) |
| `hour` | Hour of the day (0-23) |
| `month` | Month of the year (1-12) |
| `dayofweek` | Day of the week (0 = Monday, 6 = Sunday) |
| `is_weekend` | 1 if Saturday or Sunday, else 0 |

**Outputs** (per hour, for the next 24 hours):

| Feature | Description |
|---|---|
| `solar_mw` | Predicted solar production (MW) |
| `wind_mw` | Predicted wind production (MW) |

The system is available as two services:

**Web Service** - a REST API that accepts weather forecast data and returns predicted energy production. Useful for real-time dashboards or trading decisions.

**Batch Service** - a daily Prefect pipeline that runs at 06:00 UTC. It fetches live weather data, runs predictions, compares them against real Elia production data, and automatically triggers retraining if RMSE exceeds 50 MW.

## Flows & Actions

### Training Pipeline

Runs automatically on first startup and is re-triggered by the batch service when RMSE is too high.

1. Load CSVs, join on `tijd` column and save as parquet
2. Train a base StandardScaler + RandomForest pipeline and log results to MLFlow (validation and test RMSE)
3. Run Optuna HPO with 10 trials and log each trial to MLFlow
4. Retrain the top-5 models and register the best one in the MLFlow model registry

### Batch Pipeline

Runs daily at 06:00 UTC via Prefect cron schedule.

1. Fetch the next 24h weather forecast from Open-Meteo
2. Run model on forecast data and save predictions as parquet
3. Fetch yesterday's weather from Open-Meteo and run model on it
4. Fetch yesterday's real production from the Elia API
5. Compare predictions against actuals and compute RMSE (solar, wind, mean)
6. Write metrics to PostgreSQL and generate an Evidently report
7. Trigger the training pipeline if RMSE exceeds 50 MW

### Web Service

POST /predict
```
  body:
    {
      "forecasts": [{
        "wind_speed_kmh": 18.5,
        "solar_radiation_wm2": 320.0,
        "hour": 14,
        "month": 6,
        "dayofweek": 2,
        "is_weekend": 0}]
    }

  response:
  {
    "predictions": [
      {"hour": 14, "solar_mw": 412.3, "wind_mw": 187.6}
      ]}
```
GET /health
```
  response: {"status": "ok"}
```

## How to Run

**Requirements:**
- Docker

**Start:**

```bash
docker compose up -d --build
```

The training pipeline runs automatically on first startup. The batch service then starts on the cron schedule (06:00 UTC daily).

**Services:**

| Service | URL | Credentials | What to find |
|---|---|---|---|
| Web service | http://localhost:9696 | none | POST /predict, GET /health |
| MLFlow | http://localhost:5000 | none | experiments, runs, metrics, model registry |
| Prefect | http://localhost:4200 | none | flow runs, deployments, cron schedule |
| Grafana | http://localhost:3000 | admin / admin | RMSE over time, predicted vs actual solar and wind charts |
| Evidently | http://localhost:8000 | none | daily data quality and regression reports |

**Example API call:**

```bash
curl -X POST http://localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{"forecasts": [{"wind_speed_kmh": 18.5, "solar_radiation_wm2": 320.0, "hour": 14, "month": 6, "dayofweek": 2, "is_weekend": 0}]}'
```

**Stop:**

```bash
docker compose down
```

Including all stored data:

```bash
docker compose down -v
```
