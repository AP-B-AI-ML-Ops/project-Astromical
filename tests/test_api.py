"""Tests voor de predict API."""

# pylint: disable=redefined-outer-name
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

VALID_FORECAST = {
    "wind_speed_kmh": 18.5,
    "solar_radiation_wm2": 320.0,
    "hour": 14,
    "month": 6,
    "dayofweek": 2,
    "is_weekend": 0,
}


@pytest.fixture
def client():
    mock_model = MagicMock()
    mock_model.predict.side_effect = lambda x: np.array([[100.0, 50.0]] * len(x))

    with patch("scripts.models.load.load_model", return_value=mock_model):
        sys.modules.pop("scripts.api.predict", None)
        import scripts.api.predict as predict_module  # pylint: disable=import-outside-toplevel

        predict_module.app.config["TESTING"] = True
        with predict_module.app.test_client() as test_client:
            yield test_client

    sys.modules.pop("scripts.api.predict", None)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_valid(client):
    response = client.post("/predict", json={"forecasts": [VALID_FORECAST]})
    assert response.status_code == 200
    data = response.get_json()
    assert "predictions" in data
    assert len(data["predictions"]) == 1
    prediction = data["predictions"][0]
    assert "solar_mw" in prediction
    assert "wind_mw" in prediction
    assert prediction["solar_mw"] == pytest.approx(100.0)
    assert prediction["wind_mw"] == pytest.approx(50.0)


def test_predict_24_uur(client):
    response = client.post("/predict", json={"forecasts": [VALID_FORECAST] * 24})
    assert response.status_code == 200
    assert len(response.get_json()["predictions"]) == 24


def test_predict_ontbrekend_forecasts_veld(client):
    response = client.post("/predict", json={"data": []})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_predict_ontbrekende_feature(client):
    incomplete = {k: v for k, v in VALID_FORECAST.items() if k != "wind_speed_kmh"}
    response = client.post("/predict", json={"forecasts": [incomplete]})
    assert response.status_code == 400
    assert "error" in response.get_json()
