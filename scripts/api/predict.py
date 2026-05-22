"""REST API voor energieproductie-voorspellingen."""

import numpy as np
from flask import Flask, jsonify, request

from scripts.models.load import load_model

FEATURES = [
    "wind_speed_kmh",
    "solar_radiation_wm2",
    "hour",
    "month",
    "dayofweek",
    "is_weekend",
]

app = Flask("energy-forecast")
model = load_model()


@app.route("/health", methods=["GET"])
def health():
    """Controleert of de API actief en werkend is."""
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """Ontvang weersvoorspellingen en retourneer energieproductie in MW."""
    data = request.get_json()

    if not data or "forecasts" not in data:
        return jsonify({"error": "Verplicht veld 'forecasts' ontbreekt"}), 400

    forecasts = data["forecasts"]

    # Invoer omzetten naar numpy array in de juiste feature volgorde
    try:
        x = np.array([[f[feature] for feature in FEATURES] for f in forecasts])
    except KeyError as e:
        return jsonify({"error": f"Ontbrekend veld in forecast: {e}"}), 400

    # Model voorspellingen
    predictions = model.predict(x)

    result = [
        {
            "hour": forecasts[i].get("hour", i),
            "solar_mw": round(float(predictions[i][0]), 4),
            "wind_mw": round(float(predictions[i][1]), 4),
        }
        for i in range(len(forecasts))
    ]

    return jsonify({"predictions": result})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=9696)
