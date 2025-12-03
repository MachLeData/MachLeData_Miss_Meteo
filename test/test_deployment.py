import io
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import asciichartpy as acp
import pandas as pd
import requests

DATASET = Path("data/prepared/val_historical.parquet")
MODEL_URL = "http://localhost:3000"


def test_model_status():
    try:
        state = requests.post("http://localhost:3000" + "/status")
        if state.status_code == 200:
            status = state.json()
            print("✅ Model service is running:")
            print(json.dumps(status, indent=4))
        else:
            print("❌ Model service is not running. Exiting.")
            print("Status code: " + str(state.status_code))
            exit(1)

    except NameError:
        print("❌ Model service is not running. Exiting.")
        print("Error: " + str(NameError))
        exit(1)


def predict_payload(payload: dict) -> dict:
    try:
        response = requests.post(
            MODEL_URL + "/predict",
            json=payload,
            headers={"Content-Type": "application/json"},
        )
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction request successful.")
            return result
        else:
            print("❌ Prediction request failed.")
            print("Status code: " + str(response.status_code))
            print("Response: " + response.text)
            exit(1)
    except Exception as e:
        print("❌ Prediction request failed.")
        print("Error: " + str(e))
        exit(1)


def main():
    print("=" * 60)
    print("     TEST MODEL DEPLOYED")
    print("=" * 60 + "\n")

    # Test if model is deployed
    test_model_status()

    # Get validation data as test
    df = pd.read_parquet(DATASET, engine="pyarrow")
    df["reference_timestamp"] = pd.to_datetime(df["reference_timestamp"], dayfirst=True)
    features = df.drop(["reference_timestamp", "air_temperature", "historical"], axis=1)

    payload = {"input_data": features.to_dict(orient="records")}

    # Make prediction request
    result = predict_payload(payload)

    # Display preview of results
    y = result["res"]
    nb_value = 150

    y_values = [
        item[0] if isinstance(item, (list, tuple)) else item for item in y[:nb_value]
    ]

    print("\n📈 Preview:\n")
    config = {
        "height": 30,
        "colors": [
            acp.green,  # Couleur de la ligne
        ],
    }
    plot = acp.plot(y_values, config)
    print(plot)


if __name__ == "__main__":
    main()
