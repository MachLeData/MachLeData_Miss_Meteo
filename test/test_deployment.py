import json
from pathlib import Path

import asciichartpy as acp
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import requests

DATASET = Path("data/prepared/val_historical.parquet")
MODEL_URL = "http://localhost:3000"
# MODEL_URL = "http://34.65.82.134:80"


def test_model_status():
    try:
        state = requests.post(MODEL_URL + "/status")
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


def predict(payload: dict) -> dict:
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


def preview(results):
    y_values = [
        item[0] if isinstance(item, (list, tuple)) else item for item in results
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


def plot_results(result_dict):
    timestamps = pd.to_datetime(result_dict["reference_timestamp"])
    temperatures = result_dict["air_temperature"]

    split_index = 24

    hist_dates = timestamps[:split_index]
    hist_temps = temperatures[:split_index]

    pred_dates = timestamps[split_index:]
    pred_temps = temperatures[split_index:]

    plt.figure(figsize=(12, 6))

    plt.plot(hist_dates, hist_temps, label="History", marker="o", color="blue")

    if len(hist_dates) > 0 and len(pred_dates) > 0:
        link_dates = [hist_dates[-1], pred_dates[0]]
        link_temps = [hist_temps[-1], pred_temps[0]]
        plt.plot(link_dates, link_temps, color="red", linestyle="--")

    plt.plot(
        pred_dates,
        pred_temps,
        label="Predictions",
        marker="x",
        color="red",
        linestyle="--",
    )

    # Formatage de l'axe X pour les dates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    plt.gca().xaxis.set_major_locator(
        mdates.HourLocator(interval=4)
    )  # Une étiquette toutes les 4h
    plt.gcf().autofmt_xdate()  # Rotation automatique des dates

    plt.title("Temperatures prediction (last 24h + next 24h)")
    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()

    plt.show()


def main():
    print("=" * 60)
    print("     TEST MODEL DEPLOYED")
    print("=" * 60 + "\n")

    # Test if model is deployed
    test_model_status()

    print("\n" + "=" * 60 + "\n")

    # Get validation data as test
    df = pd.read_parquet(DATASET, engine="pyarrow")

    # Prepare last 24h data for prediction
    df_last24h = df[df["historical"] == 1].tail(24)
    df_last24h.reset_index(drop=True, inplace=True)
    df_last24h["reference_timestamp"] = df_last24h["reference_timestamp"].astype(str)

    payload = {"data": df_last24h.to_dict(orient="list")}

    # Make prediction request
    result = predict(payload)

    print("\n" + "=" * 60 + "\n")

    # Display results
    preview(result["air_temperature"])
    plot_results(result)


if __name__ == "__main__":
    main()
