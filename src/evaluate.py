import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import pickle

from sklearn.metrics import root_mean_squared_error


def preview_prediction(date_values, target_values, predict_values):
    fig = plt.figure(figsize=(10, 6))
    plt.ylabel("Temperature [°C]")
    plt.xlabel("Time [m]")
    plt.plot(date_values, target_values, label="True values")
    plt.plot(date_values, predict_values, label="Predicted values")
    plt.title("Temperature prediction")
    plt.legend()

    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))

    return fig


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 evaluate.py <model-folder> <prepared-dataset-folder>\n")
        exit(1)

    model_file = Path(sys.argv[1])
    testset_file = Path(sys.argv[2])
    evaluation_folder = Path("evaluation")
    plots_folder = Path("plots")

    # Create folders
    (evaluation_folder / plots_folder).mkdir(parents=True, exist_ok=True)

    # Load the test dataset (path to the prepared test dataset)
    test_df = pd.read_parquet(testset_file, engine="pyarrow")
    test_df["reference_timestamp"] = pd.to_datetime(
        test_df["reference_timestamp"], dayfirst=True
    )

    test_features = test_df.drop(["reference_timestamp", "air_temperature"], axis=1)
    test_target = test_df["air_temperature"]

    # Load the model (path to the model)
    model = pickle.load(open(model_file, "rb"))
    test_predict = model.predict(test_features)

    # Preview prediction
    fig_preduction = preview_prediction(
        test_df["reference_timestamp"],
        test_target,
        test_predict,
    )
    fig_preduction.savefig(evaluation_folder / plots_folder / "prediction_preview.png")

    # Compute performances
    pred_mse = root_mean_squared_error(test_target, test_predict)
    print(f"RMSE: {pred_mse:.3f}%")

    with open(evaluation_folder / "metrics.json", "w") as f:
        json.dump({"\nRMSE": pred_mse}, f)

    print(
        f"\nEvaluation metrics and plot files saved at {evaluation_folder.absolute()}"
    )


if __name__ == "__main__":
    main()
