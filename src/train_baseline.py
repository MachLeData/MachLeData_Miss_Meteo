from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error
from pathlib import Path
from utils.seed import set_seed
from utils.train_utils import create_model, plot_training_history
import yaml
import pickle 
import sys
import pandas as pd
import tensorflow as tf
import bentoml
import matplotlib.pyplot as plt

def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 train.py <val_historical_dataframe_file> <val_finetuning_dataframe_file>\n")
        exit(1)

    # Load parameters
    train_params = yaml.safe_load(open("params.yaml"))["train_baseline"]
    seed = train_params["seed"]
    n_neurons1 = train_params["n_neurons1"]
    n_neurons2 = train_params["n_neurons2"]
    dropout = train_params["dropout"]
    activation = train_params["activation"]
    kernel_initializer = train_params["kernel_initializer"]
    learning_rate = train_params["learning_rate"]
    loss = train_params["loss"]
    n_epochs = train_params["n_epochs"]
    batch_size = train_params["batch_size"]

    evaluation_folder = Path("evaluation")
    plots_folder = Path("plots")

    # Create folders
    (evaluation_folder / plots_folder).mkdir(parents=True, exist_ok=True)
    model_folder = Path("model")
    model_folder.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    set_seed(seed)

    # Load train and validation datasets
    train_historical_df = pd.read_parquet(Path(sys.argv[1]))
    val_historical_df = pd.read_parquet(Path(sys.argv[2]))
 
     # Shuffle train
    train_df = train_historical_df.sample(frac=1)

    train_features = train_df.drop(["reference_timestamp", "air_temperature", "historical"], axis=1)
    train_target = train_df['air_temperature']

    val_features = val_historical_df.drop(["reference_timestamp", "air_temperature", "historical"], axis=1)
    val_target = val_historical_df['air_temperature']

    # historical dataframe
    train_historical_df = train_historical_df.sample(frac=1)
    
    train_features = train_historical_df.drop(["reference_timestamp", "air_temperature", "historical"], axis=1)
    train_target = train_historical_df['air_temperature']

    val_features = val_historical_df.drop(["reference_timestamp", "air_temperature", "historical"], axis=1)
    val_target = val_historical_df['air_temperature']    

    # Create baseline model
    model = create_model(
        input_shape=train_features.shape[1:],
        n_neurons1=n_neurons1,
        n_neurons2=n_neurons2,
        dropout=dropout,
        activation=activation,
        kernel_initializer=kernel_initializer,
        learning_rate=learning_rate,
        loss=loss,
    )
    
    # FIXME: should we freeze some weights?
    print(model.summary())

    # Fine-tuning model
    history = model.fit(
        x=train_features,
        y=train_target,
        validation_data=(val_features, val_target),
        epochs=n_epochs,
        verbose=2,
        batch_size=batch_size
    )

    plot_training_history(history)
    plt.savefig(evaluation_folder / plots_folder / 'baseline_train_history.png')

    # Save the fine tuned model using BentoML
    # Export the model from the model store to the local model folder
    model_path = f"{model_folder.absolute()}/baseline.bentomodel"
    bentoml.keras.save_model("baseline", model)
    bentoml.models.export_model(
        "baseline:latest",
        model_path,
    )

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()