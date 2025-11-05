from sklearn.dummy import DummyRegressor
from sklearn.metrics import root_mean_squared_error
from pathlib import Path
from utils.seed import set_seed
import yaml
import pickle 
import sys
import pandas as pd
import tensorflow as tf
import bentoml
import matplotlib.pyplot as plt
from tensorflow.keras import Model, layers, activations, initializers, optimizers

def create_model(
    *,
    input_shape: tuple,
    n_neurons1: int,
    n_neurons2: int,
    dropout: int,
    activation: str,
    kernel_initializer: str,
    learning_rate: float,
    loss: str
):
    input_layer = layers.Input(input_shape)
    model_layer = layers.Dense(n_neurons1, activation=activation, kernel_initializer=kernel_initializer)(input_layer)
    model_layer = layers.Dropout(0.2)(model_layer)
    model_layer = layers.Dense(n_neurons2, activation=activation, kernel_initializer=kernel_initializer)(model_layer)
    output_layer = layers.Dense(1)(model_layer)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=optimizers.Adam(learning_rate),
        loss=loss
    )

    return model

def plot_training_history(history, ax=None):
    plt.style.use("seaborn-v0_8-darkgrid")

    train_loss = history.history["loss"]
    val_loss = history.history.get("val_loss")

    if not ax:
        _, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    epochs = range(1, len(train_loss) + 1)

    # Plot training and validation loss
    ax.plot(
        epochs, train_loss, "o-",
        label="Training MSE", color="#1f77b4", linewidth=2, markersize=6
    )
    if val_loss:
        ax.plot(
            epochs, val_loss, "s--",
            label="Validation MSE", color="#ff7f0e", linewidth=2, markersize=6
        )

    ax.set_title("Model Training History", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)

def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 train.py <train_dataframe_file> <val_dataframe_file>\n")
        exit(1)

    # Load parameters
    train_params = yaml.safe_load(open("params.yaml"))["train"]
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
    reload_previous_model = train_params["reload_previous_model"]

    evaluation_folder = Path("evaluation")
    plots_folder = Path("plots")

    # Create folders
    (evaluation_folder / plots_folder).mkdir(parents=True, exist_ok=True)
    model_folder = Path("model")
    model_folder.mkdir(parents=True, exist_ok=True)

    # Set seed for reproducibility
    set_seed(seed)

    # Load train and validation datasets
    train_df = pd.read_parquet(Path(sys.argv[1]))
    val_df = pd.read_parquet(Path(sys.argv[2]))

    # Shuffle train
    train_df = train_df.sample(frac=1)

    train_features = train_df.drop(["reference_timestamp", "air_temperature"], axis=1)
    train_target = train_df['air_temperature']

    val_features = val_df.drop(["reference_timestamp", "air_temperature"], axis=1)
    val_target = val_df['air_temperature']

    # Train model
    if reload_previous_model:
        try:
            model = bentoml.keras.load_model("air_temperature_regressor")
        except:
            raise Exception(f"Load previous model (air_temperature_regressor) failed!")
    else:
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

    print(model.summary())

    history = model.fit(
        x=train_features,
        y=train_target,
        validation_data=(val_features, val_target),
        epochs=n_epochs,
        verbose=2,
        batch_size=batch_size
    )

    plot_training_history(history)
    plt.savefig(evaluation_folder / plots_folder / 'train_history.png')

    # Save the model using BentoML
    # Export the model from the model store to the local model folder
    model_path = f"{model_folder.absolute()}/air_temperature_regressor.bentomodel"
    bentoml.keras.save_model("air_temperature_regressor", model)
    bentoml.models.export_model(
        "air_temperature_regressor:latest",
        model_path,
    )

    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()