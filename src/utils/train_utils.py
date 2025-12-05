from tensorflow.keras import Model, layers, activations, initializers, optimizers
import matplotlib.pyplot as plt

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