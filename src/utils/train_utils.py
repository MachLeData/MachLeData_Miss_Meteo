import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

def mark_model_as_safe():
    torch.serialization.add_safe_globals([
        NeuralNetModel,
        torch.nn.modules.linear.Linear,
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.dropout.Dropout
    ])

class NeuralNetModel(nn.Module):
    def __init__(
        self,
        input_shape: int,
        n_neurons1: int,
        n_neurons2: int,
        dropout: float,
        activation: str,
        kernel_initializer: str,
    ):
        super().__init__()
        
        # Define layers
        self.dense1 = nn.Linear(input_shape, n_neurons1)
        self.activation1 = self._get_activation(activation)
        self.dropout = nn.Dropout(dropout)
        
        self.dense2 = nn.Linear(n_neurons1, n_neurons2)
        self.activation2 = self._get_activation(activation)
        
        self.output = nn.Linear(n_neurons2, 1)
        
        # Initialize weights
        self._init_weights(kernel_initializer)
    
    def _get_activation(self, activation: str):
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'elu': nn.ELU(),
            'leaky_relu': nn.LeakyReLU(),
        }
        return activations.get(activation.lower(), nn.ReLU())
    
    def _init_weights(self, kernel_initializer: str):
        if kernel_initializer.lower() == 'glorot_uniform':
            for layer in [self.dense1, self.dense2, self.output]:
                nn.init.xavier_uniform_(layer.weight)
        elif kernel_initializer.lower() == 'he_normal':
            for layer in [self.dense1, self.dense2, self.output]:
                nn.init.kaiming_normal_(layer.weight)
    
    def forward(self, x):
        x = self.dense1(x)
        x = self.activation1(x)
        x = self.dropout(x)
        
        x = self.dense2(x)
        x = self.activation2(x)
        
        x = self.output(x)
        return x


def create_model(
    *,
    input_shape: int,
    n_neurons1: int,
    n_neurons2: int,
    dropout: float,
    activation: str,
    kernel_initializer: str,
    learning_rate: float,
    loss: str,
):
    # Create model
    model = NeuralNetModel(
        input_shape=input_shape,
        n_neurons1=n_neurons1,
        n_neurons2=n_neurons2,
        dropout=dropout,
        activation=activation,
        kernel_initializer=kernel_initializer,
    )
    
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_fn_map = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'binary_crossentropy': nn.BCEWithLogitsLoss(),
        'categorical_crossentropy': nn.CrossEntropyLoss(),
    }

    loss_fn = loss_fn_map.get(loss.lower(), nn.MSELoss())
    
    return model, optimizer, loss_fn

def plot_training_history(train_losses, val_losses=None, ax=None):
    plt.style.use("seaborn-v0_8-darkgrid")
    
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    
    epochs = range(1, len(train_losses) + 1)
    
    # Plot training loss
    ax.plot(
        epochs, train_losses, "o-",
        label="Training MSE", color="#1f77b4", linewidth=2, markersize=6
    )
    
    # Plot validation loss if provided
    if val_losses is not None and len(val_losses) > 0:
        ax.plot(
            epochs, val_losses, "s--",
            label="Validation MSE", color="#ff7f0e", linewidth=2, markersize=6
        )
    
    ax.set_title("Model Training History", fontsize=16, fontweight="bold", pad=15)
    ax.set_xlabel("Epoch", fontsize=13)
    ax.set_ylabel("Mean Squared Error (MSE)", fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.6)
    
    return ax