import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import bentoml
from torch.utils.data import TensorDataset, DataLoader
from utils.train_utils import create_model 
from utils.train_utils import plot_training_history


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def train_epoch(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x_batch, y_batch in train_loader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        output = model(x_batch)
        loss = loss_fn(output, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def validate(model, val_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            output = model(x_batch)
            loss = loss_fn(output, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


def main() -> None:
    if len(sys.argv) != 3:
        print("Arguments error. Usage:\n")
        print("\tpython3 train.py <train_historical_dataframe_file> <val_historical_dataframe_file>\n")
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
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load train and validation datasets
    train_historical_df = pd.read_parquet(Path(sys.argv[1]))
    val_historical_df = pd.read_parquet(Path(sys.argv[2]))
    
    # Shuffle train data
    train_historical_df = train_historical_df.sample(frac=1, random_state=seed)
    train_features = train_historical_df.drop(["reference_timestamp", "air_temperature", "historical"], axis=1)
    train_target = train_historical_df['air_temperature']
    
    val_features = val_historical_df.drop(["reference_timestamp", "air_temperature", "historical"], axis=1)
    val_target = val_historical_df['air_temperature']
    
    # Convert to PyTorch tensors
    train_features_tensor = torch.FloatTensor(train_features.values)
    train_target_tensor = torch.FloatTensor(train_target.values.reshape(-1, 1))
    val_features_tensor = torch.FloatTensor(val_features.values)
    val_target_tensor = torch.FloatTensor(val_target.values.reshape(-1, 1))
    
    # Create datasets and dataloaders
    train_dataset = TensorDataset(train_features_tensor, train_target_tensor)
    val_dataset = TensorDataset(val_features_tensor, val_target_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create baseline model
    model, optimizer, loss_fn = create_model(
        input_shape=train_features.shape[1],
        n_neurons1=n_neurons1,
        n_neurons2=n_neurons2,
        dropout=dropout,
        activation=activation,
        kernel_initializer=kernel_initializer,
        learning_rate=learning_rate,
        loss=loss,
    )
    model = model.to(device)
    
    # Print model summary
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Training loop
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = validate(model, val_loader, loss_fn, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            print(f"Epoch {epoch + 1}/{n_epochs} - "
                  f"Training Loss: {train_loss:.6f}, "
                  f"Validation Loss: {val_loss:.6f}")
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    plt.savefig(evaluation_folder / plots_folder / 'baseline_train_history.png')
    print(f"Plot saved to {evaluation_folder / plots_folder / 'baseline_train_history.png'}")
    
    # Save the model using BentoML
    bentoml.pytorch.save_model(
        "baseline",
        model,
        signatures={"__call__": {"batchable": True}},
    )
    print("Model saved to BentoML model store as 'baseline'")
    
    # Export the model from the model store to the local model folder
    model_path = f"{model_folder.absolute()}/baseline.bentomodel"
    bentoml.models.export_model("baseline:latest", model_path)
    print(f"Model exported to {model_path}")


if __name__ == "__main__":
    main()