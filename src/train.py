import yaml
import sys
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import bentoml
import matplotlib.pyplot as plt
import numpy as np
from utils.train_utils import plot_training_history, mark_model_as_safe
from utils.seed import set_seed
from pathlib import Path

mark_model_as_safe()

def train_epoch(model, train_loader, optimizer, loss_fn, device):
    """Train for one epoch."""
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
    """Validate the model."""
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
    if len(sys.argv) != 5:
        print("Arguments error. Usage:\n")
        print("\tpython3 train.py <train_historical_dataframe_file> <val_historical_dataframe_file> <train_finetuning_dataframe_file> <val_finetuning_dataframe_file>\n")
        exit(1)
    
    # Load parameters
    train_params = yaml.safe_load(open("params.yaml"))["train"]
    seed = train_params["seed"]
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
    train_finetuning_df = pd.read_parquet(Path(sys.argv[3]))
    val_finetuning_df = pd.read_parquet(Path(sys.argv[4]))
    
    # Shuffle train
    train_df = train_finetuning_df.sample(frac=1)
    train_features = train_df.drop(["reference_timestamp", "air_temperature", "historical"], axis=1)
    train_target = train_df['air_temperature']
    
    val_features = val_finetuning_df.drop(["reference_timestamp", "air_temperature", "historical"], axis=1)
    val_target = val_finetuning_df['air_temperature']
    
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
    
    # Load baseline model from BentoML
    model = bentoml.pytorch.load_model("baseline")
    model = model.to(device)
    
    # FIXME: should we freeze some weights?
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    loss_fn_map = {
        'mse': nn.MSELoss(),
        'mae': nn.L1Loss(),
        'binary_crossentropy': nn.BCEWithLogitsLoss(),
        'categorical_crossentropy': nn.CrossEntropyLoss(),
    }
    loss_fn = loss_fn_map.get(loss.lower(), nn.MSELoss())
    
    # Fine-tuning training loop
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
    plt.savefig(evaluation_folder / plots_folder / 'train_history.png')
    print(f"Plot saved to {evaluation_folder / plots_folder / 'train_history.png'}")
    
    # Save the fine-tuned model using BentoML
    bentoml.pytorch.save_model(
        "model",
        model,
        signatures={"__call__": {"batchable": True}},
    )
    print("Fine-tuned model saved to BentoML model store as 'model'")
    
    # Export the model from the model store to the local model folder
    model_path = f"{model_folder.absolute()}/model.bentomodel"
    bentoml.models.export_model("model:latest", model_path)
    print(f"Model exported to {model_path}")


if __name__ == "__main__":
    main()