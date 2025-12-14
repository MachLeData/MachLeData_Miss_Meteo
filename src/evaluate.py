import sys
import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import torch
import bentoml
from sklearn.metrics import root_mean_squared_error
from utils.train_utils import mark_model_as_safe

mark_model_as_safe()

def preview_prediction(date_values, target_values, predict_values, title):
    """Generate prediction visualization plot."""
    fig = plt.figure(figsize=(10, 6))
    plt.ylabel("Temperature [°C]")
    plt.xlabel("Time [m]")
    plt.plot(date_values, target_values, label="True values")
    plt.plot(date_values, predict_values, label="Predicted values")
    plt.title(title)
    plt.legend()
    plt.gcf().autofmt_xdate()
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M"))
    return fig


def load_dataset(dataset_file):
    """Load and prepare dataset from parquet file."""
    df = pd.read_parquet(dataset_file, engine="pyarrow")
    df["reference_timestamp"] = pd.to_datetime(
        df["reference_timestamp"], dayfirst=True
    )
    features = df.drop(["reference_timestamp", "air_temperature", "historical"], axis=1)
    target = df["air_temperature"]
    timestamps = df["reference_timestamp"]
    return features, target, timestamps


def make_predictions(model, features, device):
    """Generate predictions from PyTorch model."""
    model.eval()
    features_tensor = torch.FloatTensor(features.values)
    
    with torch.no_grad():
        predictions_tensor = model(features_tensor.to(device))
    
    return predictions_tensor.cpu().numpy().flatten()


def evaluate_model(model, features, target, timestamps, model_name, dataset_name, plots_folder, device):
    """Evaluate a model on a dataset and return metrics and plot."""
    predictions = make_predictions(model, features, device)
    rmse = root_mean_squared_error(target, predictions)
    
    # Generate plot
    fig = preview_prediction(
        timestamps,
        target,
        predictions,
        f"{model_name} - {dataset_name}"
    )
    plot_filename = f"{model_name.lower().replace(' ', '_')}_{dataset_name.lower().replace(' ', '_')}.png"
    fig.savefig(plots_folder / plot_filename)
    plt.close(fig)
    
    return rmse, plot_filename


def load_model_from_store(model_name):
    """Load a model from BentoML model store."""
    try:
        model = bentoml.pytorch.load_model(model_name)
        print(f"{model_name} loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading {model_name}: {e}")
        return None


def import_model_to_store(model_file):
    """Import a model to BentoML model store."""
    try:
        bentoml.models.import_model(f"{model_file.absolute()}")
    except bentoml.exceptions.BentoMLException:
        print("Model already exists in the model store - skipping import.")


def evaluate_model_on_datasets(model, model_name, historical_data, finetuning_data, plots_folder, device):
    """Evaluate a model on both validation datasets."""
    hist_features, hist_target, hist_timestamps = historical_data
    ft_features, ft_target, ft_timestamps = finetuning_data
    
    print(f"\nEvaluating {model_name} on historical validation set...")
    hist_rmse, hist_plot = evaluate_model(
        model,
        hist_features,
        hist_target,
        hist_timestamps,
        model_name,
        "Historical Validation",
        plots_folder,
        device
    )
    print(f"RMSE: {hist_rmse:.3f}")
    
    print(f"\nEvaluating {model_name} on finetuning validation set...")
    ft_rmse, ft_plot = evaluate_model(
        model,
        ft_features,
        ft_target,
        ft_timestamps,
        model_name,
        "Finetuning Validation",
        plots_folder,
        device
    )
    print(f"RMSE: {ft_rmse:.3f}")
    
    return hist_rmse, ft_rmse


def calculate_improvement_metrics(baseline_metrics, finetuned_metrics):
    """Calculate improvement metrics between baseline and finetuned models."""
    bl_hist, bl_ft = baseline_metrics
    ft_hist, ft_ft = finetuned_metrics
    
    return {
        "improvement_historical_validation": bl_hist - ft_hist,
        "improvement_finetuning_validation": bl_ft - ft_ft,
        "improvement_historical_validation_pct": (bl_hist - ft_hist) / bl_hist * 100,
        "improvement_finetuning_validation_pct": (bl_ft - ft_ft) / bl_ft * 100,
    }


def should_update_baseline(baseline_metrics, finetuned_metrics, baseline_exists):
    """Determine if baseline should be updated based on performance."""
    if not baseline_exists:
        print("\n" + "="*60)
        print("No baseline model found. Setting current model as baseline.")
        print("="*60)
        return True
    
    bl_hist, bl_ft = baseline_metrics
    ft_hist, ft_ft = finetuned_metrics
    
    ft_better_historical = ft_hist < bl_hist
    ft_better_finetuning = ft_ft < bl_ft
    
    if ft_better_historical and ft_better_finetuning:
        print("\n" + "="*60)
        print("Finetuned model outperforms baseline on BOTH validation sets!")
        print("Updating baseline model...")
        print("="*60)
        return True
    else:
        print("\n" + "="*60)
        print("Finetuned model does NOT outperform baseline on all validation sets.")
        print("Keeping existing baseline model.")
        print("="*60)
        return False


def save_baseline_model(model):
    """Save model as the new baseline in BentoML store and export to folder."""
    try:
        bentoml.pytorch.save_model(
            "baseline",
            model,
            signatures={"__call__": {"batchable": True}},
        )
        
        baseline_model_folder = Path("model/")
        baseline_model_folder.mkdir(parents=True, exist_ok=True)
        baseline_model_path = baseline_model_folder / "baseline.bentomodel"
        
        bentoml.models.export_model(
            "baseline:latest",
            str(baseline_model_path.absolute()),
        )
        
        print(f"New baseline model saved to {baseline_model_path.absolute()}")
        return True, baseline_model_path
        
    except Exception as e:
        print(f"Error saving baseline model: {e}")
        return False, None


def save_metrics(metrics, metrics_file):
    """Save metrics dictionary to JSON file."""
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)


def print_summary(all_metrics, metrics_file, plots_folder):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    for key, value in all_metrics.items():
        if key in ["baseline_updated", "baseline_update_timestamp"]:
            print(f"{key}: {value}")
        elif "pct" in key:
            print(f"{key}: {value:.2f}%")
        elif isinstance(value, (int, float)):
            print(f"{key}: {value:.3f}")
    print("="*60)
    print(f"\nEvaluation metrics saved to: {metrics_file.absolute()}")
    print(f"Plots saved to: {plots_folder.absolute()}")


def main() -> None:
    if len(sys.argv) != 4:
        print("Arguments error. Usage:\n")
        print("\tpython3 evaluate.py <model-folder> <dataset-historical-validation> <dataset-finetuning-validation>\n")
        exit(1)
    
    model_file = Path(sys.argv[1])
    val_historical_df_file = Path(sys.argv[2])
    val_finetuning_df_file = Path(sys.argv[3])
    
    evaluation_folder = Path("evaluation")
    plots_folder = evaluation_folder / "plots"
    metrics_file = evaluation_folder / "metrics.json"
    
    # Create folders
    plots_folder.mkdir(parents=True, exist_ok=True)
    
    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Import the finetuned model to the model store
    import_model_to_store(model_file)
    
    # Load both datasets
    print("Loading datasets...")
    historical_data = load_dataset(val_historical_df_file)
    finetuning_data = load_dataset(val_finetuning_df_file)
    
    # Initialize metrics dictionary
    all_metrics = {}
    
    # Load and evaluate finetuned model
    print("\nLoading finetuned model...")
    finetuned_model = load_model_from_store("model")
    if finetuned_model is None:
        print("Failed to load finetuned model. Exiting.")
        exit(1)
    
    finetuned_model = finetuned_model.to(device)
    
    ft_hist_rmse, ft_ft_rmse = evaluate_model_on_datasets(
        finetuned_model,
        "Finetuned Model",
        historical_data,
        finetuning_data,
        plots_folder,
        device
    )
    
    all_metrics["finetuned_model_historical_validation_rmse"] = ft_hist_rmse
    all_metrics["finetuned_model_finetuning_validation_rmse"] = ft_ft_rmse
    
    # Load and evaluate baseline model if it exists
    print("\nLoading baseline model...")
    baseline_model = load_model_from_store("baseline")
    baseline_exists = baseline_model is not None
    
    if baseline_exists:
        baseline_model = baseline_model.to(device)
        
        bl_hist_rmse, bl_ft_rmse = evaluate_model_on_datasets(
            baseline_model,
            "Baseline Model",
            historical_data,
            finetuning_data,
            plots_folder,
            device
        )
        
        all_metrics["baseline_model_historical_validation_rmse"] = bl_hist_rmse
        all_metrics["baseline_model_finetuning_validation_rmse"] = bl_ft_rmse
        
        # Calculate improvements
        improvement_metrics = calculate_improvement_metrics(
            (bl_hist_rmse, bl_ft_rmse),
            (ft_hist_rmse, ft_ft_rmse)
        )
        all_metrics.update(improvement_metrics)
        
        baseline_metrics = (bl_hist_rmse, bl_ft_rmse)
    else:
        print("Warning: No baseline model found.")
        baseline_metrics = None
    
    # Determine if we should update the baseline
    update_baseline = should_update_baseline(
        baseline_metrics,
        (ft_hist_rmse, ft_ft_rmse),
        baseline_exists
    )
    
    if update_baseline:
        success, baseline_path = save_baseline_model(finetuned_model)
        all_metrics["baseline_updated"] = success
        if success:
            all_metrics["baseline_update_timestamp"] = pd.Timestamp.now().isoformat()
    else:
        all_metrics["baseline_updated"] = False
    
    # Save metrics and print summary
    save_metrics(all_metrics, metrics_file)
    print_summary(all_metrics, metrics_file, plots_folder)


if __name__ == "__main__":
    main()