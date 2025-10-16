import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from data.dataloader import DataLoader
from data.preprocessing import DataPreprocessor, TrainTestSplitter
from config.config import CONFIG, DataConfig, WandBConfig
import wandb
import joblib
import pickle
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    """Load and preprocess data."""
    data_loader = DataLoader(CONFIG.data)
    raw_data = data_loader.load_data()
    
    preprocessor = DataPreprocessor(DataConfig())
    preprocessed_data = preprocessor.preprocess(raw_data)
    
    # Split data
    splitter = TrainTestSplitter()
    X_train, X_test, y_train, y_test = splitter.split_data(
        preprocessed_data, 'credit_risk', test_size=0.2, random_state=42, stratify=True
    )
    
    return X_train, X_test, y_train, y_test

def load_model_from_wandb(model_name: str, version: str):
    """Load model from W&B artifact without logging."""
    # Use the correct project and artifact names
    artifact_name = f"jojs-it-universitetet-i-k-benhavn/QuantitativeRiskManagement/{model_name}:{version}"
    
    print(f"Attempting to load artifact: {artifact_name}")
    
    # Initialize W&B with the correct project name
    wandb.init(project="QuantitativeRiskManagement", name=f"load_{model_name}_model")
    
    try:
        # Load the model artifact
        artifact = wandb.use_artifact(artifact_name, type="model")
        if artifact is None:
            raise ValueError(f"Artifact {artifact_name} not found or returned None")
        
        artifact_dir = artifact.download()
        
        # List all files in the artifact directory for debugging
        files_in_artifact = os.listdir(artifact_dir)
        print(f"Files in artifact: {files_in_artifact}")
        
        # Based on your artifact info: xgb_model.pkl pattern
        possible_files = [
            f"{model_name.split('_')[0]}_model.pkl",  # xgb_model.pkl, logistic_model.pkl
            f"{model_name}.pkl",
            "model.pkl"
        ]
        
        model_path = None
        for file_name in possible_files:
            potential_path = os.path.join(artifact_dir, file_name)
            if os.path.exists(potential_path):
                model_path = potential_path
                print(f"Found model file: {file_name}")
                break
        
        # Fallback: find any .pkl file
        if model_path is None:
            for file_name in files_in_artifact:
                if file_name.endswith('.pkl'):
                    model_path = os.path.join(artifact_dir, file_name)
                    print(f"Found model file (fallback): {file_name}")
                    break
        
        if model_path is None:
            raise FileNotFoundError(f"No .pkl file found. Available files: {files_in_artifact}")
        
        # Load with pickle
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        print(f"Successfully loaded model from {model_path}")
        
    except Exception as e:
        print(f"Error loading artifact {artifact_name}: {str(e)}")
        raise
    finally:
        wandb.finish()
    
    return model

def evaluate_model(model, X_test, y_test, model_name: str):
    """Evaluate model and display results."""
    print(f"\n{'='*50}")
    print(f"EVALUATION - {model_name.upper()}")
    print(f"{'='*50}")
    
    # Get predictions
    y_pred = model.predict(X_test)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def main():
    """Main evaluation script."""
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_data()
    
    # Use the correct artifact names
    models_to_evaluate = [
        {"name": "logistic_regression_best_model", "version": "v2"},
        {"name": "xgboost_best_model", "version": "v0"}
    ]
    
    for model_config in models_to_evaluate:
        model_name = model_config["name"]
        version = model_config["version"]
        
        try:
            print(f"\nLoading {model_name} model (version {version})...")
            model = load_model_from_wandb(model_name, version)
            evaluate_model(model, X_test, y_test, model_name)
            
        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")

if __name__ == "__main__":
    main()