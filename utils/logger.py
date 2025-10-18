import wandb
import pickle
import joblib
import os
from typing import Dict, Any, Optional, Union
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from config.config import CONFIG
from utils.visualization import ModelVisualizer

class WandBLogger:
    """Simple W&B logger for model training and evaluation."""
    
    def __init__(self, 
                 model_type: str = "default",
                 model: Optional[BaseEstimator] = None,
                 config: Optional[Dict[str, Any]] = None, 
                 model_name: Optional[str] = None):
        """Initialize W&B logger with model-specific configuration.
        
        Args:
            model_type: Type of model ("logistic_regression", "xgboost", or "default")
            model: Optional model instance for auto-detection
            config: Optional config overrides
            model_name: Optional custom model name
        """
        # Auto-detect model type if model is provided but model_type is default
        if model is not None and model_type == "default":
            model_type = self._detect_model_type(model)
        
        self.model_type = model_type
        self.visualizer = ModelVisualizer()
        
        # Get appropriate W&B config based on model type
        wandb_config = self._get_wandb_config(model_type)
        
        # Override with custom config if provided
        if config:
            for key, value in config.items():
                setattr(wandb_config, key, value)
        
        # Set model name (custom name > config name > model type)
        self.model_name = model_name or wandb_config.run_name or model_type
        
        # Initialize W&B run
        self.run = wandb.init(
            project=wandb_config.project,
            name=wandb_config.run_name,
            tags=wandb_config.tags,
            entity=wandb_config.entity
        )
    
    def _detect_model_type(self, model: BaseEstimator) -> str:
        """Auto-detect model type from model instance."""
        if isinstance(model, LogisticRegression):
            return "logistic_regression"
        elif isinstance(model, XGBClassifier):
            return "xgboost"
        else:
            return "default"
    
    def _get_wandb_config(self, model_type: str):
        """Get the appropriate W&B config based on model type."""
        if model_type == "logistic_regression":
            return CONFIG.wandb_logreg
        elif model_type == "xgboost":
            return CONFIG.wandb_xgb
        else:
            # Fallback to base wandb config
            return CONFIG.wandb
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters to W&B config."""
        wandb.config.update(params)
    
    def log_best_params(self, best_params: Dict[str, Any]):
        """Log best parameters from grid search."""
        params_log = {f"best_{k}": v for k, v in best_params.items()}
        wandb.log(params_log)
    
    def log_metrics(self, metrics: Dict[str, float]):
        """Log training and test metrics."""
        wandb.log(metrics)
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, dataset: str = "test"):
        """Log confusion matrix visualization."""
        fig = self.visualizer.plot_confusion_matrix(
            y_true, y_pred, 
            title=f"{self.model_name} - {dataset.title()} Confusion Matrix"
        )
        
        wandb.log({f"{dataset}_confusion_matrix": wandb.Image(fig)})
        plt.close(fig)
    
    def log_precision_recall_curve(self, y_true: np.ndarray, y_prob: np.ndarray, dataset: str = "test", pos_label: int = 1):
        """Log precision-recall curve with threshold."""
        fig = self.visualizer.plot_precision_recall_threshold(
            y_true, y_prob, 
            pos_label=pos_label,
            title=f"{self.model_name} - {dataset.title()} Precision-Recall Curve"
        )
        
        wandb.log({f"{dataset}_precision_recall_curve": wandb.Image(fig)})
        plt.close(fig)
    
    def log_model_artifact(self, model: BaseEstimator, description: str = None):
        """Save and log best model as W&B artifact."""
        if description is None:
            description = f"Best {self.model_type} model from grid search"
        
        artifact_name = f"{self.model_name}_best_model"
        
        # Debug prints to see the naming process
        print(f"DEBUG - Artifact name: {artifact_name}")
        print(f"DEBUG - Model name: {self.model_name}")
        print(f"DEBUG - Model type: {self.model_type}")
        
        # Create artifact
        model_artifact = wandb.Artifact(
            name=artifact_name,
            type="model",
            description=description
        )
        
        # Create models directory if it doesn't exist
        os.makedirs(CONFIG.model.base_save_path, exist_ok=True)
        
        # Generate model filename with model type
        model_filename = f"{self.model_name}_{self.model_type}"
        model_path = CONFIG.model.get_model_name(model_filename)
        
        # Debug prints to see the file naming
        print(f"DEBUG - Model filename: {model_filename}")
        print(f"DEBUG - Full model path: {model_path}")
        print(f"DEBUG - Save format: {CONFIG.model.save_format}")
        
        # Save model
        if CONFIG.model.save_format == "pkl":
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        elif CONFIG.model.save_format == "joblib":
            joblib.dump(model, model_path)
        else:
            raise ValueError(f"Unsupported save format: {CONFIG.model.save_format}")
        
        # Debug: Check if file actually exists and show its actual name
        if os.path.exists(model_path):
            actual_filename = os.path.basename(model_path)
            print(f"DEBUG - File saved successfully as: {actual_filename}")
        else:
            print(f"DEBUG - ERROR: File was not saved at {model_path}")
        
        # Add file to artifact and log
        model_artifact.add_file(model_path)
        wandb.log_artifact(model_artifact)
        
        print(f"Model saved to: {model_path}")
        return model_path
    
    def finish(self):
        """Finish the W&B run."""
        wandb.finish()

    def log_feature_importance(self, 
                              model: BaseEstimator, 
                              feature_names: list,
                              top_n: int = 20,
                              show_direction: bool = False):
        """Log feature importance visualization.
        
        Args:
            model: Trained model (supports LogisticRegression, XGBoost, or Pipeline)
            feature_names: List of feature names
            top_n: Number of top features to display
            show_direction: If True, show coefficient direction (only for LogReg)
        """
        fig = self.visualizer.plot_feature_importance(
            model=model,
            feature_names=feature_names,
            top_n=top_n,
            title=f"{self.model_name} - Feature Importance",
            show_direction=show_direction
        )
        
        wandb.log({"feature_importance": wandb.Image(fig)})
        plt.close(fig)
        
        print(f"Feature importance plot logged to W&B")