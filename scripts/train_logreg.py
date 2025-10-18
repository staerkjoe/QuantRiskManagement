import sys
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import CONFIG, DataConfig, LogRegWandBConfig
from config.model_configs import LogRegConfig
from utils.logger import WandBLogger
from data.dataloader import DataLoader
from data.preprocessing import DataPreprocessor, TrainTestSplitter
from models.training import LogRegTrainer

def main():
    """Train logistic regression with grid search."""
    
    # Load and preprocess data
    data_loader = DataLoader(CONFIG.data)
    raw_data = data_loader.load_data()
    
    preprocessor = DataPreprocessor(DataConfig())
    preprocessed_data = preprocessor.preprocess(raw_data)
    
    # Split data
    splitter = TrainTestSplitter()
    X_train, X_test, y_train, y_test = splitter.split_data(
        preprocessed_data, 'credit_risk', test_size=0.2, random_state=42, stratify=True
    )
    
    # Initialize trainer and perform grid search
    trainer = LogRegTrainer(DataConfig(), LogRegConfig())
    grid_search = trainer.train(X_train, y_train)
    best_model = grid_search.best_estimator_
    
    # Get predictions
    y_train_pred = best_model.predict(X_train)
    y_train_prob = best_model.predict_proba(X_train)[:, 1]
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'train_accuracy': accuracy_score(y_train, y_train_pred),
        'train_precision': precision_score(y_train, y_train_pred),
        'train_recall': recall_score(y_train, y_train_pred),
        'train_f1': f1_score(y_train, y_train_pred),
        'train_auc': roc_auc_score(y_train, y_train_prob),
        'test_accuracy': accuracy_score(y_test, y_test_pred),
        'test_precision': precision_score(y_test, y_test_pred),
        'test_recall': recall_score(y_test, y_test_pred),
        'test_f1': f1_score(y_test, y_test_pred),
        'test_auc': roc_auc_score(y_test, y_test_prob)
    }
    
    # Log to W&B
    logreg_wandb_config = LogRegWandBConfig()
    logger = WandBLogger(model_name='logistic_regression', config=logreg_wandb_config.__dict__)
    all_model_params = best_model.get_params()
    
    logger.log_hyperparameters(all_model_params)

    logger.log_metrics(metrics)
    logger.log_confusion_matrix(y_test, y_test_pred)
    logger.log_precision_recall_curve(y_test, y_test_prob)
    logger.log_model_artifact(best_model)
    
    # Extract feature names from the preprocessing pipeline
    if hasattr(best_model, 'named_steps'):
        # Get the preprocessor from the pipeline
        if 'preprocessor' in best_model.named_steps:
            preprocessor = best_model.named_steps['preprocessor']
            # Try to get feature names from the preprocessor
            if hasattr(preprocessor, 'get_feature_names_out'):
                feature_names = preprocessor.get_feature_names_out().tolist()
            else:
                # Fallback: use transformed data columns if available
                feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
        else:
            feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
    else:
        feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else [f'feature_{i}' for i in range(X_train.shape[1])]
    
    # Debug: Check what X_train looks like
    print(f"\nDEBUG: X_train type: {type(X_train)}")
    print(f"DEBUG: X_train shape: {X_train.shape}")
    if hasattr(X_train, 'columns'):
        print(f"DEBUG: X_train has columns attribute")
        feature_names = X_train.columns.tolist()
        print(f"DEBUG: First 5 columns: {feature_names[:5]}")
    else:
        print(f"DEBUG: X_train is a numpy array, no column names available")
        # You need to extract from the pipeline instead
    
    print(f"Number of features in model: {len(best_model[-1].coef_[0]) if hasattr(best_model, '__getitem__') else len(best_model.coef_[0])}")
    print(f"Number of feature names extracted: {len(feature_names)}")
    print(f"First few feature names: {feature_names[:5]}")
    
    logger.log_feature_importance(
        model=best_model,
        feature_names=feature_names,
        top_n=20,
        show_direction=True
    )
    
    print(f"Best F1 Score: {metrics['test_f1']:.4f}")
    print(f"Best Parameters: {grid_search.best_params_}")
    
    logger.finish()

if __name__ == "__main__":
    main()
