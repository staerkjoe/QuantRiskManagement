# src/models/logreg/trainer.py
from sklearn.model_selection import GridSearchCV
import pandas as pd
from models.pipelines import LogRegPipelineBuilder, XGBPipelineBuilder
from config.model_configs import LogRegConfig, XGBConfig
from config.config import DataConfig

class LogRegTrainer:
    """Handles LogisticRegression training and hyperparameter tuning."""
    
    def __init__(self, data_config: DataConfig, model_config: LogRegConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.builder = LogRegPipelineBuilder(data_config, model_config)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
        """Train LogisticRegression with GridSearchCV."""
        pipeline = self.builder.get_pipeline()
        
        grid_search = GridSearchCV(
            pipeline,
            self.model_config.param_grid,
            cv=self.model_config.cv_folds,
            scoring="f1",
            n_jobs=-1,
            verbose=2
        )
        
        print("=" * 60)
        print("Training LogisticRegression")
        print("=" * 60)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search

class XGBTrainer:
    """Handles XGBoost training and hyperparameter tuning."""
    
    def __init__(self, data_config: DataConfig, model_config: XGBConfig):
        self.data_config = data_config
        self.model_config = model_config
        self.builder = XGBPipelineBuilder(data_config, model_config)
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> GridSearchCV:
        """Train XGBoost with GridSearchCV."""
        pipeline = self.builder.get_pipeline()
        
        grid_search = GridSearchCV(
            pipeline,
            self.model_config.param_grid,
            cv=self.model_config.cv_folds,
            scoring="f1",
            n_jobs=-1,
            verbose=2
        )
        
        print("=" * 60)
        print("Training XGBoost")
        print("=" * 60)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV F1 score: {grid_search.best_score_:.4f}")
        
        return grid_search