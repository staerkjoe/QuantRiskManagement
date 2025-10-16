# src/models/logreg/pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from config.config import DataConfig
from config.model_configs import LogRegConfig, XGBConfig

class LogRegPipelineBuilder:
    """Builds preprocessing and full pipeline for LogisticRegression."""
    
    def __init__(self, data_config: DataConfig, model_config: LogRegConfig):
        self.data_config = data_config
        self.model_config = model_config
    
    def get_preprocessor(self) -> ColumnTransformer:
        """Create preprocessing pipeline (scaling needed for LogReg)."""
        return ColumnTransformer([
            ("nominal", OneHotEncoder(handle_unknown="ignore"), 
             self.data_config.nominal_features),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", 
             unknown_value=-1), self.data_config.ordinal_features),
            ("num", StandardScaler(), self.data_config.num_features)
        ])
    
    def get_pipeline(self) -> ImbPipeline:
        """Create full pipeline including model."""
        steps = [
            ("preprocess", self.get_preprocessor()),
        ]
        
        if self.model_config.use_smote:
            steps.append(("smote", SMOTE(random_state=42)))
        
        steps.append(("clf", LogisticRegression(
            penalty=self.model_config.penalty,
            C=self.model_config.C,
            solver=self.model_config.solver,
            max_iter=self.model_config.max_iter,
            class_weight=self.model_config.class_weight
        )))
        
        return ImbPipeline(steps=steps)
    
    def fit_pipeline(self, X_train, y_train) -> ImbPipeline:
        """Fit the pipeline using training data."""
        pipeline = self.get_pipeline()
        pipeline.fit(X_train, y_train)
        return pipeline
    


class XGBPipelineBuilder:
    """Builds preprocessing and full pipeline for XGBoost."""
    
    def __init__(self, data_config: DataConfig, model_config: XGBConfig):
        self.data_config = data_config
        self.model_config = model_config
    
    def get_preprocessor(self) -> ColumnTransformer:
        """Create preprocessing pipeline (no scaling needed for tree models)."""
        return ColumnTransformer([
            ("nominal", OrdinalEncoder(handle_unknown="use_encoded_value", 
             unknown_value=-1), self.data_config.nominal_features),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", 
             unknown_value=-1), self.data_config.ordinal_features),
            ("num", "passthrough", self.data_config.num_features)
        ])
    
    def get_pipeline(self) -> ImbPipeline:
        """Create full pipeline including model."""
        return ImbPipeline(steps=[
            ("preprocess", self.get_preprocessor()),
            ("clf", XGBClassifier(
                n_estimators=self.model_config.n_estimators,
                max_depth=self.model_config.max_depth,
                learning_rate=self.model_config.learning_rate,
                subsample=self.model_config.subsample,
                colsample_bytree=self.model_config.colsample_bytree,
                reg_lambda=self.model_config.reg_lambda,
                reg_alpha=self.model_config.reg_alpha,
                gamma=self.model_config.gamma,
                tree_method=self.model_config.tree_method,
                enable_categorical=True,
                eval_metric="logloss",
                random_state=42
            ))
        ])
    
    def fit_pipeline(self, X_train, y_train) -> ImbPipeline:
        """Fit the pipeline using training data."""
        pipeline = self.get_pipeline()
        pipeline.fit(X_train, y_train)
        return pipeline