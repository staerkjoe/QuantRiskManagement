from dataclasses import dataclass

@dataclass
class LogRegConfig:
    """LogisticRegression-specific configuration."""
    penalty: str = "l2"
    C: float = 0.0001
    solver: str = "liblinear"
    max_iter: int = 2000
    class_weight: str = None
    use_smote: bool = False
    
    # GridSearch parameters
    param_grid: dict = None
    cv_folds: int = 5
    
    # Model naming
    model_name: str = "logistic_regression"
    
    def __post_init__(self):
        self.param_grid = {
            "clf__penalty": ["l2"],
            "clf__C": [0.0001],
            "clf__solver": ["liblinear"],
        }

@dataclass
class XGBConfig:
    """XGBoost-specific configuration."""
    n_estimators: int = 500
    max_depth: int = 5
    learning_rate: float = 0.01
    subsample: float = 0.7
    colsample_bytree: float = 0.9
    reg_lambda: float = 10
    reg_alpha: float = 0.5
    gamma: float = 0
    tree_method: str = "hist"  # Will be set automatically based on GPU availability
    device: str = "cpu"  # Will be set automatically based on GPU availability
    
    # GridSearch parameters
    param_grid: dict = None
    cv_folds: int = 5
    
    # Model naming
    model_name: str = "xgboost"
    
    def _detect_gpu_availability(self) -> bool:
        """
        Detect if CUDA/GPU is available for XGBoost.
        
        Returns:
            bool: True if GPU is available, False otherwise.
        """
        try:
            import xgboost as xgb
            # Try to create a simple XGBoost model with GPU
            # This will fail if CUDA is not available
            dmat = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])
            model = xgb.train(
                params={"tree_method": "gpu_hist", "device": "cuda"},
                dtrain=dmat,
                num_boost_round=1,
                verbose_eval=False
            )
            return True
        except Exception:
            # Any exception means GPU is not available
            return False
    
    def __post_init__(self):
        # Automatically configure device settings based on GPU availability
        if self._detect_gpu_availability():
            self.tree_method = "gpu_hist"
            self.device = "cuda"
        else:
            self.tree_method = "hist"
            self.device = "cpu"
        
        self.param_grid = {
            "clf__n_estimators": [500, 750],
            "clf__max_depth": [5],
            "clf__learning_rate": [0.01],
            "clf__subsample": [0.7],
            "clf__colsample_bytree": [0.9],
            "clf__reg_lambda": [10],
            "clf__reg_alpha": [0.5],
            "clf__tree_method": [self.tree_method],
            "clf__device": [self.device],
        }