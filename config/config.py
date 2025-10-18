# src/config.py
from dataclasses import dataclass
from typing import List
import os

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    dataset_id: int = 144  # UCI ML Repo ID
    test_size: float = 0.2
    random_state: int = 21
    
    # Feature groups
    ordinal_features: List[str] = None
    nominal_features: List[str] = None
    num_features: List[str] = None
    
    def __post_init__(self):
        self.ordinal_features = [
            "checking_status", "credit_history", "savings", 
            "employment", "job"
        ]
        self.nominal_features = [
            "purpose", "personal_status_sex", "other_debtors", 
            "property", "other_installment", "housing", 
            "telephone", "foreign_worker"
        ]
        self.num_features = [
            "duration", "credit_amount", "installment_rate",
            "residence_since", "age", "existing_credits",
            "people_liable", "credit_per_existing", 
            "credit_per_age", "credit_per_duration"
        ]

@dataclass
class WandBConfig:
    """Base Weights & Biases logging configuration."""
    project: str = "QuantitativeRiskManagement"
    run_name: str = "BaseRun"
    tags: List[str] = None
    api_key: str = "1e4211a4b1feb397a67944a37634fd6d0b72686b"
    entity: str = "jojs-it-universitetet-i-k-benhavn"

@dataclass
class LogRegWandBConfig(WandBConfig):
    """Logistic Regression specific W&B configuration."""
    run_name: str = "LogReg1"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = ["LogReg", "Smote"]

@dataclass
class XGBWandBConfig(WandBConfig):
    """XGBoost specific W&B configuration."""
    run_name: str = "XGBoost1"
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = ["XGBoost", "NoSmote"]

@dataclass
class ModelConfig:
    """Model naming and saving configuration."""
    base_save_path: str = "models"
    model_prefix: str = "credit_risk"
    use_timestamp: bool = False
    save_format: str = "pkl"  # "pkl" or "joblib"
    
    def get_model_name(self, model_type: str, performance_metric: float = None) -> str:
        """Generate model filename based on configuration."""
        name_parts = [self.model_prefix, model_type]
        if performance_metric:
            name_parts.append(f"acc_{performance_metric:.3f}")
        
        filename = "_".join(name_parts) + f".{self.save_format}"
        return os.path.join(self.base_save_path, filename)

@dataclass
class ProjectConfig:
    """Main project configuration."""
    data: DataConfig = None
    wandb: WandBConfig = None
    model: ModelConfig = None
    random_state: int = 42
    use_gpu: bool = True
    
    def __post_init__(self):
        self.data = DataConfig()
        self.wandb = WandBConfig()
        self.model = ModelConfig()

# Create a global config instance
CONFIG = ProjectConfig()