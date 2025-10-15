# src/config.py
from dataclasses import dataclass
from typing import List

@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    dataset_id: int = 144  # UCI ML Repo ID
    test_size: float = 0.2
    random_state: int = 19
    
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
class ProjectConfig:
    """Main project configuration."""
    data: DataConfig = None
    random_state: int = 42
    use_gpu: bool = True
    
    def __post_init__(self):
        self.data = DataConfig()

# Create a global config instance
CONFIG = ProjectConfig()