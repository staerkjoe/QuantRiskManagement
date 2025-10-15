# src/data/loader.py
import pandas as pd
from ucimlrepo import fetch_ucirepo
from config.config import DataConfig

class DataLoader:
    """Handles data loading and initial setup."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.attribute_mapping = {
            "Attribute1": "checking_status",
            "Attribute2": "duration",
            "Attribute3": "credit_history",
            "Attribute4": "purpose",
            "Attribute5": "credit_amount",
            "Attribute6": "savings",
            "Attribute7": "employment",
            "Attribute8": "installment_rate",
            "Attribute9": "personal_status_sex",
            "Attribute10": "other_debtors",
            "Attribute11": "residence_since",
            "Attribute12": "property",
            "Attribute13": "age",
            "Attribute14": "other_installment",
            "Attribute15": "housing",
            "Attribute16": "existing_credits",
            "Attribute17": "job",
            "Attribute18": "people_liable",
            "Attribute19": "telephone",
            "Attribute20": "foreign_worker",
            "class": "credit_risk"
        }
    
    def load_raw_data(self) -> pd.DataFrame:
        """Fetch and merge raw data from UCI repository."""
        dataset = fetch_ucirepo(id=self.config.dataset_id)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        df = df.rename(columns=self.attribute_mapping)
        return df
    
    def prepare_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert target variable: 1=Good -> 1, 2=Bad -> 0."""
        df = df.copy()
        df["credit_risk"] = df["credit_risk"].map({1: 1, 2: 0})
        return df