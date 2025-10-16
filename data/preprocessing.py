# src/data/preprocessor.py
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional
from config.config import DataConfig

class DataPreprocessor:
    """Handles feature engineering and data preparation."""
    
    def __init__(self, config: DataConfig):
        self.config = config
    
    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features from existing ones."""
        df = df.copy()
        df["credit_per_duration"] = df["credit_amount"] / (df["duration"] + 1)
        df["credit_per_age"] = df["credit_amount"] / (df["age"] + 1)
        df["credit_per_existing"] = df["credit_amount"] / (df["existing_credits"] + 1)
        return df
    
    def validate_features(self, df: pd.DataFrame) -> bool:
        """Validate that all expected features are present."""
        all_features = (
            self.config.ordinal_features + 
            self.config.nominal_features + 
            self.config.num_features +
            ["credit_risk"]
        )
        missing = set(all_features) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return True
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full preprocessing pipeline."""
        df = self.create_derived_features(df)
        self.validate_features(df)
        return df

class TrainTestSplitter:
    """Handles train/test splitting of preprocessed data."""
    
    def __init__(self):
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
    
    def split_data(self, data: pd.DataFrame, target_column: str, 
                   test_size: float = 0.2, random_state: int = 42,
                   stratify: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Perform train/test split on preprocessed data.
        
        Args:
            data: Preprocessed DataFrame
            target_column: Name of target column
            test_size: Proportion of data for test set
            random_state: Random seed for reproducibility
            stratify: Whether to stratify the split based on target
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Perform train/test split
        stratify_param = y if stratify else None
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, 
            stratify=stratify_param
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_train_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get training data."""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not split. Call split_data() first.")
        return self.X_train, self.y_train
    
    def get_test_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Get test data."""
        if self.X_test is None or self.y_test is None:
            raise ValueError("Data not split. Call split_data() first.")
        return self.X_test, self.y_test