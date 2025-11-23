"""Data Processing Module for Student Performance Predictor

This module handles data loading, cleaning, validation, and preprocessing
for the student performance prediction system.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, Dict, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles all data processing operations for student performance prediction."""
    
    def __init__(self):
        """Initialize the DataProcessor with necessary components."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.categorical_features = []
        self.numerical_features = []
        logger.info("DataProcessor initialized")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file with error handling.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame containing the loaded data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
            return df
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Invalid file format: {str(e)}")
    
    def validate_data(self, df: pd.DataFrame, required_columns: List[str]) -> bool:
        """Validate that dataframe contains required columns.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            
        Returns:
            True if validation passes
            
        Raises:
            ValueError: If validation fails
        """
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info("Data validation passed")
        return True
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'mean') -> pd.DataFrame:
        """Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values ('mean', 'median', 'drop')
            
        Returns:
            DataFrame with missing values handled
        """
        df_copy = df.copy()
        initial_rows = len(df_copy)
        
        if strategy == 'drop':
            df_copy = df_copy.dropna()
            logger.info(f"Dropped {initial_rows - len(df_copy)} rows with missing values")
        else:
            for col in df_copy.select_dtypes(include=[np.number]).columns:
                if df_copy[col].isnull().any():
                    if strategy == 'mean':
                        df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                    elif strategy == 'median':
                        df_copy[col].fillna(df_copy[col].median(), inplace=True)
            
            # Forward fill for categorical
            for col in df_copy.select_dtypes(include=['object']).columns:
                df_copy[col].fillna(method='ffill', inplace=True)
                df_copy[col].fillna(method='bfill', inplace=True)
            
            logger.info(f"Missing values handled using {strategy} strategy")
        
        return df_copy
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """Encode categorical features using Label Encoding.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_copy = df.copy()
        
        for col in categorical_cols:
            if col in df_copy.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df_copy[col] = self.label_encoders[col].fit_transform(df_copy[col].astype(str))
                else:
                    df_copy[col] = self.label_encoders[col].transform(df_copy[col].astype(str))
                
                logger.info(f"Encoded categorical feature: {col}")
        
        return df_copy
    
    def scale_numerical_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Scale numerical features using StandardScaler.
        
        Args:
            X: Input features array
            fit: Whether to fit the scaler (True for training, False for testing)
            
        Returns:
            Scaled features array
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
            logger.info("Features scaled (fit_transform)")
        else:
            X_scaled = self.scaler.transform(X)
            logger.info("Features scaled (transform)")
        
        return X_scaled
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create engineered features for better prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with additional engineered features
        """
        df_copy = df.copy()
        
        # Example feature engineering
        if 'study_hours' in df_copy.columns and 'attendance' in df_copy.columns:
            df_copy['study_attendance_ratio'] = df_copy['study_hours'] / (df_copy['attendance'] + 1)
        
        if 'previous_marks' in df_copy.columns and 'assignment_score' in df_copy.columns:
            df_copy['avg_performance'] = (df_copy['previous_marks'] + df_copy['assignment_score']) / 2
        
        logger.info(f"Feature engineering complete. Total features: {df_copy.shape[1]}")
        return df_copy
    
    def split_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2, 
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split data into training and testing sets.
        
        Args:
            X: Features array
            y: Target array
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """Generate comprehensive data summary statistics.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'total_records': len(df),
            'total_features': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'numerical_features': df.select_dtypes(include=[np.number]).columns.tolist(),
            'categorical_features': df.select_dtypes(include=['object']).columns.tolist(),
            'statistics': df.describe().to_dict()
        }
        
        return summary
    
    def save_preprocessor(self, filepath: str):
        """Save the preprocessing components for later use.
        
        Args:
            filepath: Path to save the preprocessor
        """
        import joblib
        joblib.dump({
            'scaler': self.scaler,
            'label_encoders': self.label_encoders
        }, filepath)
        logger.info(f"Preprocessor saved to {filepath}")
    
    def load_preprocessor(self, filepath: str):
        """Load preprocessing components from file.
        
        Args:
            filepath: Path to load the preprocessor from
        """
        import joblib
        components = joblib.load(filepath)
        self.scaler = components['scaler']
        self.label_encoders = components['label_encoders']
        logger.info(f"Preprocessor loaded from {filepath}")
