"""Model Training Module for Student Performance Predictor

This module handles training multiple ML models, hyperparameter tuning,
and model evaluation for student performance prediction.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score, GridSearchCV
import logging
import joblib
from typing import Dict, Tuple, List
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles training and evaluation of multiple regression models."""
    
    def __init__(self):
        """Initialize ModelTrainer with available models."""
        self.models = {
            'linear_regression': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'decision_tree': DecisionTreeRegressor(random_state=42),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'svr': SVR()
        }
        self.trained_models = {}
        self.best_model = None
        self.best_model_name = None
        self.evaluation_results = {}
        logger.info(f"ModelTrainer initialized with {len(self.models)} models")
    
    def train_single_model(self, model_name: str, X_train: np.ndarray, 
                          y_train: np.ndarray) -> object:
        """Train a single model.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model object
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found in available models")
        
        model = self.models[model_name]
        logger.info(f"Training {model_name}...")
        model.fit(X_train, y_train)
        self.trained_models[model_name] = model
        logger.info(f"{model_name} training complete")
        
        return model
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Train all available models.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary of trained models
        """
        logger.info("Starting training for all models...")
        
        for model_name in self.models.keys():
            try:
                self.train_single_model(model_name, X_train, y_train)
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        logger.info(f"Training complete. {len(self.trained_models)} models trained successfully")
        return self.trained_models
    
    def evaluate_model(self, model: object, X_test: np.ndarray, 
                      y_test: np.ndarray, model_name: str) -> Dict:
        """Evaluate a trained model on test data.
        
        Args:
            model: Trained model object
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2_score': r2_score(y_test, y_pred)
        }
        
        self.evaluation_results[model_name] = metrics
        logger.info(f"{model_name} - RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2_score']:.4f}")
        
        return metrics
    
    def evaluate_all_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info("Evaluating all trained models...")
        
        for model_name, model in self.trained_models.items():
            try:
                self.evaluate_model(model, X_test, y_test, model_name)
            except Exception as e:
                logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        return self.evaluation_results
    
    def get_best_model(self, metric: str = 'r2_score') -> Tuple[str, object, Dict]:
        """Get the best performing model based on specified metric.
        
        Args:
            metric: Metric to use for comparison ('r2_score', 'rmse', 'mae', 'mse')
            
        Returns:
            Tuple of (model_name, model_object, metrics)
        """
        if not self.evaluation_results:
            raise ValueError("No models have been evaluated yet")
        
        if metric in ['rmse', 'mae', 'mse']:
            # Lower is better
            best_model_name = min(self.evaluation_results, 
                                 key=lambda x: self.evaluation_results[x][metric])
        else:
            # Higher is better (r2_score)
            best_model_name = max(self.evaluation_results, 
                                 key=lambda x: self.evaluation_results[x][metric])
        
        self.best_model_name = best_model_name
        self.best_model = self.trained_models[best_model_name]
        
        logger.info(f"Best model: {best_model_name} with {metric}={self.evaluation_results[best_model_name][metric]:.4f}")
        
        return best_model_name, self.best_model, self.evaluation_results[best_model_name]
    
    def hyperparameter_tuning(self, model_name: str, param_grid: Dict, 
                            X_train: np.ndarray, y_train: np.ndarray) -> object:
        """Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            model_name: Name of the model to tune
            param_grid: Parameter grid for tuning
            X_train: Training features
            y_train: Training target
            
        Returns:
            Best model after tuning
        """
        logger.info(f"Starting hyperparameter tuning for {model_name}...")
        
        base_model = self.models[model_name]
        grid_search = GridSearchCV(base_model, param_grid, cv=5, 
                                  scoring='r2', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {grid_search.best_score_:.4f}")
        
        self.trained_models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_estimator_
    
    def cross_validate_model(self, model_name: str, X: np.ndarray, 
                           y: np.ndarray, cv: int = 5) -> Dict:
        """Perform cross-validation on a model.
        
        Args:
            model_name: Name of the model
            X: Features
            y: Target
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation scores
        """
        model = self.trained_models.get(model_name, self.models[model_name])
        
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        
        results = {
            'cv_scores': cv_scores.tolist(),
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std()
        }
        
        logger.info(f"{model_name} CV R2: {results['mean_score']:.4f} (+/- {results['std_score']:.4f})")
        
        return results
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions using a trained model.
        
        Args:
            model_name: Name of the model to use
            X: Features for prediction
            
        Returns:
            Predictions array
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained yet")
        
        return self.trained_models[model_name].predict(X)
    
    def save_model(self, model_name: str, filepath: str):
        """Save a trained model to disk.
        
        Args:
            model_name: Name of the model to save
            filepath: Path to save the model
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained yet")
        
        joblib.dump(self.trained_models[model_name], filepath)
        logger.info(f"Model {model_name} saved to {filepath}")
    
    def load_model(self, model_name: str, filepath: str):
        """Load a trained model from disk.
        
        Args:
            model_name: Name to assign to the loaded model
            filepath: Path to load the model from
        """
        self.trained_models[model_name] = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath} as {model_name}")
    
    def get_feature_importance(self, model_name: str) -> Dict:
        """Get feature importance for tree-based models.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with feature importances
        """
        if model_name not in self.trained_models:
            raise ValueError(f"Model {model_name} has not been trained yet")
        
        model = self.trained_models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            return {'feature_importances': importances.tolist()}
        else:
            logger.warning(f"Model {model_name} does not support feature importance")
            return {}
    
    def generate_report(self) -> Dict:
        """Generate comprehensive training and evaluation report.
        
        Returns:
            Dictionary containing complete training report
        """
        report = {
            'models_trained': list(self.trained_models.keys()),
            'evaluation_results': self.evaluation_results,
            'best_model': {
                'name': self.best_model_name,
                'metrics': self.evaluation_results.get(self.best_model_name, {})
            } if self.best_model_name else None
        }
        
        return report
    
    def save_report(self, filepath: str):
        """Save training report to JSON file.
        
        Args:
            filepath: Path to save the report
        """
        report = self.generate_report()
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=4)
        logger.info(f"Training report saved to {filepath}")
