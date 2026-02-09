"""
Model Training Module

This module handles model training, hyperparameter tuning, and cross-validation.
It provides a unified interface for training multiple regression algorithms.

Design Decision: Strategy pattern allows easy swapping of different models
while maintaining consistent training pipeline.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.config import config
from src.utils.logger import get_logger
from src.utils.helpers import save_pickle

logger = get_logger(__name__)


class ModelTrainer:
    """
    ModelTrainer class for training and optimizing regression models.
    
    Responsibilities:
    - Train multiple regression algorithms
    - Perform cross-validation
    - Hyperparameter tuning (GridSearch/RandomSearch)
    - Model persistence
    - Performance evaluation
    
    Design Pattern: Strategy pattern - encapsulates different training
    algorithms behind a common interface
    """
    
    def __init__(self, random_state: int = config.RANDOM_STATE):
        """
        Initialize ModelTrainer.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.best_models = {}
        self.training_results = {}
        logger.info(f"ModelTrainer initialized with random_state={random_state}")
    
    def train_baseline_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """
        Train baseline models without hyperparameter tuning.
        
        Args:
            X_train: Training features
            y_train: Training target
        
        Returns:
            Dictionary of trained models
        
        Design Decision: Baseline models provide quick performance benchmark
        before investing in hyperparameter optimization.
        """
        logger.info("Training baseline models...")
        
        baseline_models = {
            'Linear Regression': LinearRegression(),
            'Ridge (alpha=1.0)': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso (alpha=1.0)': Lasso(alpha=1.0, random_state=self.random_state),
            'Decision Tree': DecisionTreeRegressor(random_state=self.random_state, max_depth=10),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        }
        
        for name, model in baseline_models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Evaluate on training set
            y_train_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            
            self.training_results[name] = {
                'train_r2': train_r2,
                'train_rmse': train_rmse,
                'train_mae': train_mae
            }
            
            logger.info(f"{name} - Train R²: {train_r2:.4f}, RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}")
        
        logger.info("Baseline model training completed")
        return self.models
    
    def perform_cross_validation(self, model: Any, X: pd.DataFrame, y: pd.Series, 
                                 cv: int = config.CV_FOLDS) -> Dict[str, float]:
        """
        Perform k-fold cross-validation.
        
        Args:
            model: Scikit-learn model instance
            X: Features
            y: Target
            cv: Number of cross-validation folds
        
        Returns:
            Dictionary with CV scores
        
        Design Decision: Cross-validation provides robust estimate of
        generalization performance and reduces overfitting risk.
        """
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        # R² scores
        r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)
        
        # Negative MSE (sklearn returns negative for loss metrics)
        neg_mse_scores = cross_val_score(model, X, y, cv=cv, 
                                         scoring='neg_mean_squared_error', n_jobs=-1)
        rmse_scores = np.sqrt(-neg_mse_scores)
        
        # MAE
        neg_mae_scores = cross_val_score(model, X, y, cv=cv, 
                                         scoring='neg_mean_absolute_error', n_jobs=-1)
        mae_scores = -neg_mae_scores
        
        cv_results = {
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std(),
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std()
        }
        
        logger.info(f"CV Results - R²: {cv_results['r2_mean']:.4f} (±{cv_results['r2_std']:.4f})")
        logger.info(f"CV Results - RMSE: {cv_results['rmse_mean']:.2f} (±{cv_results['rmse_std']:.2f})")
        
        return cv_results
    
    def tune_hyperparameters(self, model_type: str, X_train: pd.DataFrame, 
                            y_train: pd.Series, param_grid: Dict, 
                            search_type: str = 'grid') -> Tuple[Any, Dict]:
        """
        Perform hyperparameter tuning using GridSearch or RandomizedSearch.
        
        Args:
            model_type: Type of model ('ridge', 'lasso', 'decision_tree', 'random_forest')
            X_train: Training features
            y_train: Training target
            param_grid: Hyperparameter grid
            search_type: 'grid' or 'random'
        
        Returns:
            Tuple of (best_model, best_params)
        
        Design Decision: Grid/Random search automates hyperparameter
        optimization and finds optimal model configuration.
        """
        logger.info(f"Tuning hyperparameters for {model_type} using {search_type} search...")
        
        # Select base model
        if model_type.lower() == 'ridge':
            base_model = Ridge(random_state=self.random_state)
        elif model_type.lower() == 'lasso':
            base_model = Lasso(random_state=self.random_state)
        elif model_type.lower() == 'decision_tree':
            base_model = DecisionTreeRegressor(random_state=self.random_state)
        elif model_type.lower() == 'random_forest':
            base_model = RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Perform search
        if search_type == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=config.CV_FOLDS,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=1
            )
        else:  # random
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=20,
                cv=config.CV_FOLDS,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                random_state=self.random_state,
                verbose=1
            )
        
        search.fit(X_train, y_train)
        
        best_model = search.best_estimator_
        best_params = search.best_params_
        best_score = -search.best_score_  # Convert back to positive RMSE
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best CV RMSE: {np.sqrt(best_score):.2f}")
        
        self.best_models[model_type] = best_model
        
        return best_model, best_params
    
    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: pd.Series, 
                      model_name: str) -> Dict[str, float]:
        """
        Evaluate model on test set.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model for logging
        
        Returns:
            Dictionary of evaluation metrics
        
        Design Decision: Comprehensive metrics (R², RMSE, MAE, MAPE) provide
        multi-faceted view of model performance.
        """
        logger.info(f"Evaluating {model_name} on test set...")
        
        y_pred = model.predict(X_test)
        
        metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / y_test)) * 100  # Mean Absolute Percentage Error
        }
        
        logger.info(f"{model_name} Test Metrics:")
        logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.2f}")
        logger.info(f"  MAE: {metrics['mae']:.2f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        
        return metrics
    
    def save_model(self, model: Any, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            model: Trained model to save
            filepath: Destination file path
        """
        save_pickle(model, filepath)
        logger.info(f"Model saved successfully to {filepath}")
    
    def get_feature_importance(self, model: Any, feature_names: list) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models or coefficients from linear models.
        
        Args:
            model: Trained model
            feature_names: List of feature names
        
        Returns:
            DataFrame with feature names and importance values
        
        Design Decision: Feature importance reveals which variables drive
        predictions and supports business insights.
        """
        if hasattr(model, 'feature_importances_'):
            # Tree-based models
            importances = model.feature_importances_
            importance_type = 'Feature Importance'
        elif hasattr(model, 'coef_'):
            # Linear models
            importances = np.abs(model.coef_)  # Absolute value of coefficients
            importance_type = 'Coefficient Magnitude'
        else:
            logger.warning("Model does not support feature importance extraction")
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        logger.info(f"Extracted {importance_type} for {len(feature_names)} features")
        logger.info(f"Top 5 features: {list(importance_df.head()['Feature'])}")
        
        return importance_df