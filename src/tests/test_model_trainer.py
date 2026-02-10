"""
Unit Tests for ModelTrainer Class

Tests model training functionality including baseline models,
cross-validation, hyperparameter tuning, and evaluation.

Test Coverage:
- Model initialization
- Baseline model training
- Cross-validation
- Hyperparameter tuning
- Model evaluation
- Feature importance extraction
"""

import pytest
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from src.models.trainer import ModelTrainer


class TestModelTrainer:
    """Test suite for ModelTrainer class."""
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test ModelTrainer initialization."""
        trainer = ModelTrainer(random_state=42)
        
        assert trainer.random_state == 42
        assert trainer.models == {}
        assert trainer.best_models == {}
        assert trainer.training_results == {}
    
    @pytest.mark.unit
    def test_train_baseline_models(self, sample_train_test_data):
        """Test training baseline models."""
        X_train, _, y_train, _ = sample_train_test_data
        
        trainer = ModelTrainer()
        models = trainer.train_baseline_models(X_train, y_train)
        
        assert len(models) >= 3  # At least Linear, Ridge, Lasso
        assert 'Linear Regression' in models
        assert all(hasattr(model, 'predict') for model in models.values())
    
    @pytest.mark.unit
    def test_train_baseline_models_stores_results(self, sample_train_test_data):
        """Test that training stores results."""
        X_train, _, y_train, _ = sample_train_test_data
        
        trainer = ModelTrainer()
        trainer.train_baseline_models(X_train, y_train)
        
        assert len(trainer.training_results) > 0
        assert 'train_r2' in trainer.training_results['Linear Regression']
        assert 'train_rmse' in trainer.training_results['Linear Regression']
    
    @pytest.mark.unit
    def test_perform_cross_validation(self, sample_train_test_data):
        """Test cross-validation."""
        X_train, _, y_train, _ = sample_train_test_data
        
        model = LinearRegression()
        trainer = ModelTrainer()
        cv_results = trainer.perform_cross_validation(model, X_train, y_train, cv=3)
        
        assert 'r2_mean' in cv_results
        assert 'r2_std' in cv_results
        assert 'rmse_mean' in cv_results
        assert 'mae_mean' in cv_results
    
    @pytest.mark.unit
    def test_perform_cross_validation_values(self, sample_train_test_data):
        """Test that CV results are within expected ranges."""
        X_train, _, y_train, _ = sample_train_test_data
        
        model = LinearRegression()
        trainer = ModelTrainer()
        cv_results = trainer.perform_cross_validation(model, X_train, y_train, cv=3)
        
        # R2 should be between -inf and 1
        assert cv_results['r2_mean'] <= 1.0
        # RMSE and MAE should be positive
        assert cv_results['rmse_mean'] > 0
        assert cv_results['mae_mean'] > 0
        # Standard deviations should be non-negative
        assert cv_results['r2_std'] >= 0
        assert cv_results['rmse_std'] >= 0
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_tune_hyperparameters_ridge(self, sample_train_test_data):
        """Test hyperparameter tuning for Ridge regression."""
        X_train, _, y_train, _ = sample_train_test_data
        
        param_grid = {'alpha': [0.1, 1.0, 10.0]}
        
        trainer = ModelTrainer()
        best_model, best_params = trainer.tune_hyperparameters(
            'ridge', X_train, y_train, param_grid, search_type='grid'
        )
        
        assert best_model is not None
        assert 'alpha' in best_params
        assert best_params['alpha'] in param_grid['alpha']
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_tune_hyperparameters_random_forest(self, sample_train_test_data):
        """Test hyperparameter tuning for Random Forest."""
        X_train, _, y_train, _ = sample_train_test_data
        
        param_grid = {
            'n_estimators': [10, 20],
            'max_depth': [5, 10]
        }
        
        trainer = ModelTrainer()
        best_model, best_params = trainer.tune_hyperparameters(
            'random_forest', X_train, y_train, param_grid, search_type='grid'
        )
        
        assert isinstance(best_model, RandomForestRegressor)
        assert 'n_estimators' in best_params
        assert 'max_depth' in best_params
    
    @pytest.mark.unit
    def test_tune_hyperparameters_stores_best_model(self, sample_train_test_data):
        """Test that tuning stores best model."""
        X_train, _, y_train, _ = sample_train_test_data
        
        param_grid = {'alpha': [0.1, 1.0]}
        
        trainer = ModelTrainer()
        trainer.tune_hyperparameters('ridge', X_train, y_train, param_grid)
        
        assert 'ridge' in trainer.best_models
    
    @pytest.mark.unit
    def test_evaluate_model(self, sample_train_test_data):
        """Test model evaluation."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        trainer = ModelTrainer()
        metrics = trainer.evaluate_model(model, X_test, y_test, 'Test Model')
        
        assert 'r2_score' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'mape' in metrics
    
    @pytest.mark.unit
    def test_evaluate_model_metrics_types(self, sample_train_test_data):
        """Test that evaluation metrics have correct types."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        trainer = ModelTrainer()
        metrics = trainer.evaluate_model(model, X_test, y_test, 'Test')
        
        assert isinstance(metrics['r2_score'], (float, np.floating))
        assert isinstance(metrics['rmse'], (float, np.floating))
        assert isinstance(metrics['mae'], (float, np.floating))
        assert metrics['rmse'] > 0
        assert metrics['mae'] > 0
    
    @pytest.mark.unit
    def test_get_feature_importance_tree_model(self, sample_train_test_data):
        """Test feature importance extraction from tree-based model."""
        X_train, _, y_train, _ = sample_train_test_data
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        trainer = ModelTrainer()
        importance_df = trainer.get_feature_importance(model, X_train.columns.tolist())
        
        assert not importance_df.empty
        assert 'Feature' in importance_df.columns
        assert 'Importance' in importance_df.columns
        assert len(importance_df) == len(X_train.columns)
    
    @pytest.mark.unit
    def test_get_feature_importance_linear_model(self, sample_train_test_data):
        """Test feature importance extraction from linear model."""
        X_train, _, y_train, _ = sample_train_test_data
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        trainer = ModelTrainer()
        importance_df = trainer.get_feature_importance(model, X_train.columns.tolist())
        
        assert not importance_df.empty
        assert 'Feature' in importance_df.columns
        assert 'Importance' in importance_df.columns
    
    @pytest.mark.unit
    def test_get_feature_importance_sorted(self, sample_train_test_data):
        """Test that feature importance is sorted descending."""
        X_train, _, y_train, _ = sample_train_test_data
        
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        trainer = ModelTrainer()
        importance_df = trainer.get_feature_importance(model, X_train.columns.tolist())
        
        # Check that importances are in descending order
        importances = importance_df['Importance'].values
        assert all(importances[i] >= importances[i+1] for i in range(len(importances)-1))


class TestModelTrainerIntegration:
    """Integration tests for complete training workflows."""
    
    @pytest.mark.integration
    def test_complete_training_pipeline(self, sample_train_test_data):
        """Test complete model training and evaluation pipeline."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        
        trainer = ModelTrainer()
        
        # Train baseline models
        models = trainer.train_baseline_models(X_train, y_train)
        
        # Evaluate each model
        for name, model in models.items():
            metrics = trainer.evaluate_model(model, X_test, y_test, name)
            assert all(key in metrics for key in ['r2_score', 'rmse', 'mae'])
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_training_with_cv_and_tuning(self, sample_train_test_data):
        """Test training with cross-validation and hyperparameter tuning."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        
        trainer = ModelTrainer()
        
        # Train baseline
        trainer.train_baseline_models(X_train, y_train)
        
        # Perform CV on one model
        model = LinearRegression()
        cv_results = trainer.perform_cross_validation(model, X_train, y_train, cv=3)
        
        # Tune hyperparameters
        param_grid = {'alpha': [0.1, 1.0]}
        best_model, _ = trainer.tune_hyperparameters('ridge', X_train, y_train, param_grid)
        
        # Evaluate tuned model
        metrics = trainer.evaluate_model(best_model, X_test, y_test, 'Tuned Ridge')
        
        assert cv_results is not None
        assert metrics is not None


class TestModelTrainerEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_small_dataset(self):
        """Test training with very small dataset."""
        X_train = pd.DataFrame({'A': [1, 2, 3, 4, 5]})
        y_train = pd.Series([1, 2, 3, 4, 5])
        
        trainer = ModelTrainer()
        models = trainer.train_baseline_models(X_train, y_train)
        
        assert len(models) > 0
    
    @pytest.mark.unit
    def test_single_feature(self):
        """Test training with single feature."""
        X_train = pd.DataFrame({'A': range(20)})
        y_train = pd.Series(range(20))
        X_test = pd.DataFrame({'A': range(5)})
        y_test = pd.Series(range(5))
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        trainer = ModelTrainer()
        metrics = trainer.evaluate_model(model, X_test, y_test, 'Single Feature')
        
        assert metrics is not None
    
    @pytest.mark.unit
    def test_perfect_predictions(self):
        """Test evaluation with perfect predictions."""
        X_train = pd.DataFrame({'A': range(10)})
        y_train = pd.Series(range(10))
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        trainer = ModelTrainer()
        metrics = trainer.evaluate_model(model, X_train, y_train, 'Perfect')
        
        # R2 should be 1.0 for perfect fit
        assert abs(metrics['r2_score'] - 1.0) < 1e-6
        # RMSE should be near 0
        assert metrics['rmse'] < 1e-6