"""
Integration Tests

Tests the interaction between multiple components of the system.
These tests verify that modules work correctly together.

Test Coverage:
- End-to-end preprocessing pipeline
- Feature engineering + model training workflow
- Data loading + preprocessing + modeling
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator


class TestDataPipeline:
    """Integration tests for data pipeline."""
    
    @pytest.mark.integration
    def test_load_and_preprocess_workflow(self, sample_csv_file):
        """Test complete data loading and preprocessing workflow."""
        # Load data
        loader = DataLoader(sample_csv_file)
        data = loader.load_data()
        
        # Preprocess
        preprocessor = DataPreprocessor(data)
        cleaned = (preprocessor
                  .remove_duplicates()
                  .handle_missing_values()
                  .get_processed_data())
        
        assert len(cleaned) > 0
        assert cleaned.isnull().sum().sum() == 0  # No missing values
    
    @pytest.mark.integration
    def test_preprocess_and_feature_engineering(self, sample_flight_data):
        """Test preprocessing followed by feature engineering."""
        # Preprocess
        preprocessor = DataPreprocessor(sample_flight_data)
        cleaned = preprocessor.remove_duplicates().get_processed_data()
        
        # Engineer features
        engineer = FeatureEngineer()
        
        # Convert date if present
        if 'Date' in cleaned.columns:
            preprocessor.convert_date_columns(['Date'])
            cleaned = preprocessor.get_processed_data()
            cleaned = engineer.create_temporal_features(cleaned, 'Date')
        
        # Encode categoricals
        cat_cols = cleaned.select_dtypes(include=['object']).columns.tolist()
        if cat_cols:
            cleaned = engineer.encode_categorical_features(
                cleaned, cat_cols, method='label', fit=True
            )
        
        assert len(cleaned) > 0
        assert cleaned.select_dtypes(include=['object']).shape[1] == 0  # All encoded


class TestModelingPipeline:
    """Integration tests for modeling pipeline."""
    
    @pytest.mark.integration
    def test_feature_engineering_and_training(self, sample_train_test_data):
        """Test feature engineering followed by model training."""
        X_train, X_test, y_train, y_test = sample_train_test_data
        
        # Feature engineering
        engineer = FeatureEngineer()
        X_train_scaled = engineer.scale_numerical_features(
            X_train, X_train.columns.tolist(), method='standard', fit=True
        )
        X_test_scaled = engineer.scale_numerical_features(
            X_test, X_test.columns.tolist(), method='standard', fit=False
        )
        
        # Train model
        trainer = ModelTrainer()
        models = trainer.train_baseline_models(X_train_scaled, y_train)
        
        # Evaluate
        evaluator = ModelEvaluator()
        for name, model in list(models.items())[:2]:  # Test first 2 models
            metrics = trainer.evaluate_model(model, X_test_scaled, y_test, name)
            evaluator.add_model_results(name, metrics)
        
        comparison = evaluator.create_comparison_table()
        
        assert len(models) > 0
        assert not comparison.empty
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_complete_ml_pipeline(self, sample_csv_file):
        """Test complete ML pipeline from data loading to evaluation."""
        # 1. Load data
        loader = DataLoader(sample_csv_file)
        data = loader.load_data()
        
        # 2. Preprocess
        preprocessor = DataPreprocessor(data)
        cleaned = (preprocessor
                  .remove_duplicates()
                  .handle_missing_values()
                  .get_processed_data())
        
        # 3. Split data (using Fare as target)
        if 'Fare' in cleaned.columns:
            prep_split = DataPreprocessor(cleaned)
            X_train, X_test, y_train, y_test = prep_split.split_data('Fare')
            
            # 4. Feature engineering
            engineer = FeatureEngineer()
            
            # Encode categorical features
            cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
            if cat_cols:
                X_train = engineer.encode_categorical_features(
                    X_train, cat_cols, method='label', fit=True
                )
                X_test = engineer.encode_categorical_features(
                    X_test, cat_cols, method='label', fit=False
                )
            
            # Scale numerical features
            num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
            if num_cols:
                X_train = engineer.scale_numerical_features(
                    X_train, num_cols, method='standard', fit=True
                )
                X_test = engineer.scale_numerical_features(
                    X_test, num_cols, method='standard', fit=False
                )
            
            # 5. Train models
            trainer = ModelTrainer()
            models = trainer.train_baseline_models(X_train, y_train)
            
            # 6. Evaluate
            evaluator = ModelEvaluator()
            for name, model in list(models.items())[:2]:
                metrics = trainer.evaluate_model(model, X_test, y_test, name)
                evaluator.add_model_results(name, metrics)
            
            # 7. Get best model
            comparison = evaluator.create_comparison_table()
            best_name, best_score = evaluator.get_best_model()
            
            assert len(models) > 0
            assert not comparison.empty
            assert best_name is not None


class TestErrorHandling:
    """Integration tests for error handling across modules."""
    
    @pytest.mark.integration
    def test_pipeline_with_missing_data(self, sample_flight_data_with_issues):
        """Test pipeline handles data quality issues gracefully."""
        # This should handle duplicates, missing values, and outliers
        preprocessor = DataPreprocessor(sample_flight_data_with_issues)
        cleaned = (preprocessor
                  .remove_duplicates()
                  .handle_missing_values()
                  .fix_invalid_values('Fare', min_value=0)
                  .get_processed_data())
        
        # Should have removed problems
        assert len(cleaned) < len(sample_flight_data_with_issues)
        assert cleaned['Fare'].min() >= 0
        assert cleaned.isnull().sum().sum() == 0
    
    @pytest.mark.integration
    def test_pipeline_with_minimal_data(self):
        """Test pipeline with minimal valid data."""
        # Create minimal dataset
        minimal_data = pd.DataFrame({
            'Feature1': [1, 2, 3, 4, 5],
            'Feature2': [5, 4, 3, 2, 1],
            'Target': [10, 20, 15, 25, 30]
        })
        
        preprocessor = DataPreprocessor(minimal_data)
        X_train, X_test, y_train, y_test = preprocessor.split_data('Target', test_size=0.2)
        
        trainer = ModelTrainer()
        models = trainer.train_baseline_models(X_train, y_train)
        
        # Should still train successfully
        assert len(models) > 0