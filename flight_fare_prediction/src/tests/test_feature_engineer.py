"""
Unit Tests for FeatureEngineer Class

Tests feature engineering functionality including temporal features,
encoding, scaling, and feature transformations.

Test Coverage:
- Temporal feature extraction
- Categorical encoding (OneHot, Label)
- Numerical scaling (Standard, MinMax, Robust)
- Feature creation and transformation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from src.features.feature_engineer import FeatureEngineer


class TestFeatureEngineer:
    """Test suite for FeatureEngineer class."""
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        
        assert engineer.scalers == {}
        assert engineer.encoders == {}
        assert engineer.feature_names == []
    
    @pytest.mark.unit
    def test_create_temporal_features(self):
        """Test temporal feature creation from datetime column."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=5),
            'value': [1, 2, 3, 4, 5]
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_temporal_features(df, 'date')
        
        assert 'month' in result.columns
        assert 'day' in result.columns
        assert 'weekday' in result.columns
        assert 'season' in result.columns
        assert 'is_weekend' in result.columns
    
    @pytest.mark.unit
    def test_create_temporal_features_values(self):
        """Test that temporal features have correct values."""
        df = pd.DataFrame({
            'date': [datetime(2024, 6, 15)]  # June 15, 2024 (Saturday)
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_temporal_features(df, 'date')
        
        assert result['month'].iloc[0] == 6
        assert result['day'].iloc[0] == 15
        assert result['weekday'].iloc[0] == 5  # Saturday
        assert result['season'].iloc[0] == 'Summer'
        assert result['is_weekend'].iloc[0] == 1
    
    @pytest.mark.unit
    def test_create_temporal_features_nonexistent_column(self):
        """Test temporal features with nonexistent column."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        engineer = FeatureEngineer()
        result = engineer.create_temporal_features(df, 'nonexistent')
        
        # Should return original dataframe
        assert result.equals(df)
    
    @pytest.mark.unit
    def test_encode_categorical_onehot(self):
        """Test OneHot encoding of categorical features."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C'],
            'value': [1, 2, 3, 4]
        })
        
        engineer = FeatureEngineer()
        result = engineer.encode_categorical_features(df, ['category'], method='onehot', fit=True)
        
        assert 'category' not in result.columns
        # OneHot creates n-1 columns (drop_first=True)
        encoded_cols = [col for col in result.columns if col.startswith('category_')]
        assert len(encoded_cols) == 2  # 3 categories - 1
    
    @pytest.mark.unit
    def test_encode_categorical_label(self):
        """Test Label encoding of categorical features."""
        df = pd.DataFrame({
            'category': ['A', 'B', 'A', 'C'],
            'value': [1, 2, 3, 4]
        })
        
        engineer = FeatureEngineer()
        result = engineer.encode_categorical_features(df, ['category'], method='label', fit=True)
        
        assert 'category' in result.columns
        # Label encoding should create integers
        assert pd.api.types.is_integer_dtype(result['category'])
        assert result['category'].nunique() == 3
    
    @pytest.mark.unit
    def test_encode_categorical_fit_transform_consistency(self):
        """Test that fit and transform produce consistent encodings."""
        df_train = pd.DataFrame({'cat': ['A', 'B', 'C']})
        df_test = pd.DataFrame({'cat': ['A', 'B', 'C']})
        
        engineer = FeatureEngineer()
        train_encoded = engineer.encode_categorical_features(df_train, ['cat'], method='onehot', fit=True)
        test_encoded = engineer.encode_categorical_features(df_test, ['cat'], method='onehot', fit=False)
        
        assert list(train_encoded.columns) == list(test_encoded.columns)
    
    @pytest.mark.unit
    def test_encode_categorical_unseen_category(self):
        """Test encoding with unseen categories in test set."""
        df_train = pd.DataFrame({'cat': ['A', 'B']})
        df_test = pd.DataFrame({'cat': ['A', 'B', 'C']})  # C is unseen
        
        engineer = FeatureEngineer()
        engineer.encode_categorical_features(df_train, ['cat'], method='onehot', fit=True)
        result = engineer.encode_categorical_features(df_test, ['cat'], method='onehot', fit=False)
        
        # Should handle unseen category gracefully
        assert result is not None
    
    @pytest.mark.unit
    @pytest.mark.parametrize("method", ['standard', 'minmax', 'robust'])
    def test_scale_numerical_features(self, method):
        """Test numerical feature scaling with different methods."""
        df = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [10, 20, 30, 40, 50]
        })
        
        engineer = FeatureEngineer()
        result = engineer.scale_numerical_features(df, ['num1', 'num2'], method=method, fit=True)
        
        # After scaling, mean should be close to 0 for standard scaler
        if method == 'standard':
            assert abs(result['num1'].mean()) < 1e-10
            assert abs(result['num2'].mean()) < 1e-10
        
        # For minmax, values should be between 0 and 1
        if method == 'minmax':
            assert result['num1'].min() >= 0
            assert result['num1'].max() <= 1
    
    @pytest.mark.unit
    def test_scale_numerical_features_fit_transform(self):
        """Test scaling fit and transform separately."""
        df_train = pd.DataFrame({'num': [1, 2, 3, 4, 5]})
        df_test = pd.DataFrame({'num': [2, 3, 4]})
        
        engineer = FeatureEngineer()
        train_scaled = engineer.scale_numerical_features(df_train, ['num'], method='standard', fit=True)
        test_scaled = engineer.scale_numerical_features(df_test, ['num'], method='standard', fit=False)
        
        # Test should use same scaler fitted on train
        assert 'num' in engineer.scalers
        assert test_scaled is not None
    
    @pytest.mark.unit
    def test_create_interaction_features(self):
        """Test creation of interaction features."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6]
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_interaction_features(df, [('A', 'B')])
        
        assert 'A_x_B' in result.columns
        assert result['A_x_B'].iloc[0] == 4  # 1 * 4
        assert result['A_x_B'].iloc[1] == 10  # 2 * 5
    
    @pytest.mark.unit
    def test_create_interaction_features_multiple(self):
        """Test creation of multiple interaction features."""
        df = pd.DataFrame({
            'A': [1, 2],
            'B': [3, 4],
            'C': [5, 6]
        })
        
        engineer = FeatureEngineer()
        result = engineer.create_interaction_features(df, [('A', 'B'), ('A', 'C')])
        
        assert 'A_x_B' in result.columns
        assert 'A_x_C' in result.columns
    
    @pytest.mark.unit
    def test_create_interaction_features_missing_column(self):
        """Test interaction creation with missing column."""
        df = pd.DataFrame({'A': [1, 2]})
        
        engineer = FeatureEngineer()
        result = engineer.create_interaction_features(df, [('A', 'B')])
        
        # Should not create interaction for missing column
        assert 'A_x_B' not in result.columns
    
    @pytest.mark.unit
    def test_drop_features(self):
        """Test dropping specified features."""
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })
        
        engineer = FeatureEngineer()
        result = engineer.drop_features(df, ['B', 'C'])
        
        assert 'A' in result.columns
        assert 'B' not in result.columns
        assert 'C' not in result.columns
    
    @pytest.mark.unit
    def test_drop_features_nonexistent(self):
        """Test dropping nonexistent features."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        engineer = FeatureEngineer()
        result = engineer.drop_features(df, ['B', 'C'])
        
        # Should not raise error
        assert 'A' in result.columns
        assert len(result.columns) == 1
    
    @pytest.mark.unit
    def test_get_feature_names(self):
        """Test getting feature names."""
        df = pd.DataFrame({
            'A': [1, 2],
            'B': [3, 4],
            'C': [5, 6]
        })
        
        engineer = FeatureEngineer()
        feature_names = engineer.get_feature_names(df)
        
        assert feature_names == ['A', 'B', 'C']
        assert engineer.feature_names == ['A', 'B', 'C']


class TestFeatureEngineerIntegration:
    """Integration tests for complete feature engineering workflows."""
    
    @pytest.mark.integration
    def test_complete_feature_pipeline(self):
        """Test complete feature engineering pipeline."""
        df = pd.DataFrame({
            'date': pd.date_range('2024-01-01', periods=10),
            'category': ['A', 'B'] * 5,
            'value': range(10)
        })
        
        engineer = FeatureEngineer()
        
        # Create temporal features
        df = engineer.create_temporal_features(df, 'date')
        
        # Encode categorical
        df = engineer.encode_categorical_features(df, ['category'], method='label', fit=True)
        
        # Scale numerical
        df = engineer.scale_numerical_features(df, ['value'], method='standard', fit=True)
        
        assert 'month' in df.columns
        assert 'category' in df.columns
        assert 'value' in df.columns
    
    @pytest.mark.integration
    def test_train_test_feature_consistency(self):
        """Test that feature engineering is consistent between train and test."""
        df_train = pd.DataFrame({
            'cat': ['A', 'B', 'C'] * 3,
            'num': range(9)
        })
        df_test = pd.DataFrame({
            'cat': ['A', 'B'] * 2,
            'num': range(4)
        })
        
        engineer = FeatureEngineer()
        
        # Fit on train
        train_processed = engineer.encode_categorical_features(df_train, ['cat'], method='label', fit=True)
        train_processed = engineer.scale_numerical_features(train_processed, ['num'], method='standard', fit=True)
        
        # Transform test
        test_processed = engineer.encode_categorical_features(df_test, ['cat'], method='label', fit=False)
        test_processed = engineer.scale_numerical_features(test_processed, ['num'], method='standard', fit=False)
        
        # Columns should match
        assert list(train_processed.columns) == list(test_processed.columns)


class TestFeatureEngineerEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_empty_dataframe(self):
        """Test feature engineering on empty DataFrame."""
        df = pd.DataFrame()
        engineer = FeatureEngineer()
        
        result = engineer.create_temporal_features(df, 'date')
        assert result.empty
    
    @pytest.mark.unit
    def test_single_category_encoding(self):
        """Test encoding with single category."""
        df = pd.DataFrame({'cat': ['A', 'A', 'A']})
        
        engineer = FeatureEngineer()
        result = engineer.encode_categorical_features(df, ['cat'], method='onehot', fit=True)
        
        # With drop_first=True, single category creates no columns
        assert 'cat' not in result.columns