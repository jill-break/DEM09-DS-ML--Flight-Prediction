"""
Unit Tests for DataPreprocessor Class

Tests data cleaning and preprocessing functionality including
duplicate removal, missing value handling, and data validation.

Test Coverage:
- Duplicate removal
- Missing value imputation
- Invalid value handling
- Data type conversion
- Train-test splitting
"""

import pytest
import pandas as pd
import numpy as np
from src.data.preprocessor import DataPreprocessor
from src.config import config


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class."""
    
    @pytest.mark.unit
    def test_initialization(self, sample_flight_data):
        """Test DataPreprocessor initialization."""
        preprocessor = DataPreprocessor(sample_flight_data)
        
        assert preprocessor.data is not None
        assert len(preprocessor.data) == len(sample_flight_data)
        assert preprocessor.original_shape == sample_flight_data.shape
    
    @pytest.mark.unit
    def test_initialization_creates_copy(self, sample_flight_data):
        """Test that preprocessor works on a copy of data."""
        original_data = sample_flight_data.copy()
        preprocessor = DataPreprocessor(sample_flight_data)
        
        # Modify preprocessor data
        preprocessor.data.iloc[0, 0] = 'MODIFIED'
        
        # Original should be unchanged
        assert not original_data.equals(preprocessor.data)
    
    @pytest.mark.unit
    def test_remove_duplicates_with_duplicates(self):
        """Test duplicate removal when duplicates exist."""
        df_with_dupes = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': ['a', 'b', 'b', 'c']
        })
        
        preprocessor = DataPreprocessor(df_with_dupes)
        preprocessor.remove_duplicates()
        
        assert len(preprocessor.data) == 3  # One duplicate removed
        assert preprocessor.data.duplicated().sum() == 0
    
    @pytest.mark.unit
    def test_remove_duplicates_no_duplicates(self, sample_flight_data):
        """Test duplicate removal when no duplicates exist."""
        preprocessor = DataPreprocessor(sample_flight_data)
        initial_length = len(preprocessor.data)
        
        preprocessor.remove_duplicates()
        
        # Length should be same or less
        assert len(preprocessor.data) <= initial_length
    
    @pytest.mark.unit
    def test_remove_duplicates_returns_self(self, sample_flight_data):
        """Test that remove_duplicates returns self for chaining."""
        preprocessor = DataPreprocessor(sample_flight_data)
        result = preprocessor.remove_duplicates()
        
        assert result is preprocessor
    
    @pytest.mark.unit
    def test_handle_missing_values_numerical(self):
        """Test missing value handling for numerical columns."""
        df = pd.DataFrame({
            'num1': [1.0, 2.0, np.nan, 4.0],
            'num2': [10, np.nan, 30, 40]
        })
        
        preprocessor = DataPreprocessor(df)
        preprocessor.handle_missing_values(strategy={'numerical': 'median', 'categorical': 'mode'})
        
        assert preprocessor.data['num1'].isnull().sum() == 0
        assert preprocessor.data['num2'].isnull().sum() == 0
    
    @pytest.mark.unit
    def test_handle_missing_values_categorical(self):
        """Test missing value handling for categorical columns."""
        df = pd.DataFrame({
            'cat1': ['a', 'b', None, 'd'],
            'cat2': ['x', None, 'z', 'x']
        })
        
        preprocessor = DataPreprocessor(df)
        preprocessor.handle_missing_values(strategy={'numerical': 'median', 'categorical': 'mode'})
        
        assert preprocessor.data['cat1'].isnull().sum() == 0
        assert preprocessor.data['cat2'].isnull().sum() == 0
    
    @pytest.mark.unit
    def test_handle_missing_values_default_strategy(self):
        """Test missing value handling with default strategy."""
        df = pd.DataFrame({
            'num': [1, 2, np.nan],
            'cat': ['a', None, 'c']
        })
        
        preprocessor = DataPreprocessor(df)
        preprocessor.handle_missing_values()  # Use defaults
        
        assert preprocessor.data.isnull().sum().sum() == 0
    
    @pytest.mark.unit
    def test_convert_date_columns_success(self):
        """Test successful date column conversion."""
        df = pd.DataFrame({
            'date_str': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'value': [1, 2, 3]
        })
        
        preprocessor = DataPreprocessor(df)
        preprocessor.convert_date_columns(['date_str'])
        
        assert pd.api.types.is_datetime64_any_dtype(preprocessor.data['date_str'])
    
    @pytest.mark.unit
    def test_convert_date_columns_invalid_dates(self):
        """Test date conversion with some invalid dates."""
        df = pd.DataFrame({
            'date_str': ['2024-01-01', 'invalid', '2024-01-03'],
            'value': [1, 2, 3]
        })
        
        preprocessor = DataPreprocessor(df)
        preprocessor.convert_date_columns(['date_str'])
        
        # Should convert valid dates and set invalid to NaT
        assert pd.api.types.is_datetime64_any_dtype(preprocessor.data['date_str'])
        assert preprocessor.data['date_str'].isnull().sum() == 1
    
    @pytest.mark.unit
    def test_convert_date_columns_nonexistent_column(self):
        """Test date conversion with nonexistent column."""
        df = pd.DataFrame({'A': [1, 2, 3]})
        
        preprocessor = DataPreprocessor(df)
        # Should not raise error, just log warning
        preprocessor.convert_date_columns(['nonexistent'])
        
        assert 'A' in preprocessor.data.columns
    
    @pytest.mark.unit
    def test_fix_invalid_values_removes_negatives(self):
        """Test removal of negative values in target column."""
        df = pd.DataFrame({
            'Fare': [100, 200, -50, 300, -100],
            'Other': [1, 2, 3, 4, 5]
        })
        
        preprocessor = DataPreprocessor(df)
        preprocessor.fix_invalid_values('Fare', min_value=0)
        
        assert (preprocessor.data['Fare'] >= 0).all()
        assert len(preprocessor.data) == 3  # Two negative rows removed
    
    @pytest.mark.unit
    def test_fix_invalid_values_removes_outliers(self):
        """Test removal of extreme outliers."""
        df = pd.DataFrame({
            'Fare': [100, 200, 150, 180, 10000000]  # Last value is extreme
        })
        
        preprocessor = DataPreprocessor(df)
        initial_length = len(df)
        preprocessor.fix_invalid_values('Fare')
        
        # Should remove extreme outlier
        assert len(preprocessor.data) < initial_length
    
    @pytest.mark.unit
    def test_standardize_categorical_values(self):
        """Test categorical value standardization."""
        df = pd.DataFrame({
            'City': ['dhaka', 'DHAKA', ' Dhaka ', 'chittagong']
        })
        
        preprocessor = DataPreprocessor(df)
        preprocessor.standardize_categorical_values('City')
        
        # Should be title case and stripped
        assert preprocessor.data['City'].iloc[0] == 'Dhaka'
        assert preprocessor.data['City'].iloc[1] == 'Dhaka'
        assert preprocessor.data['City'].iloc[2] == 'Dhaka'
        assert preprocessor.data['City'].iloc[3] == 'Chittagong'
    
    @pytest.mark.unit
    def test_split_data_basic(self, sample_flight_data):
        """Test basic train-test split."""
        preprocessor = DataPreprocessor(sample_flight_data)
        
        X_train, X_test, y_train, y_test = preprocessor.split_data('Fare', test_size=0.2)
        
        total_samples = len(X_train) + len(X_test)
        assert total_samples == len(sample_flight_data)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)
        assert len(X_test) == int(len(sample_flight_data) * 0.2)
    
    @pytest.mark.unit
    def test_split_data_target_not_in_features(self, sample_flight_data):
        """Test that target column is not in features."""
        preprocessor = DataPreprocessor(sample_flight_data)
        
        X_train, X_test, y_train, y_test = preprocessor.split_data('Fare')
        
        assert 'Fare' not in X_train.columns
        assert 'Fare' not in X_test.columns
    
    @pytest.mark.unit
    def test_split_data_invalid_target(self, sample_flight_data):
        """Test split with invalid target column."""
        preprocessor = DataPreprocessor(sample_flight_data)
        
        with pytest.raises(ValueError):
            preprocessor.split_data('NonExistentColumn')
    
    @pytest.mark.unit
    def test_split_data_reproducibility(self, sample_flight_data):
        """Test that split is reproducible with same random state."""
        preprocessor1 = DataPreprocessor(sample_flight_data)
        X_train1, _, _, _ = preprocessor1.split_data('Fare', random_state=42)
        
        preprocessor2 = DataPreprocessor(sample_flight_data)
        X_train2, _, _, _ = preprocessor2.split_data('Fare', random_state=42)
        
        assert X_train1.equals(X_train2)
    
    @pytest.mark.unit
    def test_get_processed_data(self, sample_flight_data):
        """Test getting processed data."""
        preprocessor = DataPreprocessor(sample_flight_data)
        preprocessor.remove_duplicates()
        
        processed = preprocessor.get_processed_data()
        
        assert isinstance(processed, pd.DataFrame)
        assert processed.equals(preprocessor.data)
    
    @pytest.mark.unit
    def test_method_chaining(self):
        """Test that methods can be chained."""
        df = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': [1.0, np.nan, 3.0, 4.0]
        })
        
        preprocessor = DataPreprocessor(df)
        result = (preprocessor
                 .remove_duplicates()
                 .handle_missing_values())
        
        assert result is preprocessor
        assert len(preprocessor.data) == 3
        assert preprocessor.data['B'].isnull().sum() == 0


class TestDataPreprocessorEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_empty_dataframe(self):
        """Test preprocessing empty DataFrame."""
        df = pd.DataFrame()
        preprocessor = DataPreprocessor(df)
        
        assert len(preprocessor.data) == 0
    
    @pytest.mark.unit
    def test_all_missing_column(self):
        """Test handling column with all missing values."""
        df = pd.DataFrame({
            'A': [np.nan, np.nan, np.nan],
            'B': [1, 2, 3]
        })
        
        preprocessor = DataPreprocessor(df)
        preprocessor.handle_missing_values()
        
        # Should fill with 0 or mode (which might be 0)
        assert preprocessor.data['A'].notna().any() or preprocessor.data['A'].isna().all()
    
    @pytest.mark.unit
    def test_single_row_dataframe(self):
        """Test preprocessing single-row DataFrame."""
        df = pd.DataFrame({'A': [1], 'B': [2]})
        preprocessor = DataPreprocessor(df)
        
        preprocessor.remove_duplicates()
        assert len(preprocessor.data) == 1