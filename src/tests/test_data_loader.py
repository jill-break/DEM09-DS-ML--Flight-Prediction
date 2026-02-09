"""
Unit Tests for DataLoader Class

Tests data loading functionality including file validation,
CSV parsing, and data quality reporting.

Test Coverage:
- File existence validation
- CSV loading
- Data shape validation
- Missing value detection
- Basic statistics generation
"""

import pytest
import pandas as pd
from pathlib import Path
from src.data.data_loader import DataLoader


class TestDataLoader:
    """Test suite for DataLoader class."""
    
    @pytest.mark.unit
    def test_initialization(self, sample_csv_file):
        """Test DataLoader initialization."""
        loader = DataLoader(sample_csv_file)
        assert loader.filepath == sample_csv_file
        assert loader.data is None
    
    @pytest.mark.unit
    def test_load_data_success(self, sample_csv_file):
        """Test successful data loading from CSV."""
        loader = DataLoader(sample_csv_file)
        data = loader.load_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert loader.data is not None
        assert data.equals(loader.data)
    
    @pytest.mark.unit
    def test_load_data_file_not_found(self, test_data_dir):
        """Test error handling when file doesn't exist."""
        non_existent_file = test_data_dir / "nonexistent.csv"
        loader = DataLoader(non_existent_file)
        
        with pytest.raises(FileNotFoundError):
            loader.load_data()
    
    @pytest.mark.unit
    def test_load_data_invalid_csv(self, test_data_dir):
        """Test error handling with malformed CSV."""
        invalid_csv = test_data_dir / "invalid.csv"
        # Create invalid CSV
        with open(invalid_csv, 'w') as f:
            f.write("invalid,csv,data\n")
            f.write("missing,columns\n")  # Inconsistent columns
        
        loader = DataLoader(invalid_csv)
        # Should load but may have issues - pandas is forgiving
        data = loader.load_data()
        assert isinstance(data, pd.DataFrame)
    
    @pytest.mark.unit
    def test_get_basic_statistics(self, sample_csv_file):
        """Test generation of basic statistics."""
        loader = DataLoader(sample_csv_file)
        loader.load_data()
        
        stats = loader.get_basic_statistics()
        assert isinstance(stats, pd.DataFrame)
        assert not stats.empty
        assert 'count' in stats.index or 'mean' in stats.index
    
    @pytest.mark.unit
    def test_get_basic_statistics_before_loading(self, sample_csv_file):
        """Test statistics generation fails before data loading."""
        loader = DataLoader(sample_csv_file)
        
        with pytest.raises(ValueError):
            loader.get_basic_statistics()
    
    @pytest.mark.unit
    def test_get_missing_value_report(self, sample_csv_file):
        """Test missing value report generation."""
        loader = DataLoader(sample_csv_file)
        loader.load_data()
        
        missing_report = loader.get_missing_value_report()
        assert isinstance(missing_report, pd.DataFrame)
        # Report may be empty if no missing values
        if not missing_report.empty:
            assert 'Column' in missing_report.columns
            assert 'Missing_Count' in missing_report.columns
            assert 'Missing_Percentage' in missing_report.columns
    
    @pytest.mark.unit
    def test_get_missing_value_report_with_missing_data(self, test_data_dir):
        """Test missing value report with data containing NaN."""
        # Create CSV with missing values
        df_with_missing = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': ['a', None, 'c', 'd'],
            'C': [1.1, 2.2, 3.3, 4.4]
        })
        csv_path = test_data_dir / "missing_data.csv"
        df_with_missing.to_csv(csv_path, index=False)
        
        loader = DataLoader(csv_path)
        loader.load_data()
        missing_report = loader.get_missing_value_report()
        
        assert len(missing_report) == 2  # A and B have missing values
        assert missing_report['Column'].tolist() == ['A', 'B'] or missing_report['Column'].tolist() == ['B', 'A']
    
    @pytest.mark.unit
    def test_data_shape_validation(self, sample_csv_file, sample_flight_data):
        """Test that loaded data matches expected shape."""
        loader = DataLoader(sample_csv_file)
        data = loader.load_data()
        
        assert data.shape[0] == len(sample_flight_data)
        assert data.shape[1] == len(sample_flight_data.columns)
    
    @pytest.mark.unit
    def test_column_names_preserved(self, sample_csv_file, sample_flight_data):
        """Test that column names are preserved during loading."""
        loader = DataLoader(sample_csv_file)
        data = loader.load_data()
        
        assert list(data.columns) == list(sample_flight_data.columns)
    
    @pytest.mark.unit
    def test_data_types_inference(self, sample_csv_file):
        """Test that pandas infers appropriate data types."""
        loader = DataLoader(sample_csv_file)
        data = loader.load_data()
        
        # Check that numerical columns are numeric
        assert pd.api.types.is_numeric_dtype(data['Fare'])
        # Check that string columns are object type
        assert pd.api.types.is_object_dtype(data['Airline'])


class TestDataLoaderEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_load_empty_csv(self, test_data_dir):
        """Test loading empty CSV file."""
        empty_csv = test_data_dir / "empty.csv"
        pd.DataFrame().to_csv(empty_csv, index=False)
        
        loader = DataLoader(empty_csv)
        data = loader.load_data()
        
        assert isinstance(data, pd.DataFrame)
        assert data.empty
    
    @pytest.mark.unit
    def test_load_single_row_csv(self, test_data_dir):
        """Test loading CSV with single data row."""
        single_row = pd.DataFrame({
            'A': [1],
            'B': ['test']
        })
        csv_path = test_data_dir / "single_row.csv"
        single_row.to_csv(csv_path, index=False)
        
        loader = DataLoader(csv_path)
        data = loader.load_data()
        
        assert len(data) == 1
        assert data.iloc[0]['A'] == 1
    
    @pytest.mark.unit
    def test_load_large_dataset(self, test_data_dir):
        """Test loading larger dataset for performance."""
        # Create larger dataset
        large_df = pd.DataFrame({
            'col1': range(10000),
            'col2': range(10000, 20000)
        })
        csv_path = test_data_dir / "large.csv"
        large_df.to_csv(csv_path, index=False)
        
        loader = DataLoader(csv_path)
        data = loader.load_data()
        
        assert len(data) == 10000
        assert list(data.columns) == ['col1', 'col2']