"""
PyTest Configuration and Fixtures

This module provides shared fixtures and configuration for all tests.
Fixtures are reusable test data and setup/teardown logic.

Design Decision: Centralized fixtures reduce code duplication and ensure
consistent test data across all test modules.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta


@pytest.fixture(scope="session")
def test_data_dir():
    """
    Create a temporary directory for test data files.
    
    Scope: session - Created once per test session, shared across all tests.
    
    Yields:
        Path: Temporary directory path
    
    Design Decision: Session-scoped to avoid recreating directory for each test,
    improving test execution speed.
    """
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup after all tests complete
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def sample_flight_data():
    """
    Generate sample flight fare dataset for testing.
    
    Returns:
        pd.DataFrame: Synthetic flight data with realistic structure
    
    Design Decision: Synthetic data ensures tests don't depend on external
    files and can run in any environment.
    """
    np.random.seed(42)
    n_samples = 100
    
    airlines = ['Biman Bangladesh', 'US-Bangla', 'Novoair', 'Regent Airways']
    cities = ['Dhaka', 'Chittagong', 'Sylhet', 'Cox\'s Bazar', 'Jessore']
    
    # Generate dates
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_samples)]
    
    data = {
        'Date': dates,
        'Airline': np.random.choice(airlines, n_samples),
        'From': np.random.choice(cities, n_samples),
        'To': np.random.choice(cities, n_samples),
        'Fare': np.random.uniform(2000, 15000, n_samples),
        'Class': np.random.choice(['Economy', 'Business'], n_samples),
        'Duration': np.random.uniform(30, 180, n_samples),  # minutes
        'Stops': np.random.choice([0, 1, 2], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure From != To
    mask = df['From'] == df['To']
    df.loc[mask, 'To'] = df.loc[mask, 'From'].apply(
        lambda x: cities[(cities.index(x) + 1) % len(cities)]
    )
    
    return df


@pytest.fixture
def sample_flight_data_with_issues():
    """
    Generate flight data with common data quality issues.
    
    Returns:
        pd.DataFrame: Data with missing values, duplicates, and outliers
    
    Design Decision: Testing with problematic data ensures preprocessing
    logic handles real-world scenarios correctly.
    """
    np.random.seed(42)
    n_samples = 50
    
    airlines = ['Biman Bangladesh', 'US-Bangla', 'Novoair']
    cities = ['Dhaka', 'Chittagong', 'Sylhet']
    
    dates = [datetime(2024, 1, 1) + timedelta(days=i) for i in range(n_samples)]
    
    data = {
        'Date': dates,
        'Airline': np.random.choice(airlines, n_samples),
        'From': np.random.choice(cities, n_samples),
        'To': np.random.choice(cities, n_samples),
        'Fare': np.random.uniform(2000, 15000, n_samples),
        'Class': np.random.choice(['Economy', 'Business'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Add data quality issues
    # 1. Missing values
    df.loc[0:5, 'Fare'] = np.nan
    df.loc[10:12, 'Airline'] = np.nan
    
    # 2. Duplicate rows
    df = pd.concat([df, df.iloc[0:3]], ignore_index=True)
    
    # 3. Negative values (invalid)
    df.loc[20, 'Fare'] = -1000
    
    # 4. Extreme outliers
    df.loc[25, 'Fare'] = 500000
    
    return df


@pytest.fixture
def sample_csv_file(sample_flight_data, test_data_dir):
    """
    Create a temporary CSV file with sample data.
    
    Args:
        sample_flight_data: Fixture providing sample data
        test_data_dir: Fixture providing temp directory
    
    Returns:
        Path: Path to the CSV file
    
    Design Decision: Provides realistic file I/O testing without requiring
    actual data files in the repository.
    """
    csv_path = test_data_dir / "test_flight_data.csv"
    sample_flight_data.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def sample_processed_data():
    """
    Generate preprocessed data ready for modeling.
    
    Returns:
        pd.DataFrame: Clean, encoded, scaled data
    
    Design Decision: Allows testing of modeling components without running
    entire preprocessing pipeline.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 5
    
    # Generate numerical features
    data = {
        f'feature_{i}': np.random.randn(n_samples) for i in range(n_features)
    }
    
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def sample_train_test_data():
    """
    Generate train/test split data for model testing.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    
    Design Decision: Provides ready-to-use train/test sets for rapid
    model testing.
    """
    np.random.seed(42)
    n_train = 80
    n_test = 20
    n_features = 5
    
    X_train = pd.DataFrame(
        np.random.randn(n_train, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    X_test = pd.DataFrame(
        np.random.randn(n_test, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    
    y_train = pd.Series(np.random.uniform(2000, 15000, n_train), name='Fare')
    y_test = pd.Series(np.random.uniform(2000, 15000, n_test), name='Fare')
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def mock_config(tmp_path):
    """
    Create mock configuration for testing.
    
    Args:
        tmp_path: PyTest's built-in temporary directory fixture
    
    Returns:
        dict: Configuration dictionary
    
    Design Decision: Isolated config prevents tests from interfering with
    actual project configuration.
    """
    return {
        'BASE_DIR': tmp_path,
        'DATA_DIR': tmp_path / 'data',
        'RAW_DATA_DIR': tmp_path / 'data' / 'raw',
        'PROCESSED_DATA_DIR': tmp_path / 'data' / 'processed',
        'MODELS_DIR': tmp_path / 'models',
        'LOGS_DIR': tmp_path / 'logs',
        'FIGURES_DIR': tmp_path / 'reports' / 'figures',
        'RANDOM_STATE': 42,
        'TEST_SIZE': 0.2,
        'CV_FOLDS': 3  # Fewer folds for faster testing
    }


@pytest.fixture
def sample_correlation_matrix():
    """
    Generate sample correlation matrix for testing.
    
    Returns:
        pd.DataFrame: Correlation matrix
    """
    np.random.seed(42)
    n_features = 5
    
    # Generate random correlation matrix
    data = np.random.randn(100, n_features)
    df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(n_features)])
    
    return df.corr()


@pytest.fixture
def sample_model_results():
    """
    Generate sample model evaluation results.
    
    Returns:
        dict: Dictionary of model names to metrics
    
    Design Decision: Enables testing of evaluation logic without
    training actual models.
    """
    return {
        'Linear Regression': {
            'r2_score': 0.75,
            'rmse': 1500.0,
            'mae': 1200.0,
            'mape': 12.5
        },
        'Random Forest': {
            'r2_score': 0.85,
            'rmse': 1200.0,
            'mae': 950.0,
            'mape': 10.2
        },
        'Ridge Regression': {
            'r2_score': 0.76,
            'rmse': 1480.0,
            'mae': 1180.0,
            'mape': 12.3
        }
    }


# Parametrize fixtures for testing multiple scenarios
@pytest.fixture(params=['standard', 'minmax', 'robust'])
def scaling_methods(request):
    """
    Parametrized fixture for different scaling methods.
    
    Design Decision: Automatically runs tests with all scaling methods,
    ensuring comprehensive coverage.
    """
    return request.param


@pytest.fixture(params=['onehot', 'label'])
def encoding_methods(request):
    """
    Parametrized fixture for different encoding methods.
    """
    return request.param


# Hooks for custom test behavior
def pytest_configure(config):
    """
    Custom pytest configuration hook.
    
    Design Decision: Allows custom initialization before test run.
    """
    config.addinivalue_line(
        "markers", "requires_data: Tests that require actual data files"
    )


def pytest_collection_modifyitems(config, items):
    """
    Modify test collection to add markers automatically.
    
    Design Decision: Auto-marks slow tests based on naming convention.
    """
    for item in items:
        # Mark tests with 'slow' in name as slow
        if 'slow' in item.nodeid.lower():
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if 'integration' in item.nodeid.lower():
            item.add_marker(pytest.mark.integration)