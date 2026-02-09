"""
Data Loader Module

This module handles data ingestion and initial validation.
It provides a clean interface for loading raw datasets and performing
basic data quality checks.

Design Decision: Separating data loading from preprocessing follows
the Single Responsibility Principle and makes the pipeline modular.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
from src.utils.logger import get_logger
from src.utils.helpers import validate_file_exists

logger = get_logger(__name__)


class DataLoader:
    """
    DataLoader class for loading and validating raw datasets.
    
    Responsibilities:
    - Load CSV data into pandas DataFrame
    - Validate data integrity
    - Log dataset characteristics
    - Provide initial data quality insights
    
    Design Pattern: Single Responsibility - focuses only on data ingestion
    """
    
    def __init__(self, filepath: Path):
        """
        Initialize DataLoader with file path.
        
        Args:
            filepath: Path to the CSV file to load
        """
        self.filepath = filepath
        self.data: Optional[pd.DataFrame] = None
        logger.info(f"DataLoader initialized for file: {filepath}")
    
    def load_data(self) -> pd.DataFrame:
        """
        Load CSV data and perform initial validation.
        
        Returns:
            Loaded DataFrame
        
        Raises:
            FileNotFoundError: If file doesn't exist
            pd.errors.ParserError: If CSV is malformed
        
        Design Decision: Fail fast with clear error messages rather than
        silently handling missing files.
        """
        if not validate_file_exists(self.filepath):
            error_msg = f"Data file not found: {self.filepath}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        try:
            self.data = pd.read_csv(self.filepath)
            logger.info(f"Successfully loaded data from {self.filepath}")
            self._log_data_info()
            return self.data
        
        except pd.errors.ParserError as e:
            logger.error(f"Failed to parse CSV file: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading data: {str(e)}")
            raise
    
    def _log_data_info(self) -> None:
        """
        Log comprehensive dataset information.
        
        Logs:
        - Dataset shape (rows, columns)
        - Column names and data types
        - Missing value statistics
        - Memory usage
        
        Design Decision: Detailed logging helps with debugging and
        provides audit trail for data pipeline execution.
        """
        if self.data is None:
            logger.warning("No data loaded yet. Call load_data() first.")
            return
        
        logger.info("=" * 60)
        logger.info("DATASET INFORMATION")
        logger.info("=" * 60)
        logger.info(f"Shape: {self.data.shape[0]} rows × {self.data.shape[1]} columns")
        logger.info(f"Memory usage: {self.data.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        logger.info("\nColumn Information:")
        for col in self.data.columns:
            dtype = self.data[col].dtype
            null_count = self.data[col].isnull().sum()
            null_pct = (null_count / len(self.data)) * 100
            logger.info(f"  - {col}: {dtype} | Missing: {null_count} ({null_pct:.2f}%)")
        
        logger.info("=" * 60)
    
    def get_basic_statistics(self) -> pd.DataFrame:
        """
        Generate basic statistical summary of the dataset.
        
        Returns:
            DataFrame containing descriptive statistics
        
        Design Decision: Provides quick statistical overview for
        initial data understanding.
        """
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            raise ValueError("Data not loaded")
        
        stats = self.data.describe(include='all')
        logger.info("Generated basic statistics for dataset")
        return stats
    
    def get_missing_value_report(self) -> pd.DataFrame:
        """
        Generate detailed missing value report.
        
        Returns:
            DataFrame with columns: [Column, Missing_Count, Missing_Percentage]
        
        Design Decision: Explicit missing value analysis is critical for
        data quality assessment and preprocessing strategy.
        """
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            raise ValueError("Data not loaded")
        
        missing_df = pd.DataFrame({
            'Column': self.data.columns,
            'Missing_Count': self.data.isnull().sum().values,
            'Missing_Percentage': (self.data.isnull().sum().values / len(self.data) * 100)
        })
        
        missing_df = missing_df[missing_df['Missing_Count'] > 0].sort_values(
            'Missing_Percentage', ascending=False
        ).reset_index(drop=True)
        
        if len(missing_df) > 0:
            logger.info(f"Found missing values in {len(missing_df)} columns")
        else:
            logger.info("No missing values detected in dataset")
        
        return missing_df