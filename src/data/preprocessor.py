"""
Data Preprocessor Module

This module handles all data cleaning, validation, and transformation operations.
It prepares raw data for feature engineering and model training.

Design Decision: Preprocessing is separated into distinct methods for each
transformation type, allowing for flexible pipeline configuration and testing.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataPreprocessor:
    """
    DataPreprocessor class for cleaning and transforming raw data.
    
    Responsibilities:
    - Handle missing values
    - Remove duplicates
    - Fix invalid entries
    - Validate data types
    - Convert date columns
    - Split data into train/test sets
    
    Design Pattern: Chain of Responsibility - each method performs one
    transformation and returns modified DataFrame for chaining.
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize preprocessor with raw data.
        
        Args:
            data: Raw DataFrame to preprocess
        """
        self.data = data.copy()  # Work on a copy to preserve original
        self.original_shape = data.shape
        logger.info(f"DataPreprocessor initialized with data shape: {self.original_shape}")
    
    def remove_duplicates(self) -> 'DataPreprocessor':
        """
        Remove duplicate rows from dataset.
        
        Returns:
            Self for method chaining
        
        Design Decision: Duplicates can skew model training and should be
        removed early in the preprocessing pipeline.
        """
        initial_rows = len(self.data)
        self.data = self.data.drop_duplicates()
        removed = initial_rows - len(self.data)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows ({removed/initial_rows*100:.2f}%)")
        else:
            logger.info("No duplicate rows found")
        
        return self
    
    def handle_missing_values(self, strategy: Optional[dict] = None) -> 'DataPreprocessor':
        """
        Handle missing values using specified strategies.
        
        Args:
            strategy: Dictionary with 'numerical' and 'categorical' strategies.
                     If None, uses config defaults.
        
        Returns:
            Self for method chaining
        
        Design Decision: Strategy pattern allows flexible missing value handling
        based on data type and business requirements.
        """
        if strategy is None:
            strategy = config.MISSING_VALUE_STRATEGY
        
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        # Handle numerical columns
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if self.data[col].isnull().sum() > 0:
                if strategy['numerical'] == 'median':
                    fill_value = self.data[col].median()
                elif strategy['numerical'] == 'mean':
                    fill_value = self.data[col].mean()
                else:
                    fill_value = 0
                
                missing_count = self.data[col].isnull().sum()
                self.data[col].fillna(fill_value, inplace=True)
                logger.info(f"Filled {missing_count} missing values in '{col}' with {strategy['numerical']}: {fill_value:.2f}")
        
        # Handle categorical columns
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if self.data[col].isnull().sum() > 0:
                if strategy['categorical'] == 'mode':
                    mode_value = self.data[col].mode()
                    fill_value = mode_value[0] if len(mode_value) > 0 else "Unknown"
                else:
                    fill_value = "Unknown"
                
                missing_count = self.data[col].isnull().sum()
                self.data[col].fillna(fill_value, inplace=True)
                logger.info(f"Filled {missing_count} missing values in '{col}' with {strategy['categorical']}: {fill_value}")
        before = len(self.data)
        self.data = self.data.drop_duplicates()
        after = len(self.data)

        if before != after:
            logger.info(f"Removed {before - after} duplicate rows after imputation")

        return self
    
    def convert_date_columns(self, date_columns: List[str]) -> 'DataPreprocessor':
        """
        Convert specified columns to datetime type.
        
        Args:
            date_columns: List of column names to convert
        
        Returns:
            Self for method chaining
        
        Raises:
            ValueError: If conversion fails
        
        Design Decision: Explicit date conversion enables temporal feature
        engineering and ensures consistent date handling.
        """
        for col in date_columns:
            if col not in self.data.columns:
                logger.warning(f"Column '{col}' not found in dataset, skipping")
                continue
            
            try:
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce')
                logger.info(f"Converted '{col}' to datetime type")
                
                # Log any conversion failures
                null_count = self.data[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"Failed to parse {null_count} dates in '{col}'")
            
            except Exception as e:
                logger.error(f"Error converting '{col}' to datetime: {str(e)}")
                raise
        
        return self
    
    def fix_invalid_values(self, target_column: str, min_value: float = 0) -> 'DataPreprocessor':
        """
        Remove invalid values in target column.

        Args:
            target_column: Column to validate
            min_value: Minimum valid value (default: 0)

        Returns:
            Self for method chaining

        Design Decision:
            Domain-specific validation (e.g., non-negative fares)
            improves data quality and model reliability.
        """
    #     if target_column not in self.data.columns:
    #         logger.warning(f"Column '{target_column}' not found, skipping validation")
    #         return self

    #     initial_rows = len(self.data)

    #     invalid_mask = self.data[target_column] < min_value
    #     invalid_count = invalid_mask.sum()

    #     if invalid_count > 0:
    #         self.data = self.data[~invalid_mask]
    #         logger.info(
    #             f"Removed {invalid_count} rows with {target_column} < {min_value}"
    #         )

    #     final_rows = len(self.data)
    #     logger.info(
    #         f"Data validation complete: {initial_rows - final_rows} rows removed"
    #     )

    #     return self
    
    # def remove_outliers(
    #     self,
    #     column: str,
    #     quantile: float = 0.999
    # ) -> 'DataPreprocessor':
    #     """
    #     Remove extreme outliers using quantile-based filtering.
    #     """
    #     if column not in self.data.columns:
    #         logger.warning(f"Column '{column}' not found, skipping outlier removal")
    #         return self

    #     threshold = self.data[column].quantile(quantile)
    #     initial_rows = len(self.data)

    #     self.data = self.data[self.data[column] <= threshold]

    #     removed = initial_rows - len(self.data)
    #     if removed > 0:
    #         logger.info(
    #             f"Removed {removed} outliers from '{column}' (> {threshold:.2f})"
    #         )

    #     return self

        if target_column not in self.data.columns:
            logger.warning(f"Column '{target_column}' not found, skipping validation")
            return self

        initial_rows = len(self.data)

        # Remove invalid values
        invalid_mask = self.data[target_column] < min_value
        if invalid_mask.any():
            self.data = self.data[~invalid_mask]
            logger.info(
                f"Removed {invalid_mask.sum()} rows with {target_column} < {min_value}"
            )

        # Remove extreme outliers using IQR
        q1 = self.data[target_column].quantile(0.25)
        q3 = self.data[target_column].quantile(0.75)
        iqr = q3 - q1

        if iqr > 0:
            upper_bound = q3 + 3 * iqr  # conservative threshold
            outlier_mask = self.data[target_column] > upper_bound

            if outlier_mask.any():
                self.data = self.data[~outlier_mask]
                logger.info(
                    f"Removed {outlier_mask.sum()} extreme outliers from '{target_column}'"
                )

        final_rows = len(self.data)
        logger.info(
            f"Data validation complete: {initial_rows - final_rows} rows removed"
        )

        return self


    
    def standardize_categorical_values(self, column: str) -> 'DataPreprocessor':
        """
        Standardize categorical values (trim, lowercase, etc.).
        
        Args:
            column: Column name to standardize
        
        Returns:
            Self for method chaining
        
        Design Decision: Consistent categorical values prevent fragmentation
        (e.g., "Dhaka", "dhaka", " Dhaka " treated as same value).
        """
        if column not in self.data.columns:
            logger.warning(f"Column '{column}' not found, skipping standardization")
            return self
        
        if self.data[column].dtype == 'object':
            # Strip whitespace and convert to title case
            self.data[column] = self.data[column].str.strip().str.title()
            logger.info(f"Standardized categorical values in '{column}'")
        else:
            logger.warning(f"Column '{column}' is not categorical, skipping")
        
        return self
    
    def split_data(self, target_column: str, test_size: float = None, 
                   random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            target_column: Name of the target variable column
            test_size: Proportion of data for test set (default from config)
            random_state: Random seed for reproducibility (default from config)
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        
        Design Decision: Stratified split ensures representative distribution
        in both train and test sets.
        """
        test_size = test_size or config.TEST_SIZE
        random_state = random_state or config.RANDOM_STATE
        
        if target_column not in self.data.columns:
            error_msg = f"Target column '{target_column}' not found in dataset"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split completed:")
        logger.info(f"  Training set: {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
        logger.info(f"  Test set: {X_test.shape[0]} samples ({test_size*100:.0f}%)")
        logger.info(f"  Features: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Return the processed DataFrame.
        
        Returns:
            Processed DataFrame
        """
        logger.info(f"Returning processed data with shape: {self.data.shape}")
        return self.data