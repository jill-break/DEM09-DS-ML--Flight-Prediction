"""
Model Service Module

This module handles loading the trained model and making predictions.
It serves as the bridge between the web app and the ML model.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import streamlit as st

from src.utils.logger import get_logger
from src.app.utils import (
    calculate_duration, calculate_days_before_departure,
    determine_seasonality, create_input_dataframe
)

logger = get_logger(__name__)


class ModelService:
    """
    Service class for model operations.
    
    Handles model loading, input preprocessing, and predictions.
    """
    
    def __init__(self, model_path: str = "models/best_model.pkl"):
        """
        Initialize ModelService.
        
        Args:
            model_path: Path to the trained model file
        """
        self.model_path = Path(model_path)
        self.model = None
        self.is_loaded = False
        
    @st.cache_resource
    def load_model(_self):
        """
        Load the trained model from disk.
        
        Uses Streamlit caching to avoid reloading on every interaction.
        
        Returns:
            Loaded model object
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            Exception: If model loading fails
        """
        try:
            if not _self.model_path.exists():
                raise FileNotFoundError(f"Model file not found: {_self.model_path}")
            
            with open(_self.model_path, 'rb') as f:
                model = pickle.load(f)
            
            _self.model = model
            _self.is_loaded = True
            logger.info(f"Model loaded successfully from {_self.model_path}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def preprocess_input(self, user_input: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess user input data to match model's expected format.
        
        CRITICAL: The model was trained WITHOUT 'Base Fare (BDT)' and 'Tax & Surcharge (BDT)'
        to avoid data leakage (they are components of the target 'Total Fare (BDT)').
        
        The model expects:
        - Datetime columns as strings or datetime objects (NOT converted to numbers)
        - All columns encoded/scaled the same way as training
        - NO Base Fare or Tax & Surcharge features
        
        Args:
            user_input: Dictionary containing user inputs from the form
            
        Returns:
            Preprocessed DataFrame ready for prediction
            
        Raises:
            ValueError: If required fields are missing or invalid
        """
        try:
            # Calculate derived values
            duration_hrs = calculate_duration(
                user_input['departure_datetime'],
                user_input['arrival_datetime']
            )
            
            days_before = calculate_days_before_departure(
                user_input['departure_datetime']
            )
            
            seasonality = determine_seasonality(user_input['departure_datetime'])
            
            # Create feature dictionary matching EXACT CSV columns
            # NOTE: Base Fare and Tax & Surcharge are NOT included - they were dropped
            # during training to avoid data leakage (they are components of Total Fare)
            features = {
                'Airline': user_input['airline'],
                'Source': user_input['source'],
                'Source Name': user_input.get('source_name', ''),
                'Destination': user_input['destination'],
                'Destination Name': user_input.get('destination_name', ''),
                'Departure Date & Time': user_input['departure_datetime'],  # Keep as datetime!
                'Arrival Date & Time': user_input['arrival_datetime'],      # Keep as datetime!
                'Duration (hrs)': duration_hrs,
                'Stopovers': user_input['stopovers'],
                'Aircraft Type': user_input['aircraft_type'],
                'Class': user_input['travel_class'],
                'Booking Source': user_input['booking_source'],
                # 'Base Fare (BDT)' - REMOVED to avoid data leakage
                # 'Tax & Surcharge (BDT)' - REMOVED to avoid data leakage
                # 'Total Fare (BDT)' - this is the TARGET, should NOT be included
                'Seasonality': seasonality,
                'Days Before Departure': days_before
            }
            
            # Convert to DataFrame - model expects this EXACT structure
            df = pd.DataFrame([features])
            
            # Apply SAME preprocessing as training
            df_processed = self._apply_training_preprocessing(df)
            
            logger.info("Input data preprocessed successfully")
            
            return df_processed
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {str(e)}")
            raise ValueError(f"Failed to preprocess input data: {str(e)}")
    
    def _apply_training_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the EXACT same preprocessing that was done during training.
        
        Based on main.py lines 176-208:
        1. Base Fare and Tax & Surcharge were dropped during training (data leakage prevention)
        2. Keep datetime columns as-is
        3. Encode categorical columns with label encoding
        4. Scale numerical columns with standard scaler
        
        Args:
            df: DataFrame with raw features matching CSV structure
            
        Returns:
            DataFrame with preprocessed features matching training
        """
        from src.features.feature_engineer import FeatureEngineer
        
        df_processed = df.copy()
        
        # Load a sample of training data to fit the encoders/scalers
        # This ensures consistent encoding
        try:
            training_sample = pd.read_csv('data/raw/Flight_Price_Dataset_of_Bangladesh.csv', nrows=1000)
            
            # Remove target column AND leaky columns (same as training)
            leaky_columns = ['Base Fare (BDT)', 'Tax & Surcharge (BDT)', 'Total Fare (BDT)']
            cols_to_drop = [c for c in leaky_columns if c in training_sample.columns]
            if cols_to_drop:
                training_sample = training_sample.drop(columns=cols_to_drop)
            
        except Exception as e:
            logger.warning(f"Could not load training data sample: {str(e)}")
            training_sample = None
        
        # Initialize feature engineer 
        engineer = FeatureEngineer()
        
        # CRITICAL: Convert datetime columns to strings FIRST
        # The model was trained with datetime columns converted to strings then label-encoded
        datetime_cols = df_processed.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
        for col in datetime_cols:
            df_processed[col] = df_processed[col].astype(str)
        
        # Now encode categorical features (including the stringified datetime columns)
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        
        if len(categorical_cols) > 0 and training_sample is not None:
            # Also convert datetime columns in training sample to strings
            training_datetime_cols = training_sample.select_dtypes(include=['datetime', 'datetime64']).columns.tolist()
            for col in training_datetime_cols:
                training_sample[col] = training_sample[col].astype(str)
            
            # Fit on training sample
            training_cat_cols = [col for col in categorical_cols if col in training_sample.columns]
            if len(training_cat_cols) > 0:
                _ = engineer.encode_categorical_features(
                    training_sample,
                    training_cat_cols, 
                    method='label', 
                    fit=True
                )
                # Transform our input
                df_processed = engineer.encode_categorical_features(
                    df_processed, 
                    categorical_cols, 
                    method='label', 
                    fit=False
                )
        
        # Scale numerical features  
        numerical_cols = df_processed.select_dtypes(include=['number']).columns.tolist()
        
        if len(numerical_cols) > 0 and training_sample is not None:
            # Fit on training sample
            training_numerical = training_sample.select_dtypes(include=['number']).columns.tolist()
            common_num_cols = [col for col in numerical_cols if col in training_numerical]
            
            if len(common_num_cols) > 0:
                _ = engineer.scale_numerical_features(
                    training_sample,
                    common_num_cols, 
                    method='standard', 
                    fit=True
                )
                # Transform our input
                df_processed = engineer.scale_numerical_features(
                    df_processed, 
                    numerical_cols, 
                    method='standard', 
                    fit=False
                )
        
        return df_processed
    
    def predict(self, user_input: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Make a fare prediction based on user input.
        
        Args:
            user_input: Dictionary containing user inputs
            
        Returns:
            Tuple of (predicted_fare, metadata_dict)
            
        Raises:
            RuntimeError: If model is not loaded
            Exception: If prediction fails
        """
        try:
            # Ensure model is loaded
            if not self.is_loaded:
                self.load_model()
            
            # Preprocess input
            processed_data = self.preprocess_input(user_input)
            
            # Make prediction
            prediction = self.model.predict(processed_data)[0]
            
            # Prepare metadata
            metadata = {
                'model_type': type(self.model).__name__,
                'features_used': len(processed_data.columns),
                'confidence': 'High',  # Could add actual confidence intervals
            }
            
            logger.info(f"Prediction made successfully: {prediction:.2f}")
            
            return float(prediction), metadata
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded:
            return {'status': 'Not loaded'}
        
        return {
            'status': 'Loaded',
            'model_type': type(self.model).__name__,
            'model_path': str(self.model_path),
        }
