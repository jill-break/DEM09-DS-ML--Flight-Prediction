"""
Model Service Module

This module handles loading the trained model and making predictions.
It serves as the bridge between the web app and the ML model.
"""

import joblib
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
    
    def __init__(self, model_path: str = "models/best_model.joblib"):
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
            
            model = joblib.load(_self.model_path)
            
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
            
            # Determine seasonality
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
                'Departure Date & Time': user_input['departure_datetime'],  # Will be decomposed
                'Arrival Date & Time': user_input['arrival_datetime'],      # Will be decomposed
                'Duration (hrs)': duration_hrs,
                'Stopovers': user_input['stopovers'],
                'Aircraft Type': user_input['aircraft_type'],
                'Class': user_input['travel_class'],
                'Booking Source': user_input['booking_source'],
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
        
        # Load the fitted FeatureEngineer saved during training
        # This has the EXACT encoder columns and scaler parameters from training
        fitted_engineer = None
        try:
            import joblib
            fitted_engineer = joblib.load('models/feature_engineer.joblib')
            logger.info("Loaded fitted FeatureEngineer from models/feature_engineer.joblib")
        except Exception as e:
            logger.warning(f"Could not load fitted FeatureEngineer: {str(e)}")
            logger.warning("Falling back to fresh FeatureEngineer (predictions may be inaccurate)")
        
        # Use fitted engineer if available, otherwise create fresh one
        engineer = fitted_engineer if fitted_engineer is not None else FeatureEngineer()
        
        # --- FEATURE ENGINEERING MIRRORING MAIN.PY ---

        # 1. TEMPORAL FEATURES (Departure)
        df_processed = engineer.create_temporal_features(df_processed, 'Departure Date & Time')
        df_processed = df_processed.rename(columns={
            'month': 'Dep_Month', 'day': 'Dep_Day', 'weekday': 'Dep_Weekday',
            'day_of_year': 'Dep_DayOfYear', 'season': 'Dep_Season',
            'is_weekend': 'Dep_IsWeekend', 'hour': 'Dep_Hour'
        })

        # 2. TEMPORAL FEATURES (Arrival)
        df_processed = engineer.create_temporal_features(df_processed, 'Arrival Date & Time')
        df_processed = df_processed.rename(columns={
            'month': 'Arr_Month', 'day': 'Arr_Day', 'weekday': 'Arr_Weekday',
            'day_of_year': 'Arr_DayOfYear', 'season': 'Arr_Season',
            'is_weekend': 'Arr_IsWeekend', 'hour': 'Arr_Hour'
        })

        # 3. ROUTE FEATURES
        df_processed = engineer.create_route_features(df_processed)
        
        # 4. ORDINAL & BINARY ENCODING
        df_processed = engineer.encode_ordinal_class(df_processed, 'Class')
        df_processed = engineer.encode_binary_stopovers(df_processed, 'Stopovers')
        
        # 5. DEPARTURE HOUR BINNING
        df_processed = engineer.bin_departure_hour(df_processed, 'Dep_Hour')
        
        # 6. DROP RAW DATETIME COLUMNS
        cols_to_drop = ['Departure Date & Time', 'Arrival Date & Time']
        df_processed = engineer.drop_features(df_processed, cols_to_drop)
        
        # 7. ENCODE CATEGORICAL FEATURES (using fitted encoder from training)
        categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
        if len(categorical_cols) > 0 and fitted_engineer is not None:
            # Use fit=False — the encoder was already fitted during training
            df_processed = engineer.encode_categorical_features(
                df_processed,
                categorical_cols,
                method='onehot',
                fit=False
            )
        
        # 8. SCALE NUMERICAL FEATURES (using fitted scaler from training)
        # Only scale columns the scaler was trained on (excludes OneHot bool columns)
        if fitted_engineer is not None and hasattr(engineer, 'scalers') and engineer.scalers:
            # Only scale columns that have a fitted scaler
            scalable_cols = [c for c in df_processed.columns if c in engineer.scalers]
            if len(scalable_cols) > 0:
                df_processed = engineer.scale_numerical_features(
                    df_processed,
                    scalable_cols,
                    method='standard',
                    fit=False
                )

        # SANITIZE COLUMN NAMES (Required for XGBoost)
        import re
        def sanitize_colnames(df):
            new_cols = []
            for col in df.columns:
                new_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
                new_col = re.sub(r'_+', '_', new_col)
                new_col = new_col.strip('_')
                new_cols.append(new_col)
            return new_cols
            
        df_processed.columns = sanitize_colnames(df_processed)
        
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
