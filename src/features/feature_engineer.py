"""
Feature Engineering Module

This module handles creation of derived features from raw data.
It transforms domain-specific information into model-ready features.

Design Decision: Feature engineering is separated from preprocessing to
maintain clear separation of concerns and facilitate feature experimentation.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    FeatureEngineer class for creating and transforming features.
    
    Responsibilities:
    - Extract temporal features from dates
    - Encode categorical variables
    - Scale numerical features
    - Create interaction features
    - Engineer domain-specific features
    
    Design Pattern: Transformer pattern - maintains fitted state for
    consistent train/test transformation.
    """
    
    def __init__(self):
        """Initialize FeatureEngineer with empty transformers."""
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        logger.info("FeatureEngineer initialized")
    
    def create_temporal_features(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """
        Extract temporal features from a datetime column.
        
        Args:
            df: Input DataFrame
            date_column: Name of the datetime column
        
        Returns:
            DataFrame with added temporal features
        
        Design Decision: Temporal features (month, day, weekday, season) capture
        cyclical patterns in flight pricing.
        """
        df = df.copy()
        
        if date_column not in df.columns:
            logger.warning(f"Column '{date_column}' not found, skipping temporal features")
            return df
        
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            logger.warning(f"Column '{date_column}' is not datetime type, attempting conversion")
            df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        
        # Extract features
        df['month'] = df[date_column].dt.month
        df['day'] = df[date_column].dt.day
        df['weekday'] = df[date_column].dt.dayofweek  # 0=Monday, 6=Sunday
        df['day_of_year'] = df[date_column].dt.dayofyear
        df['hour'] = df[date_column].dt.hour
        
        # Season mapping (Northern Hemisphere - adjust if needed)
        df['season'] = df['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Is weekend flag
        df['is_weekend'] = (df['weekday'] >= 5).astype(int)
        
        logger.info(f"Created temporal features from '{date_column}': month, day, weekday, day_of_year, hour, season, is_weekend")
        
        logger.info(f"Created temporal features from '{date_column}': month, day, weekday, day_of_year, hour, season, is_weekend")
        
        return df

    def create_route_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create route-based features including distance and type.
        
        Args:
            df: Input DataFrame with 'Source' and 'Destination' columns
            
        Returns:
            DataFrame with 'route_distance' and 'route_type' features
        """
        df = df.copy()
        
        # Import coordinates here to avoid circular imports if any
        from src.app.utils import AIRPORT_COORDINATES
        from geopy.distance import geodesic
        
        def calculate_distance(row):
            source_code = row.get('Source')
            dest_code = row.get('Destination')
            
            if source_code in AIRPORT_COORDINATES and dest_code in AIRPORT_COORDINATES:
                return geodesic(AIRPORT_COORDINATES[source_code], AIRPORT_COORDINATES[dest_code]).km
            return np.nan

        # Calculate Distance
        if 'Source' in df.columns and 'Destination' in df.columns:
            # Check if geopy is available, otherwise use simple euclidian approximation or lookup table
            # Since geopy might not be installed, let's use a haversine implementation to be safe and dependency-free
            
            def haversine_distance(lat1, lon1, lat2, lon2):
                R = 6371  # Earth radius in km
                phi1, phi2 = np.radians(lat1), np.radians(lat2)
                dphi = np.radians(lat2 - lat1)
                dlambda = np.radians(lon2 - lon1)
                
                a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2) * np.sin(dlambda/2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
                return R * c

            def get_dist(row):
                src = AIRPORT_COORDINATES.get(row['Source'])
                dst = AIRPORT_COORDINATES.get(row['Destination'])
                if src and dst:
                    return haversine_distance(src[0], src[1], dst[0], dst[1])
                return 0 # Default to 0 if unknown

            df['route_distance'] = df.apply(get_dist, axis=1)
            logger.info("Created 'route_distance' feature")
            
            # Route Type (International if Source or Dest is not in Bangladesh)
            # BD Airports: DAC, CGP, ZYL, CXB, JSR, BZL, RJH, SPD
            bd_airports = ['DAC', 'CGP', 'ZYL', 'CXB', 'JSR', 'BZL', 'RJH', 'SPD']
            
            df['route_type'] = df.apply(
                lambda x: 'Domestic' if (x['Source'] in bd_airports and x['Destination'] in bd_airports) else 'International', 
                axis=1
            )
            logger.info("Created 'route_type' feature")
            
        else:
            logger.warning("Source or Destination columns missing, skipping route features")
            
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, columns: List[str], 
                                    method: str = 'onehot', fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using specified method.
        
        Args:
            df: Input DataFrame
            columns: List of categorical columns to encode
            method: Encoding method ('onehot' or 'label')
            fit: If True, fit encoders; if False, use existing encoders
        
        Returns:
            DataFrame with encoded features
        
        Design Decision: OneHot encoding for nominal variables preserves
        independence; Label encoding for ordinal or high-cardinality features.
        """
        df = df.copy()
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping encoding")
                continue
            
            if method == 'onehot':
                if fit:
                    # Create dummy variables
                    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                    self.encoders[col] = list(dummies.columns)
                    df = pd.concat([df, dummies], axis=1)
                else:
                    # Robust inference encoding
                    # Create empty DataFrame with expected columns learned from training
                    dummies = pd.DataFrame(0, index=df.index, columns=self.encoders[col], dtype=int)
                    
                    # Fill 1s where the category matches a column
                    for idx, value in df[col].items():
                        # Construct the column name that would have been created
                        target_col = f"{col}_{value}"
                        
                        if target_col in dummies.columns:
                            dummies.at[idx, target_col] = 1
                            
                    df = pd.concat([df, dummies], axis=1)
                
                df = df.drop(columns=[col])
                logger.info(f"OneHot encoded '{col}' using fitted encoder")
            
            elif method == 'label':
                if fit:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    self.encoders[col] = le
                else:
                    le = self.encoders[col]
                    # Handle unseen categories
                    df[col] = df[col].astype(str).map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )
                
                logger.info(f"Label encoded '{col}'")
        
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame, columns: List[str], 
                                 method: str = 'standard', fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features using specified method.
        
        Args:
            df: Input DataFrame
            columns: List of numerical columns to scale
            method: Scaling method ('standard', 'minmax', 'robust')
            fit: If True, fit scalers; if False, use existing scalers
        
        Returns:
            DataFrame with scaled features
        
        Design Decision: StandardScaler for normally distributed features,
        RobustScaler for features with outliers, MinMaxScaler for bounded ranges.
        """
        df = df.copy()
        
        # Select scaler
        if method == 'standard':
            scaler_class = StandardScaler
        elif method == 'minmax':
            scaler_class = MinMaxScaler
        elif method == 'robust':
            scaler_class = RobustScaler
        else:
            logger.warning(f"Unknown scaling method '{method}', using StandardScaler")
            scaler_class = StandardScaler
        
        for col in columns:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found, skipping scaling")
                continue
            
            if fit:
                scaler = scaler_class()
                df[col] = scaler.fit_transform(df[[col]])
                self.scalers[col] = scaler
            else:
                if col not in self.scalers:
                    logger.warning(f"No fitted scaler for '{col}', skipping")
                    continue
                scaler = self.scalers[col]
                df[col] = scaler.transform(df[[col]])
            
            logger.info(f"Scaled '{col}' using {method} scaling")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: List[tuple]) -> pd.DataFrame:
        """
        Create interaction features from pairs of existing features.
        
        Args:
            df: Input DataFrame
            feature_pairs: List of tuples, each containing two feature names
        
        Returns:
            DataFrame with added interaction features
        
        Design Decision: Interaction features capture non-linear relationships
        (e.g., airline × route effects on pricing).
        """
        df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                interaction_name = f"{feat1}_x_{feat2}"
                df[interaction_name] = df[feat1] * df[feat2]
                logger.info(f"Created interaction feature: {interaction_name}")
            else:
                logger.warning(f"Cannot create interaction between '{feat1}' and '{feat2}': columns not found")
        
        return df
    
    def drop_features(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Drop specified columns from DataFrame.
        
        Args:
            df: Input DataFrame
            columns: List of column names to drop
        
        Returns:
            DataFrame with columns removed
        """
        df = df.copy()
        existing_cols = [col for col in columns if col in df.columns]
        
        if existing_cols:
            df = df.drop(columns=existing_cols)
            logger.info(f"Dropped {len(existing_cols)} features: {existing_cols}")
        else:
            logger.warning(f"None of the specified columns found: {columns}")
        
        return df
    
    def get_feature_names(self, df: pd.DataFrame) -> List[str]:
        """
        Get list of all feature names in DataFrame.
        
        Args:
            df: Input DataFrame
        
        Returns:
            List of column names
        """
        self.feature_names = list(df.columns)
        logger.info(f"Total features: {len(self.feature_names)}")
        return self.feature_names