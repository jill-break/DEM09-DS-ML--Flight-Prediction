"""
Exploratory Data Analysis Module

This module performs comprehensive statistical analysis and generates
insights from the dataset.

Design Decision: Separates analysis logic from visualization to allow
independent testing and flexible reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EDAAnalyzer:
    """
    EDAAnalyzer class for statistical analysis and insight generation.
    
    Responsibilities:
    - Compute descriptive statistics
    - Analyze distributions
    - Identify correlations
    - Discover patterns and trends
    - Generate business insights
    
    Design Pattern: Analyzer pattern - separates analysis from visualization
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize EDAAnalyzer with dataset.
        
        Args:
            data: DataFrame to analyze
        """
        self.data = data.copy()
        self.insights = []
        logger.info(f"EDAAnalyzer initialized with {len(data)} rows and {len(data.columns)} columns")
    
    def get_descriptive_statistics(self) -> pd.DataFrame:
        """
        Compute comprehensive descriptive statistics.
        
        Returns:
            DataFrame containing statistical summary
        
        Design Decision: Extended statistics beyond .describe() provide
        deeper understanding of data distribution.
        """
        stats = self.data.describe(include='all').T
        
        # Add additional metrics
        stats['missing'] = self.data.isnull().sum()
        stats['missing_pct'] = (self.data.isnull().sum() / len(self.data) * 100).round(2)
        stats['unique'] = self.data.nunique()
        stats['dtype'] = self.data.dtypes
        
        logger.info("Generated descriptive statistics for all features")
        return stats
    
    def analyze_target_distribution(self, target_column: str) -> Dict:
        """
        Analyze distribution of target variable.
        
        Args:
            target_column: Name of target variable
        
        Returns:
            Dictionary containing distribution metrics
        
        Design Decision: Understanding target distribution is critical for
        model selection and evaluation strategy.
        """
        if target_column not in self.data.columns:
            logger.error(f"Target column '{target_column}' not found")
            raise ValueError(f"Column '{target_column}' not found")
        
        target = self.data[target_column]
        
        distribution = {
            'mean': target.mean(),
            'median': target.median(),
            'std': target.std(),
            'min': target.min(),
            'max': target.max(),
            'q25': target.quantile(0.25),
            'q75': target.quantile(0.75),
            'iqr': target.quantile(0.75) - target.quantile(0.25),
            'skewness': target.skew(),
            'kurtosis': target.kurtosis()
        }
        
        logger.info(f"Target '{target_column}' distribution:")
        logger.info(f"  Mean: {distribution['mean']:.2f}, Median: {distribution['median']:.2f}")
        logger.info(f"  Std: {distribution['std']:.2f}, IQR: {distribution['iqr']:.2f}")
        logger.info(f"  Skewness: {distribution['skewness']:.2f}, Kurtosis: {distribution['kurtosis']:.2f}")
        
        # Generate insights
        if abs(distribution['skewness']) > 1:
            self.insights.append(f"Target variable is highly skewed ({distribution['skewness']:.2f}). Consider transformation.")
        
        if distribution['std'] / distribution['mean'] > 0.5:
            self.insights.append(f"Target variable has high variance (CV={distribution['std']/distribution['mean']:.2f})")
        
        return distribution
    
    def analyze_categorical_feature(self, column: str, top_n: int = 10) -> pd.DataFrame:
        """
        Analyze distribution of categorical feature.
        
        Args:
            column: Name of categorical column
            top_n: Number of top categories to return
        
        Returns:
            DataFrame with category counts and percentages
        
        Design Decision: Top-N analysis prevents overwhelming output for
        high-cardinality features while highlighting dominant categories.
        """
        if column not in self.data.columns:
            logger.error(f"Column '{column}' not found")
            raise ValueError(f"Column '{column}' not found")
        
        value_counts = self.data[column].value_counts()
        
        result = pd.DataFrame({
            'Category': value_counts.index[:top_n],
            'Count': value_counts.values[:top_n],
            'Percentage': (value_counts.values[:top_n] / len(self.data) * 100).round(2)
        })
        
        logger.info(f"Top {top_n} categories for '{column}':")
        for _, row in result.head(5).iterrows():
            logger.info(f"  {row['Category']}: {row['Count']} ({row['Percentage']}%)")
        
        # Generate insights
        if len(value_counts) > 50:
            self.insights.append(f"Feature '{column}' has high cardinality ({len(value_counts)} unique values)")
        
        if value_counts.iloc[0] / len(self.data) > 0.5:
            self.insights.append(f"Feature '{column}' is dominated by one category ({value_counts.index[0]}: {value_counts.iloc[0]/len(self.data)*100:.1f}%)")
        
        return result
    
    def analyze_target_by_category(self, target_column: str, categorical_column: str) -> pd.DataFrame:
        """
        Analyze target variable grouped by categorical feature.
        
        Args:
            target_column: Name of target variable
            categorical_column: Name of categorical feature
        
        Returns:
            DataFrame with aggregated statistics per category
        
        Design Decision: Grouped analysis reveals feature importance and
        helps identify pricing patterns by category.
        """
        if target_column not in self.data.columns or categorical_column not in self.data.columns:
            logger.error(f"Required columns not found")
            raise ValueError("Columns not found")
        
        grouped = self.data.groupby(categorical_column)[target_column].agg([
            'count', 'mean', 'median', 'std', 'min', 'max'
        ]).round(2)
        
        grouped = grouped.sort_values('mean', ascending=False)
        
        logger.info(f"Target '{target_column}' statistics by '{categorical_column}':")
        for idx, row in grouped.head(5).iterrows():
            logger.info(f"  {idx}: mean={row['mean']:.2f}, count={row['count']}")
        
        # Generate insights
        mean_range = grouped['mean'].max() - grouped['mean'].min()
        if mean_range / self.data[target_column].mean() > 0.5:
            self.insights.append(f"Feature '{categorical_column}' shows strong variation in target (range: {mean_range:.2f})")
        
        return grouped
    
    def compute_correlation_matrix(self, method: str = 'pearson') -> pd.DataFrame:
        """
        Compute correlation matrix for numerical features.
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
        
        Returns:
            Correlation matrix DataFrame
        
        Design Decision: Correlation analysis identifies multicollinearity
        and reveals feature relationships.
        """
        numerical_data = self.data.select_dtypes(include=[np.number])
        
        if numerical_data.shape[1] < 2:
            logger.warning("Insufficient numerical features for correlation analysis")
            return pd.DataFrame()
        
        corr_matrix = numerical_data.corr(method=method)
        logger.info(f"Computed {method} correlation matrix for {numerical_data.shape[1]} features")
        
        # Find highly correlated feature pairs
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_matrix.iloc[i, j]
                    ))
        
        if high_corr_pairs:
            logger.info(f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > 0.8):")
            for feat1, feat2, corr in high_corr_pairs[:5]:
                logger.info(f"  {feat1} <-> {feat2}: {corr:.3f}")
            self.insights.append(f"High multicollinearity detected between {len(high_corr_pairs)} feature pairs")
        
        return corr_matrix
    
    def analyze_temporal_trends(self, target_column: str, date_column: str, 
                                groupby: str = 'month') -> pd.DataFrame:
        """
        Analyze temporal trends in target variable.
        
        Args:
            target_column: Name of target variable
            date_column: Name of datetime column
            groupby: Grouping level ('month', 'weekday', 'season')
        
        Returns:
            DataFrame with temporal aggregations
        
        Design Decision: Temporal analysis reveals seasonality and cyclical
        patterns in flight pricing.
        """
        if target_column not in self.data.columns or date_column not in self.data.columns:
            logger.error(f"Required columns not found")
            raise ValueError("Columns not found")
        
        df_temp = self.data.copy()
        
        if not pd.api.types.is_datetime64_any_dtype(df_temp[date_column]):
            df_temp[date_column] = pd.to_datetime(df_temp[date_column], errors='coerce')
        
        if groupby == 'month':
            df_temp['group'] = df_temp[date_column].dt.month
        elif groupby == 'weekday':
            df_temp['group'] = df_temp[date_column].dt.dayofweek
        elif groupby == 'season':
            df_temp['group'] = df_temp[date_column].dt.month.map({
                12: 'Winter', 1: 'Winter', 2: 'Winter',
                3: 'Spring', 4: 'Spring', 5: 'Spring',
                6: 'Summer', 7: 'Summer', 8: 'Summer',
                9: 'Fall', 10: 'Fall', 11: 'Fall'
            })
        
        temporal_agg = df_temp.groupby('group')[target_column].agg([
            'count', 'mean', 'median', 'std'
        ]).round(2)
        
        logger.info(f"Temporal trends by {groupby}:")
        for idx, row in temporal_agg.iterrows():
            logger.info(f"  {idx}: mean={row['mean']:.2f}, count={row['count']}")
        
        return temporal_agg
    
    def get_insights(self) -> List[str]:
        """
        Return all generated insights.
        
        Returns:
            List of insight strings
        """
        logger.info(f"Generated {len(self.insights)} insights from EDA")
        return self.insights