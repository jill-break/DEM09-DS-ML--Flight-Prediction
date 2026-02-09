"""
Visualization Module

This module handles all plotting and visualization tasks.
It provides reusable plotting functions with consistent styling.

Design Decision: Centralized visualization ensures consistent aesthetics,
reduces code duplication, and facilitates easy style updates.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Optional, Tuple
from src.config import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Plotter:
    """
    Plotter class for creating visualizations.
    
    Responsibilities:
    - Create statistical plots (histograms, boxplots, scatter plots)
    - Generate model evaluation plots (residuals, predictions)
    - Save plots to disk
    - Apply consistent styling
    
    Design Pattern: Factory pattern for creating various plot types
    """
    
    def __init__(self, output_dir: Path = config.FIGURES_DIR, style: str = 'seaborn-v0_8'):
        """
        Initialize Plotter with output directory and style.
        
        Args:
            output_dir: Directory to save plots
            style: Matplotlib style to use
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            plt.style.use(style)
        except:
            logger.warning(f"Style '{style}' not found, using default")
        
        # Set default figure parameters
        plt.rcParams['figure.figsize'] = config.FIGURE_SIZE
        plt.rcParams['figure.dpi'] = config.DPI
        sns.set_palette("husl")
        
        logger.info(f"Plotter initialized with output directory: {output_dir}")
    
    def plot_distribution(self, data: pd.Series, title: str, xlabel: str, 
                          filename: Optional[str] = None, bins: int = 30) -> None:
        """
        Create distribution histogram with KDE overlay.
        
        Args:
            data: Series to plot
            title: Plot title
            xlabel: X-axis label
            filename: Optional filename to save plot
            bins: Number of histogram bins
        
        Design Decision: Histogram with KDE provides both granular and
        smooth distribution views.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(data.dropna(), bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add statistics
        mean_val = data.mean()
        median_val = data.median()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.2f}')
        ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.2f}')
        ax.legend()
        
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Saved distribution plot to {filepath}")
        
        plt.close()
    
    def plot_boxplot(self, data: pd.DataFrame, x_column: str, y_column: str, 
                     title: str, filename: Optional[str] = None, top_n: int = 10) -> None:
        """
        Create boxplot for categorical vs numerical comparison.
        
        Args:
            data: DataFrame containing data
            x_column: Categorical column for x-axis
            y_column: Numerical column for y-axis
            title: Plot title
            filename: Optional filename to save plot
            top_n: Number of top categories to display
        
        Design Decision: Boxplots effectively show distribution differences
        across categories and highlight outliers.
        """
        # Get top N categories by count
        top_categories = data[x_column].value_counts().head(top_n).index
        filtered_data = data[data[x_column].isin(top_categories)]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.boxplot(data=filtered_data, x=x_column, y=y_column, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(x_column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel(y_column.replace('_', ' ').title(), fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Saved boxplot to {filepath}")
        
        plt.close()
    
    def plot_correlation_heatmap(self, corr_matrix: pd.DataFrame, title: str,
                                 filename: Optional[str] = None) -> None:
        """
        Create correlation heatmap.
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            title: Plot title
            filename: Optional filename to save plot
        
        Design Decision: Heatmaps provide intuitive visual representation
        of correlation strength and direction.
        """
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Saved correlation heatmap to {filepath}")
        
        plt.close()
    
    def plot_categorical_distribution(self, data: pd.DataFrame, column: str, 
                                      title: str, filename: Optional[str] = None,
                                      top_n: int = 15) -> None:
        """
        Create bar plot for categorical feature distribution.
        
        Args:
            data: DataFrame containing data
            column: Categorical column to plot
            title: Plot title
            filename: Optional filename to save plot
            top_n: Number of top categories to display
        """
        value_counts = data[column].value_counts().head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(column.replace('_', ' ').title(), fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(value_counts.values):
            ax.text(i, v + max(value_counts) * 0.01, str(v), 
                   ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Saved categorical distribution plot to {filepath}")
        
        plt.close()
    
    def plot_actual_vs_predicted(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 title: str, filename: Optional[str] = None) -> None:
        """
        Create actual vs predicted scatter plot.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            filename: Optional filename to save plot
        
        Design Decision: Scatter plot with perfect prediction line shows
        model accuracy and systematic errors.
        """
        fig, ax = plt.subplots(figsize=(10, 10))
        
        ax.scatter(y_true, y_pred, alpha=0.5, s=30, color='blue', edgecolor='black')
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Saved actual vs predicted plot to {filepath}")
        
        plt.close()
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray, 
                       title: str, filename: Optional[str] = None) -> None:
        """
        Create residual plot.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            title: Plot title
            filename: Optional filename to save plot
        
        Design Decision: Residual plot reveals heteroscedasticity and
        systematic prediction errors.
        """
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.5, s=30, color='purple', edgecolor='black')
        ax1.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax1.set_title('Residuals vs Predicted Values', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Predicted Values', fontsize=12)
        ax1.set_ylabel('Residuals', fontsize=12)
        ax1.grid(alpha=0.3)
        
        # Residual distribution
        ax2.hist(residuals, bins=50, alpha=0.7, color='green', edgecolor='black')
        ax2.set_title('Residual Distribution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Saved residual plot to {filepath}")
        
        plt.close()
    
    def plot_feature_importance(self, feature_names: List[str], importances: np.ndarray,
                               title: str, filename: Optional[str] = None, top_n: int = 15) -> None:
        """
        Create feature importance bar plot.
        
        Args:
            feature_names: List of feature names
            importances: Array of importance values
            title: Plot title
            filename: Optional filename to save plot
            top_n: Number of top features to display
        """
        # Create DataFrame and sort
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.barh(range(len(importance_df)), importance_df['Importance'], color='teal', edgecolor='black')
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['Feature'])
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_ylabel('Feature', fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if filename:
            filepath = self.output_dir / filename
            plt.savefig(filepath, dpi=config.DPI, bbox_inches='tight')
            logger.info(f"Saved feature importance plot to {filepath}")
        
        plt.close()