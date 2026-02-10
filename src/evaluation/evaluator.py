"""
Model Evaluation Module

This module provides comprehensive model evaluation and comparison capabilities.
It aggregates results from multiple models and generates comparative reports.

Design Decision: Separating evaluation from training allows independent
testing and flexible reporting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
from pathlib import Path
from src.config import config
from src.utils.logger import get_logger
from src.utils.helpers import save_json

logger = get_logger(__name__)


class ModelEvaluator:
    """
    ModelEvaluator class for comparing and reporting model performance.
    
    Responsibilities:
    - Aggregate metrics from multiple models
    - Compare model performance
    - Identify best performing model
    - Generate evaluation reports
    - Save results to disk
    
    Design Pattern: Facade pattern - provides simplified interface to
    complex evaluation operations
    """
    
    def __init__(self):
        """Initialize ModelEvaluator."""
        self.results = {}
        self.comparison_df = None
        logger.info("ModelEvaluator initialized")
    
    def add_model_results(self, model_name: str, metrics: Dict[str, float]) -> None:
        """
        Add evaluation results for a model.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metric names and values
        
        Design Decision: Accumulator pattern allows gradual collection of
        results from different models.
        """
        self.results[model_name] = metrics
        logger.info(f"Added results for {model_name}: R²={metrics.get('r2_score', 'N/A')}")
    
    def create_comparison_table(self) -> pd.DataFrame:
        """
        Create a comparison table of all model results.
        
        Returns:
            DataFrame with models as rows and metrics as columns
        
        Design Decision: Tabular format enables easy visual comparison and
        sorting by different metrics.
        """
        if not self.results:
            logger.warning("No results to compare")
            return pd.DataFrame()
        
        self.comparison_df = pd.DataFrame(self.results).T
        
        # Sort by R² score (descending)
        if 'r2_score' in self.comparison_df.columns:
            self.comparison_df = self.comparison_df.sort_values('r2_score', ascending=False)
        
        # Round numeric values
        self.comparison_df = self.comparison_df.round(4)
        
        logger.info("Model comparison table created")
        logger.info(f"\n{self.comparison_df.to_string()}")
        
        return self.comparison_df
    
    def get_best_model(self, metric: str = 'r2_score', maximize: bool = True) -> tuple:
        """
        Identify the best performing model based on a specific metric.
        
        Args:
            metric: Metric to use for comparison
            maximize: If True, higher is better; if False, lower is better
        
        Returns:
            Tuple of (best_model_name, best_metric_value)
        
        Design Decision: Flexible metric selection allows optimization for
        different business objectives (accuracy vs. error minimization).
        """
        if self.comparison_df is None or self.comparison_df.empty:
            logger.error("No comparison data available. Run create_comparison_table() first.")
            return None, None
        
        if metric not in self.comparison_df.columns:
            logger.error(f"Metric '{metric}' not found in results")
            return None, None
        
        if maximize:
            best_idx = self.comparison_df[metric].idxmax()
        else:
            best_idx = self.comparison_df[metric].idxmin()
        
        best_value = self.comparison_df.loc[best_idx, metric]
        
        logger.info(f"Best model by {metric}: {best_idx} ({metric}={best_value:.4f})")
        
        return best_idx, best_value
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation summary report.
        
        Returns:
            Dictionary containing summary statistics and insights
        
        Design Decision: Structured report format facilitates automated
        reporting and documentation generation.
        """
        if self.comparison_df is None or self.comparison_df.empty:
            logger.warning("No data for summary report")
            return {}
        
        summary = {
            'total_models_evaluated': len(self.comparison_df),
            'metrics_compared': list(self.comparison_df.columns),
            'best_model': {},
            'metric_statistics': {}
        }
        
        # Best model for each metric
        for metric in self.comparison_df.columns:
            maximize = metric in ['r2_score']  # Add more maximize metrics as needed
            best_name, best_value = self.get_best_model(metric, maximize)
            summary['best_model'][metric] = {
                'model_name': best_name,
                'value': float(best_value) if best_value is not None else None
            }
        
        # Statistics for each metric
        for metric in self.comparison_df.columns:
            summary['metric_statistics'][metric] = {
                'mean': float(self.comparison_df[metric].mean()),
                'std': float(self.comparison_df[metric].std()),
                'min': float(self.comparison_df[metric].min()),
                'max': float(self.comparison_df[metric].max())
            }
        
        logger.info("Summary report generated")
        
        return summary
    
    def save_results(self, output_dir: Path = config.MODELS_DIR) -> None:
        """
        Save evaluation results to disk.
        
        Args:
            output_dir: Directory to save results
        
        Design Decision: Persisting results enables reproducibility and
        provides audit trail for model selection.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save comparison table
        if self.comparison_df is not None:
            comparison_file = output_dir / "model_comparison.csv"
            self.comparison_df.to_csv(comparison_file)
            logger.info(f"Saved comparison table to {comparison_file}")
        
        # Save detailed results
        results_file = output_dir / "evaluation_results.json"
        save_json(self.results, results_file)
        logger.info(f"Saved detailed results to {results_file}")
        
        # Save summary report
        summary = self.generate_summary_report()
        summary_file = output_dir / "evaluation_summary.json"
        save_json(summary, summary_file)
        logger.info(f"Saved summary report to {summary_file}")
    
    def print_model_ranking(self, metric: str = 'r2_score', top_n: int = None) -> None:
        """
        Print ranked list of models by specified metric.
        
        Args:
            metric: Metric to rank by
            top_n: Number of top models to display (None for all)
        
        Design Decision: Human-readable output facilitates quick assessment
        during interactive analysis.
        """
        if self.comparison_df is None or self.comparison_df.empty:
            logger.warning("No data available for ranking")
            return
        
        if metric not in self.comparison_df.columns:
            logger.error(f"Metric '{metric}' not found")
            return
        
        # Sort by metric
        maximize = metric in ['r2_score']
        ranked = self.comparison_df.sort_values(metric, ascending=not maximize)
        
        if top_n:
            ranked = ranked.head(top_n)
        
        logger.info(f"\n{'=' * 70}")
        logger.info(f"MODEL RANKING BY {metric.upper()}")
        logger.info(f"{'=' * 70}")
        
        for idx, (model_name, row) in enumerate(ranked.iterrows(), 1):
            logger.info(f"{idx}. {model_name:30s} | {metric}: {row[metric]:.4f}")
        
        logger.info(f"{'=' * 70}\n")
    
    def compare_metrics(self, model1: str, model2: str) -> Dict[str, Dict]:
        """
        Compare two specific models across all metrics.
        
        Args:
            model1: Name of first model
            model2: Name of second model
        
        Returns:
            Dictionary showing metric-by-metric comparison
        
        Design Decision: Pairwise comparison enables detailed analysis of
        model trade-offs.
        """
        if model1 not in self.results or model2 not in self.results:
            logger.error("One or both models not found in results")
            return {}
        
        comparison = {}
        for metric in self.results[model1].keys():
            val1 = self.results[model1][metric]
            val2 = self.results[model2][metric]
            difference = round(val1 - val2, 4)
            pct_change = (difference / val2 * 100) if val2 != 0 else 0
            
            comparison[metric] = {
                model1: val1,
                model2: val2,
                'difference': difference,
                'percent_change': pct_change
            }
        
        logger.info(f"\nComparison: {model1} vs {model2}")
        for metric, values in comparison.items():
            logger.info(f"  {metric}: {values[model1]:.4f} vs {values[model2]:.4f} "
                       f"(diff: {values['difference']:.4f}, {values['percent_change']:.2f}%)")
        
        return comparison