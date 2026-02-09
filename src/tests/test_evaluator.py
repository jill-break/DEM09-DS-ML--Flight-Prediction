"""
Unit Tests for ModelEvaluator Class

Tests model evaluation and comparison functionality.

Test Coverage:
- Result collection
- Comparison table generation
- Best model identification
- Summary report generation
"""

import pytest
import pandas as pd
from src.evaluation.evaluator import ModelEvaluator


class TestModelEvaluator:
    """Test suite for ModelEvaluator class."""
    
    @pytest.mark.unit
    def test_initialization(self):
        """Test ModelEvaluator initialization."""
        evaluator = ModelEvaluator()
        
        assert evaluator.results == {}
        assert evaluator.comparison_df is None
    
    @pytest.mark.unit
    def test_add_model_results(self, sample_model_results):
        """Test adding model results."""
        evaluator = ModelEvaluator()
        
        for model_name, metrics in sample_model_results.items():
            evaluator.add_model_results(model_name, metrics)
        
        assert len(evaluator.results) == len(sample_model_results)
        assert 'Linear Regression' in evaluator.results
    
    @pytest.mark.unit
    def test_create_comparison_table(self, sample_model_results):
        """Test comparison table creation."""
        evaluator = ModelEvaluator()
        
        for model_name, metrics in sample_model_results.items():
            evaluator.add_model_results(model_name, metrics)
        
        comparison = evaluator.create_comparison_table()
        
        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == len(sample_model_results)
        assert 'r2_score' in comparison.columns
    
    @pytest.mark.unit
    def test_create_comparison_table_sorted(self, sample_model_results):
        """Test that comparison table is sorted by R2."""
        evaluator = ModelEvaluator()
        
        for model_name, metrics in sample_model_results.items():
            evaluator.add_model_results(model_name, metrics)
        
        comparison = evaluator.create_comparison_table()
        
        # Should be sorted by r2_score descending
        r2_values = comparison['r2_score'].values
        assert all(r2_values[i] >= r2_values[i+1] for i in range(len(r2_values)-1))
    
    @pytest.mark.unit
    def test_create_comparison_table_empty(self):
        """Test comparison table with no results."""
        evaluator = ModelEvaluator()
        comparison = evaluator.create_comparison_table()
        
        assert isinstance(comparison, pd.DataFrame)
        assert comparison.empty
    
    @pytest.mark.unit
    def test_get_best_model(self, sample_model_results):
        """Test best model identification."""
        evaluator = ModelEvaluator()
        
        for model_name, metrics in sample_model_results.items():
            evaluator.add_model_results(model_name, metrics)
        
        evaluator.create_comparison_table()
        best_name, best_value = evaluator.get_best_model(metric='r2_score', maximize=True)
        
        assert best_name == 'Random Forest'  # Has highest R2
        assert best_value == 0.85
    
    @pytest.mark.unit
    def test_get_best_model_minimize(self, sample_model_results):
        """Test best model identification with minimize."""
        evaluator = ModelEvaluator()
        
        for model_name, metrics in sample_model_results.items():
            evaluator.add_model_results(model_name, metrics)
        
        evaluator.create_comparison_table()
        best_name, best_value = evaluator.get_best_model(metric='rmse', maximize=False)
        
        assert best_name == 'Random Forest'  # Has lowest RMSE
        assert best_value == 1200.0
    
    @pytest.mark.unit
    def test_get_best_model_no_data(self):
        """Test best model with no data."""
        evaluator = ModelEvaluator()
        best_name, best_value = evaluator.get_best_model()
        
        assert best_name is None
        assert best_value is None
    
    @pytest.mark.unit
    def test_get_best_model_invalid_metric(self, sample_model_results):
        """Test best model with invalid metric."""
        evaluator = ModelEvaluator()
        
        for model_name, metrics in sample_model_results.items():
            evaluator.add_model_results(model_name, metrics)
        
        evaluator.create_comparison_table()
        best_name, best_value = evaluator.get_best_model(metric='nonexistent')
        
        assert best_name is None
        assert best_value is None
    
    @pytest.mark.unit
    def test_generate_summary_report(self, sample_model_results):
        """Test summary report generation."""
        evaluator = ModelEvaluator()
        
        for model_name, metrics in sample_model_results.items():
            evaluator.add_model_results(model_name, metrics)
        
        evaluator.create_comparison_table()
        summary = evaluator.generate_summary_report()
        
        assert 'total_models_evaluated' in summary
        assert summary['total_models_evaluated'] == 3
        assert 'best_model' in summary
        assert 'metric_statistics' in summary
    
    @pytest.mark.unit
    def test_generate_summary_report_best_models(self, sample_model_results):
        """Test that summary report identifies best models correctly."""
        evaluator = ModelEvaluator()
        
        for model_name, metrics in sample_model_results.items():
            evaluator.add_model_results(model_name, metrics)
        
        evaluator.create_comparison_table()
        summary = evaluator.generate_summary_report()
        
        assert summary['best_model']['r2_score']['model_name'] == 'Random Forest'
        assert summary['best_model']['rmse']['model_name'] == 'Random Forest'
    
    @pytest.mark.unit
    def test_generate_summary_report_statistics(self, sample_model_results):
        """Test that summary report includes statistics."""
        evaluator = ModelEvaluator()
        
        for model_name, metrics in sample_model_results.items():
            evaluator.add_model_results(model_name, metrics)
        
        evaluator.create_comparison_table()
        summary = evaluator.generate_summary_report()
        
        assert 'r2_score' in summary['metric_statistics']
        assert 'mean' in summary['metric_statistics']['r2_score']
        assert 'std' in summary['metric_statistics']['r2_score']
    
    @pytest.mark.unit
    def test_compare_metrics(self, sample_model_results):
        """Test pairwise model comparison."""
        evaluator = ModelEvaluator()
        
        for model_name, metrics in sample_model_results.items():
            evaluator.add_model_results(model_name, metrics)
        
        comparison = evaluator.compare_metrics('Linear Regression', 'Random Forest')
        
        assert 'r2_score' in comparison
        assert 'Linear Regression' in comparison['r2_score']
        assert 'Random Forest' in comparison['r2_score']
        assert 'difference' in comparison['r2_score']
        assert 'percent_change' in comparison['r2_score']
    
    @pytest.mark.unit
    def test_compare_metrics_invalid_models(self, sample_model_results):
        """Test comparison with invalid model names."""
        evaluator = ModelEvaluator()
        
        for model_name, metrics in sample_model_results.items():
            evaluator.add_model_results(model_name, metrics)
        
        comparison = evaluator.compare_metrics('Linear Regression', 'NonExistent')
        
        assert comparison == {}
    
    @pytest.mark.unit
    def test_compare_metrics_calculation(self):
        """Test that metric comparison calculations are correct."""
        evaluator = ModelEvaluator()
        
        evaluator.add_model_results('Model1', {'r2_score': 0.8, 'rmse': 100})
        evaluator.add_model_results('Model2', {'r2_score': 0.9, 'rmse': 80})
        
        comparison = evaluator.compare_metrics('Model1', 'Model2')
        
        # Model1 - Model2 = 0.8 - 0.9 = -0.1
        assert comparison['r2_score']['difference'] == -0.1
        # (0.8 - 0.9) / 0.9 * 100 ≈ -11.11%
        assert abs(comparison['r2_score']['percent_change'] - (-11.111111)) < 0.001


class TestModelEvaluatorIntegration:
    """Integration tests for complete evaluation workflows."""
    
    @pytest.mark.integration
    def test_complete_evaluation_workflow(self, sample_model_results):
        """Test complete evaluation workflow."""
        evaluator = ModelEvaluator()
        
        # Add results
        for model_name, metrics in sample_model_results.items():
            evaluator.add_model_results(model_name, metrics)
        
        # Create comparison
        comparison = evaluator.create_comparison_table()
        
        # Get best model
        best_name, best_value = evaluator.get_best_model()
        
        # Generate summary
        summary = evaluator.generate_summary_report()
        
        assert not comparison.empty
        assert best_name is not None
        assert summary['total_models_evaluated'] > 0


class TestModelEvaluatorEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_single_model_evaluation(self):
        """Test evaluation with single model."""
        evaluator = ModelEvaluator()
        evaluator.add_model_results('OnlyModel', {'r2_score': 0.75, 'rmse': 100})
        
        comparison = evaluator.create_comparison_table()
        
        assert len(comparison) == 1
        assert comparison.index[0] == 'OnlyModel'
    
    @pytest.mark.unit
    def test_identical_model_performance(self):
        """Test with models having identical performance."""
        evaluator = ModelEvaluator()
        
        evaluator.add_model_results('Model1', {'r2_score': 0.8, 'rmse': 100})
        evaluator.add_model_results('Model2', {'r2_score': 0.8, 'rmse': 100})
        
        evaluator.create_comparison_table()
        best_name, best_value = evaluator.get_best_model()
        
        # Should pick first one (or either)
        assert best_name in ['Model1', 'Model2']
        assert best_value == 0.8