"""
Main Pipeline Orchestration Script

This script orchestrates the entire end-to-end machine learning pipeline
for flight fare prediction, from data loading to model evaluation.

Design Decision: Single entry point with clear execution flow makes the
pipeline easy to understand, reproduce, and automate.

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from src.config import config
from src.utils.logger import get_logger
from src.utils.helpers import set_random_seed
from src.data import DataLoader, DataPreprocessor
from src.features import FeatureEngineer
from src.eda import EDAAnalyzer
from src.models import ModelTrainer
from src.evaluation import ModelEvaluator
from src.visualization import Plotter


# Initialize logger
logger = get_logger(__name__)


def main():
    """
    Execute the complete ML pipeline.
    
    Pipeline Steps:
    1. Data Loading
    2. Data Preprocessing
    3. Feature Engineering
    4. Exploratory Data Analysis
    5. Model Training (Baseline)
    6. Model Training (Optimized)
    7. Model Evaluation
    8. Visualization
    9. Model Interpretation
    """
    
    logger.info("=" * 80)
    logger.info("FLIGHT FARE PREDICTION - ML PIPELINE STARTED")
    logger.info("=" * 80)
    
    # Set random seed for reproducibility
    set_random_seed(config.RANDOM_STATE)
    
    # =========================================================================
    # STEP 1: DATA LOADING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: DATA LOADING")
    logger.info("=" * 80)
    
    data_loader = DataLoader(config.RAW_DATA_FILE)
    
    # Check if data file exists
    if not config.RAW_DATA_FILE.exists():
        logger.error(f"Data file not found: {config.RAW_DATA_FILE}")
        logger.error("Please ensure 'Flight_Price_Dataset_of_Bangladesh.csv' is in the data/raw/ directory")
        return
    
    raw_data = data_loader.load_data()
    
    # Get initial insights
    missing_report = data_loader.get_missing_value_report()
    
    # =========================================================================
    # STEP 2: DATA PREPROCESSING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: DATA PREPROCESSING")
    logger.info("=" * 80)
    
    # Identify columns (adjust based on actual dataset)
    # Note: This is a template - actual columns will depend on the dataset
    # Common columns in flight datasets: Date, Airline, From, To, Fare, etc.
    
    # For demonstration, let's assume we have these columns
    # You'll need to adjust this based on actual dataset structure
    
    preprocessor = DataPreprocessor(raw_data)
    
    # Chain preprocessing steps
    cleaned_data = (preprocessor
                    .remove_duplicates()
                    .handle_missing_values()
                    .get_processed_data())
    
    logger.info(f"Preprocessing complete. Final shape: {cleaned_data.shape}")
    
    # =========================================================================
    # STEP 3: EXPLORATORY DATA ANALYSIS
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: EXPLORATORY DATA ANALYSIS")
    logger.info("=" * 80)
    
    eda_analyzer = EDAAnalyzer(cleaned_data)
    plotter = Plotter()
    
    # Generate descriptive statistics
    desc_stats = eda_analyzer.get_descriptive_statistics()
    logger.info(f"\nDescriptive Statistics:\n{desc_stats.head(10)}")
    
    # Analyze numerical columns
    numerical_cols = cleaned_data.select_dtypes(include=['number']).columns.tolist()
    
    if len(numerical_cols) > 0:
        # Assume first numerical column is target (or specify explicitly)
        target_col = numerical_cols[0]
        logger.info(f"Using '{target_col}' as target variable for demonstration")
        
        # Analyze target distribution
        target_dist = eda_analyzer.analyze_target_distribution(target_col)
        
        # Plot target distribution
        plotter.plot_distribution(
            cleaned_data[target_col],
            title=f'Distribution of {target_col}',
            xlabel=target_col,
            filename='target_distribution.png'
        )
    
    # Analyze categorical columns
    categorical_cols = cleaned_data.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols[:3]:  # Analyze first 3 categorical columns
        cat_analysis = eda_analyzer.analyze_categorical_feature(col, top_n=10)
        
        # Plot categorical distribution
        plotter.plot_categorical_distribution(
            cleaned_data,
            col,
            title=f'Distribution of {col}',
            filename=f'{col}_distribution.png'
        )
    
    # Correlation analysis
    if len(numerical_cols) > 1:
        corr_matrix = eda_analyzer.compute_correlation_matrix()
        plotter.plot_correlation_heatmap(
            corr_matrix,
            title='Feature Correlation Heatmap',
            filename='correlation_heatmap.png'
        )
    
    # Get EDA insights
    insights = eda_analyzer.get_insights()
    logger.info("\n" + "=" * 80)
    logger.info("KEY INSIGHTS FROM EDA:")
    logger.info("=" * 80)
    for i, insight in enumerate(insights, 1):
        logger.info(f"{i}. {insight}")
    
    # =========================================================================
    # STEP 4: FEATURE ENGINEERING & DATA SPLITTING
    # =========================================================================
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: FEATURE ENGINEERING & DATA SPLITTING")
    logger.info("=" * 80)
    
    # Note: Feature engineering steps will depend on actual dataset
    # This is a template showing the pattern
    
    feature_engineer = FeatureEngineer()
    
    # Prepare data for modeling
    # Explicitly set target column to avoid data leakage
    target_column = 'Total Fare (BDT)'
    
    # --- TEMPORAL FEATURE ENGINEERING (FIX FOR DATE MEMORIZATION) ---
    logger.info("Extracting temporal features...")
    # Extract features from Departure
    cleaned_data = feature_engineer.create_temporal_features(cleaned_data, 'Departure Date & Time')
    # Rename generic temporal columns to be specific to Departure
    cleaned_data = cleaned_data.rename(columns={
        'month': 'Dep_Month', 'day': 'Dep_Day', 'weekday': 'Dep_Weekday', 
        'day_of_year': 'Dep_DayOfYear', 'season': 'Dep_Season', 
        'is_weekend': 'Dep_IsWeekend', 'hour': 'Dep_Hour'
    })
    
    # Extract features from Arrival (Optional but can be useful)
    # We mainly care about duration which is already calculated, but arrival time of day matters
    cleaned_data = feature_engineer.create_temporal_features(cleaned_data, 'Arrival Date & Time')
    cleaned_data = cleaned_data.rename(columns={
        'month': 'Arr_Month', 'day': 'Arr_Day', 'weekday': 'Arr_Weekday', 
        'day_of_year': 'Arr_DayOfYear', 'season': 'Arr_Season', 
        'is_weekend': 'Arr_IsWeekend', 'hour': 'Arr_Hour'
    })
    
    # --- ROUTE FEATURE ENGINEERING ---
    logger.info("Extracting route features...")
    cleaned_data = feature_engineer.create_route_features(cleaned_data)

    # CRITICAL: Drop raw datetime columns and other non-feature columns
    cols_to_drop = ['Departure Date & Time', 'Arrival Date & Time']
    cleaned_data = feature_engineer.drop_features(cleaned_data, cols_to_drop)
    
    # Drop leaky columns: Base Fare and Tax & Surcharge are components of Total Fare
    leaky_columns = ['Base Fare (BDT)', 'Tax & Surcharge (BDT)']
    cleaned_data = cleaned_data.drop(
        columns=[c for c in leaky_columns if c in cleaned_data.columns]
    )
    logger.info(f"Dropped leaky columns: {[c for c in leaky_columns if c in cleaned_data.columns or c in leaky_columns]}")
    
    if target_column in cleaned_data.columns:
        # Split data
        preprocessor_for_split = DataPreprocessor(cleaned_data)
        X_train, X_test, y_train, y_test = preprocessor_for_split.split_data(target_column)
        
        logger.info(f"Target column: {target_column}")
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Encode categorical features
        cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
        if len(cat_cols) > 0:
            logger.info(f"Encoding categorical columns: {cat_cols} using ONE-HOT Encoding")
            # FORCE ONE-HOT ENCODING as per requirements
            X_train = feature_engineer.encode_categorical_features(X_train, cat_cols, method='onehot', fit=True)
            X_test = feature_engineer.encode_categorical_features(X_test, cat_cols, method='onehot', fit=False)
        
        # Scale numerical features
        num_cols = X_train.select_dtypes(include=['number']).columns.tolist()
        if len(num_cols) > 0:
            logger.info(f"Scaling numerical columns: {len(num_cols)} columns")
            X_train = feature_engineer.scale_numerical_features(X_train, num_cols, method='standard', fit=True)
            X_test = feature_engineer.scale_numerical_features(X_test, num_cols, method='standard', fit=False)
            
        # SANITIZE COLUMN NAMES (Required for LightGBM/XGBoost)
        import re
        def sanitize_colnames(df):
            new_cols = []
            for col in df.columns:
                # Replace special chars with underscore, keep alphanumeric
                new_col = re.sub(r'[^a-zA-Z0-9_]', '_', col)
                # Remove repeated underscores
                new_col = re.sub(r'_+', '_', new_col)
                # Remove leading/trailing underscores
                new_col = new_col.strip('_')
                new_cols.append(new_col)
            return new_cols
            
        X_train.columns = sanitize_colnames(X_train)
        X_test.columns = sanitize_colnames(X_test)
        logger.info("Sanitized column names for model compatibility")
        
        feature_names = feature_engineer.get_feature_names(X_train)
        
        # =========================================================================
        # STEP 5: BASELINE MODEL TRAINING
        # =========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 5: BASELINE MODEL TRAINING")
        logger.info("=" * 80)
        
        trainer = ModelTrainer()
        baseline_models = trainer.train_baseline_models(X_train, y_train)
        
        # =========================================================================
        # STEP 6: MODEL EVALUATION
        # =========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 6: MODEL EVALUATION")
        logger.info("=" * 80)
        
        evaluator = ModelEvaluator()
        
        # Evaluate all baseline models
        for name, model in baseline_models.items():
            metrics = trainer.evaluate_model(model, X_test, y_test, name)
            evaluator.add_model_results(name, metrics)
            
            # Plot predictions for best performing models
            if 'Random Forest' in name or 'Linear Regression' in name:
                y_pred = model.predict(X_test)
                
                plotter.plot_actual_vs_predicted(
                    y_test.values,
                    y_pred,
                    title=f'{name} - Actual vs Predicted',
                    filename=f'{name.replace(" ", "_").lower()}_predictions.png'
                )
                
                plotter.plot_residuals(
                    y_test.values,
                    y_pred,
                    title=f'{name} - Residual Analysis',
                    filename=f'{name.replace(" ", "_").lower()}_residuals.png'
                )
        
        # Create comparison table
        comparison_table = evaluator.create_comparison_table()
        
        # Print model ranking
        evaluator.print_model_ranking(metric='r2_score')
        
        # Get best model
        best_model_name, best_r2 = evaluator.get_best_model(metric='r2_score')
        logger.info(f"\nBest Model: {best_model_name} with R² = {best_r2:.4f}")
        
        # =========================================================================
        # STEP 7: HYPERPARAMETER TUNING (OPTIMIZATION)
        # =========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 7: HYPERPARAMETER TUNING")
        logger.info("=" * 80)
        
        # Tune XGBoost (Advanced Gradient Boosting)
        logger.info("Tuning XGBoost hyperparameters...")
        best_xgb, best_xgb_params = trainer.tune_hyperparameters(
            'xgboost',
            X_train,
            y_train,
            config.XGBOOST_PARAMS,
            search_type='grid'
        )
        
        # Evaluate tuned model
        xgb_metrics = trainer.evaluate_model(best_xgb, X_test, y_test, 'XGBoost (Tuned)')
        evaluator.add_model_results('XGBoost (Tuned)', xgb_metrics)
        
        # =========================================================================
        # STEP 8: FEATURE IMPORTANCE ANALYSIS
        # =========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 8: FEATURE IMPORTANCE ANALYSIS")
        logger.info("=" * 80)
        
        # Get best performing model
        best_model = baseline_models.get(best_model_name) or best_xgb
        
        importance_df = trainer.get_feature_importance(best_model, feature_names)
        
        if not importance_df.empty:
            logger.info("\nTop 10 Most Important Features:")
            logger.info(importance_df.head(10).to_string())
            
            # Plot feature importance
            plotter.plot_feature_importance(
                importance_df['Feature'].tolist(),
                importance_df['Importance'].values,
                title='Feature Importance',
                filename='feature_importance.png',
                top_n=15
            )
        
        # =========================================================================
        # STEP 9: SAVE RESULTS
        # =========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("STEP 9: SAVING RESULTS")
        logger.info("=" * 80)
        
        # Save best model
        trainer.save_model(best_model, config.BEST_MODEL_FILE)
        
        # Save evaluation results
        evaluator.save_results()
        
        # =========================================================================
        # STEP 10: FINAL SUMMARY
        # =========================================================================
        logger.info("\n" + "=" * 80)
        logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        summary = evaluator.generate_summary_report()
        logger.info(f"\nTotal Models Evaluated: {summary['total_models_evaluated']}")
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Best R² Score: {best_r2:.4f}")
        
        logger.info("\nOutput Files:")
        logger.info(f"  - Best Model: {config.BEST_MODEL_FILE}")
        logger.info(f"  - Visualizations: {config.FIGURES_DIR}")
        logger.info(f"  - Evaluation Results: {config.MODELS_DIR}")
        logger.info(f"  - Logs: {config.LOG_FILE}")
        
    else:
        logger.error("No numerical columns found in dataset for modeling")
        logger.error("Please check the dataset structure and update the pipeline accordingly")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        sys.exit(1)