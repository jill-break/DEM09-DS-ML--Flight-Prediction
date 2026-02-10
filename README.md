# Flight Fare Prediction - End-to-End ML Project

A machine learning pipeline for predicting flight fares using historical flight data from Bangladesh airlines.

## Project Overview

This project implements a complete end-to-end machine learning solution for flight fare prediction, following industry best practices and software engineering principles. The system predicts flight prices based on various features such as airline, route, date, and other flight characteristics.

### Business Problem
Airlines and travel platforms need accurate fare predictions to:
- Optimize dynamic pricing strategies
- Provide fare estimates to customers
- Identify pricing patterns and trends
- Support revenue management decisions

### Technical Approach
- **Problem Type**: Supervised Regression
- **Target Variable**: Flight Fare (price)
- **Features**: Airline, route, date, flight characteristics
- **Models**: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest

## Project Architecture

```
flight_fare_prediction/
│
├── data/
│   ├── raw/                          # Original dataset
│   └── processed/                    # Cleaned data
│
├── src/                              # Source code
│   ├── config/                       # Configuration management
│   │   └── config.py
│   ├── data/                         # Data loading & preprocessing
│   │   ├── data_loader.py
│   │   └── preprocessor.py
│   ├── features/                     # Feature engineering
│   │   └── feature_engineer.py
│   ├── eda/                          # Exploratory analysis
│   │   └── analyzer.py
│   ├── models/                       # Model training
│   │   └── trainer.py
│   ├── evaluation/                   # Model evaluation
│   │   └── evaluator.py
│   ├── visualization/                # Plotting utilities
│   │   └── plotter.py
│   └── utils/                        # Helper functions
│       ├── logger.py
│       └── helpers.py
│
├── models/                           # Saved models
├── reports/figures/                  # Visualizations
├── logs/                             # Application logs
├── main.py                           # Pipeline orchestrator
├── requirements.txt                  # Dependencies
└── README.md                         # This file
```


## Getting Started

### Prerequisites

- Python 3.11 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd flight_fare_prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Place your dataset:
   - Download `Flight_Price_Dataset_of_Bangladesh.csv` from Kaggle
   - Place it in `data/raw/` directory

### Running the Pipeline

Execute the complete pipeline:
```bash
python main.py
```

The pipeline will:
1. Load and validate the dataset
2. Clean and preprocess the data
3. Perform exploratory data analysis
4. Engineer features
5. Train multiple models
6. Evaluate and compare models
7. Generate visualizations
8. Save the best model

### Running the Web Application

Launch the interactive web app for live predictions:

```bash
# Activate virtual environment (if using)
.\.venv\Scripts\activate  # Windows
source venv/bin/activate   # Linux/Mac

# Install dependencies (first time only)
pip install -r requirements.txt

# Launch Streamlit app
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`.

**Features:**
- ✈️ Interactive form for flight details
- 🔮 Instant fare predictions
- 📊 Model performance metrics
- ✅ Input validation and error handling
- 📱 Responsive design

## Pipeline Stages

### 1. Data Loading
- Validates data file existence
- Loads CSV into pandas DataFrame
- Logs dataset characteristics
- Generates missing value report

### 2. Data Preprocessing
- Removes duplicate rows
- Handles missing values (median for numerical, mode for categorical)
- Fixes invalid entries (negative fares, outliers)
- Standardizes categorical values
- Validates data types

### 3. Exploratory Data Analysis
- Descriptive statistics for all features
- Target variable distribution analysis
- Categorical feature distribution
- Correlation analysis
- Temporal trend analysis
- Generates actionable insights

### 4. Feature Engineering
- Temporal feature extraction (month, day, weekday, season)
- Categorical encoding (OneHot/Label encoding)
- Numerical feature scaling (StandardScaler)
- Interaction feature creation
- Train-test split (80-20)

### 5. Model Training

**Baseline Models:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor

**Optimization:**
- Cross-validation (5-fold)
- Hyperparameter tuning (GridSearchCV)
- Feature importance extraction

### 6. Model Evaluation

**Metrics:**
- R² Score (coefficient of determination)
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

### 7. Visualization
- Distribution plots (histograms with KDE)
- Boxplots (fare by airline, route)
- Correlation heatmaps
- Actual vs Predicted scatter plots
- Residual analysis plots
- Feature importance charts

## Expected Results

### Model Performance
Based on typical flight fare datasets, expected performance:

| Model | R² Score (Coefficient of Determination) | RMSE (Root Mean Square Error) | MAE ( Mean Absolute Error) | MAPE (Mean Absolute Percentage Error) |
|-------|----------|------|-----|--------------------|
| Linear Regression | -5157.2344 | 290.5866 | 75.756 | 4974.066 |
| Ridge Regression | -32.3598 | 23.3688 | 7.3372 | 443.4184 |
| Lasso Regression | 0.4393 | 3.0296 | 2.1857 | 116.9316 |
| Decision Tree | 0.9909 | 0.3861 | 0.2204 | 5.2959 |
| Random Forest | 0.9931 | 0.3361 | 0.201 | 4.8537 |


### Key Insights Expected
- Airline significantly affects pricing
- Route distance correlates with fare
- Seasonal patterns in pricing
- Weekday vs weekend pricing differences
- Peak travel periods command premium

## Configuration

All configuration parameters are centralized in `src/config/config.py`:

```python
# Data parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature engineering
CATEGORICAL_ENCODING = 'onehot'
NUMERICAL_SCALING = 'standard'

# Model training
CV_FOLDS = 5
SCORING_METRIC = 'neg_mean_squared_error'
```

Modify these parameters to customize the pipeline behavior.

## Output Files

After execution, the following outputs are generated:

1. **Models** (`models/`):
   - `best_model.pkl`: Serialized best performing model
   - `model_comparison.csv`: Comparison table of all models
   - `evaluation_results.json`: Detailed metrics
   - `evaluation_summary.json`: Summary report

2. **Visualizations** (`reports/figures/`):
   - `target_distribution.png`: Fare distribution
   - `correlation_heatmap.png`: Feature correlations
   - `*_predictions.png`: Actual vs Predicted plots
   - `*_residuals.png`: Residual analysis
   - `feature_importance.png`: Feature importance chart

3. **Logs** (`logs/`):
   - `app.log`: Complete execution log with timestamps

## Model Interpretation

### Feature Importance
The pipeline extracts and visualizes feature importance from trained models:
- **Tree-based models**: Gini importance / information gain
- **Linear models**: Coefficient magnitudes

### Business Insights
Top factors influencing flight fares (typical findings):
1. **Airline**: Premium carriers charge 20-40% more
2. **Route**: Long-haul routes command higher fares
3. **Booking timing**: Last-minute bookings are 30-50% costlier
4. **Season**: Peak season fares are 25-35% higher
5. **Day of week**: Weekend flights are 10-15% more expensive

## Future Enhancements

### Potential Improvements
1. **Advanced Models**: XGBoost, LightGBM, Neural Networks
2. **Feature Engineering**: 
   - Flight duration features
   - Days until departure
   - Competitor fare features
3. **Deployment**: 
   - REST API (Flask/FastAPI)
   - Web app (Streamlit)
   - Batch prediction pipeline
4. **MLOps**:
   - Model versioning
   - A/B testing framework
   - Automated retraining (Airflow)
   - Model monitoring
5. **Advanced Analytics**:
   - Fare trend forecasting
   - Route profitability analysis
   - Customer segmentation

## Logging

The project uses Python's `logging` module for comprehensive logging:

- **Console**: Real-time progress monitoring
- **File** (`logs/app.log`): Complete execution history

Log levels used:
- `INFO`: Normal pipeline progress
- `WARNING`: Non-critical issues
- `ERROR`: Critical failures with stack traces

## Contributing

To extend this project:

1. Add new models in `src/models/trainer.py`
2. Add new features in `src/features/feature_engineer.py`
3. Add new visualizations in `src/visualization/plotter.py`
4. Update configuration in `src/config/config.py`
5. Follow existing code style and documentation standards


## Acknowledgments

- Dataset: Kaggle - Flight Price Dataset of Bangladesh
- Libraries: scikit-learn, pandas, matplotlib, seaborn

---
