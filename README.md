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
- **Features**: Airline, route, route distance, days before departure, temporal features, flight characteristics
- **Models**: Linear Regression, Ridge, Lasso, Decision Tree, Random Forest, XGBoost, LightGBM

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
├── notebooks/                        # Jupyter notebooks
│   └── pipeline_orchestration.ipynb  # Interactive pipeline
├── airflow/                          # Airflow orchestration
│   ├── dags/                         # DAG definitions
│   └── config/                       # Airflow config
├── main.py                           # Pipeline orchestrator
├── app.py                            # Streamlit web app
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
git clone https://github.com/jill-break/DEM09-DS-ML--Flight-Prediction.git
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

**Airflow Features:**
- **Automated Workflow:** Validates data, backs up models, trains new models, and evaluates performance.
- **Smart Deployment**: Only deploys new models if R² score improves by ≥2%.
- **Gradient Boosting Support**: Includes dependencies for XGBoost and LightGBM training.
- **Isolated Training**: Uses `PythonVirtualenvOperator` to run training in a separate environment, avoiding dependency conflicts.
- **Email Alerts:** Sends success/failure notifications.

### Interactive Notebook

For experimentation and step-by-step execution:

1. Open `notebooks/pipeline_orchestration.ipynb` in Jupyter/VS Code.
2. Run cells to specific pipeline stages (EDA, Training, Tuning) interactively.
3. View embedded visualizations and detailed analysis.

The app will automatically open in your browser at `http://localhost:8501`.

**Features:**
- Interactive form for flight details
- Instant fare predictions
- Model performance metrics
- Input validation and error handling
- Responsive design

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
- **New**: Route Distance calculation (using geodesic coordinates)
- **New**: Route Type categorization (Domestic vs International)
- **Precise Encoding**: OneHot Encoding for categorical variables (Airline, Source, Destination)
- **Scaling**: StandardScaler for numerical features
- **Leakage Prevention**: Explicit removal of `Base Fare` and `Tax` columns
- Train-test split (80-20)

### 5. Model Training

**Baseline Models:**
- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor
- **XGBoost Regressor**
- **LightGBM Regressor**

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
| Linear Regression | -16.5006 | 341562.2341 | 127519.9364 | 808.137 |
| Ridge Regression | 0.273 | 69616.3342 | 51314.2689 | 232.4416 |
| Lasso Regression | 0.3362 | 66523.0199 | 48510.2185 | 198.8586 |
| Decision Tree | 0.5297 | 55995.0087 | 32086.7119 | 48.4239 |
| Random Forest | 0.6574 | 47790.3455 | 28470.7284 | 46.052 |
| **XGBoost (Tuned)** | **0.6803** | **46167.32** | 27989.45 | 48.49 |
| **LightGBM** | 0.6779 | 46337.12 | **27891.23** | 47.24 |


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
   - `best_model.joblib`: Serialized best performing model
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
- Interactive form for flight details
- Instant fare predictions
- Model performance metrics
- Input validation and error handling
- Responsive design

### Automated Retraining (Airflow)

The project includes an Apache Airflow DAG for weekly model retraining.

**Prerequisites:** Docker and Docker Compose.

1. Start Airflow services:
```bash
docker-compose up -d
```

2. Access the Airflow UI at `http://localhost:8080` (credentials: `admin`/`admin`).

3. Enable the `model_retraining` DAG. It runs every Sunday at 2:00 AM.

## Acknowledgments

- Dataset: Kaggle - Flight Price Dataset of Bangladesh
- Libraries: scikit-learn, pandas, matplotlib, seaborn

---
