"""
Flight Fare Prediction - Model Retraining DAG

This DAG automates weekly model retraining with the following workflow:
1. Validate training data availability and quality
2. Backup current model
3. Train new model
4. Evaluate performance comparison
5. Deploy if performance improves
6. Send email notification

Schedule: Every Sunday at 2:00 AM
"""

from datetime import datetime, timedelta
from pathlib import Path
import os
import shutil
import json
import pickle
import pandas as pd

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator, PythonVirtualenvOperator
from airflow.operators.email import EmailOperator
from airflow.utils.trigger_rule import TriggerRule


# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path("/opt/airflow")
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Flight_Price_Dataset_of_Bangladesh.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
BACKUP_DIR = PROJECT_ROOT / "models" / "backups"
NEW_MODEL_PATH = PROJECT_ROOT / "models" / "new_model.pkl"

# Minimum data requirements
MIN_DATA_SIZE = 1000
PERFORMANCE_IMPROVEMENT_THRESHOLD = 0.02  # 2% improvement required

# Email configuration
NOTIFICATION_EMAIL = os.getenv("AIRFLOW_NOTIFICATION_EMAIL", "admin@flight.com")


# ============================================================================
# DAG Default Arguments
# ============================================================================

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email': [NOTIFICATION_EMAIL],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2024, 1, 1),
}


# ============================================================================
# Task Functions
# ============================================================================

def check_data_availability(**context):
    """
    Validate that training data exists and meets minimum requirements.
    """
    print(f"Checking data availability at: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}")
    
    # Check data size
    df = pd.read_csv(DATA_PATH)
    data_size = len(df)
    
    print(f"Found {data_size} records in training data")
    
    if data_size < MIN_DATA_SIZE:
        raise ValueError(
            f"Insufficient data: {data_size} records (minimum: {MIN_DATA_SIZE})"
        )
    
    # Store data size for later reference
    context['ti'].xcom_push(key='data_size', value=data_size)
    
    print(f"✓ Data validation passed: {data_size} records available")
    return True


def backup_current_model(**context):
    """
    Backup the current model with timestamp before retraining.
    """
    if not MODEL_PATH.exists():
        print("No existing model to backup")
        return False
    
    # Create backup directory
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate backup filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = BACKUP_DIR / f"model_backup_{timestamp}.pkl"
    
    # Copy current model to backup
    shutil.copy2(MODEL_PATH, backup_path)
    print(f"✓ Model backed up to: {backup_path}")
    
    # Store backup path for potential rollback
    context['ti'].xcom_push(key='backup_path', value=str(backup_path))
    
    # Clean old backups (keep last 5)
    backups = sorted(BACKUP_DIR.glob("model_backup_*.pkl"), reverse=True)
    for old_backup in backups[5:]:
        old_backup.unlink()
        print(f"Deleted old backup: {old_backup}")
    
    return True


def evaluate_new_model(**context):
    """
    Compare new model performance against current model.
    Returns task_id for next step based on comparison.
    """
    # Load performance metrics from training
    metrics_path = PROJECT_ROOT / "models" / "evaluation_summary.json"
    
    if not NEW_MODEL_PATH.exists():
        raise FileNotFoundError(f"New model not found at {NEW_MODEL_PATH}")
    
    # Load new model metrics
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            summary = json.load(f)
        # Extract R2 score from summary structure: summary['best_model']['r2_score']['value']
        new_r2 = summary.get('best_model', {}).get('r2_score', {}).get('value', 0)
    else:
        print(f"Metrics file not found at {metrics_path}")
        new_r2 = 0.67  # Fallback value
        
    print(f"New model R² score: {new_r2}")
    
    # Compare with old model if it exists
    if MODEL_PATH.exists():
        # Try to load old metrics
        old_metrics_path = BACKUP_DIR / "current_model_metrics.json"
        if old_metrics_path.exists():
            with open(old_metrics_path, 'r') as f:
                old_summary = json.load(f)
            # Try to get from nested structure first, then fallback to flat
            old_r2 = old_summary.get('best_model', {}).get('r2_score', {}).get('value')
            if old_r2 is None:
                old_r2 = old_summary.get('r2_score', 0)
        else:
            old_r2 = 0.67  # Assume current performance
        
        improvement = new_r2 - old_r2
        print(f"Old model R² score: {old_r2}")
        print(f"Performance change: {improvement:+.4f}")
        
        # Store metrics for notification
        context['ti'].xcom_push(key='old_r2', value=old_r2)
        context['ti'].xcom_push(key='new_r2', value=new_r2)
        context['ti'].xcom_push(key='improvement', value=improvement)
        
        # Decide whether to deploy
        if improvement >= PERFORMANCE_IMPROVEMENT_THRESHOLD:
            print(f"✓ New model is better by {improvement:.4f}, deploying")
            return 'deploy_new_model'
        else:
            print(f"✗ New model improvement ({improvement:.4f}) below threshold ({PERFORMANCE_IMPROVEMENT_THRESHOLD})")
            return 'skip_deployment'
    else:
        # No old model, deploy new one
        print("No existing model, deploying new model")
        context['ti'].xcom_push(key='new_r2', value=new_r2)
        return 'deploy_new_model'


def deploy_new_model(**context):
    """
    Replace current model with newly trained model.
    """
    if not NEW_MODEL_PATH.exists():
        raise FileNotFoundError(f"New model not found at {NEW_MODEL_PATH}")
    
    # Move new model to production path
    shutil.move(str(NEW_MODEL_PATH), str(MODEL_PATH))
    print(f"✓ New model deployed to: {MODEL_PATH}")
    
    # Save current metrics for next comparison
    metrics_path = PROJECT_ROOT / "models" / "evaluation_summary.json"
    if metrics_path.exists():
        backup_metrics_path = BACKUP_DIR / "current_model_metrics.json"
        shutil.copy2(metrics_path, backup_metrics_path)
    
    return True


def skip_deployment(**context):
    """
    Skip deployment and keep current model.
    """
    print("Keeping current model - new model did not meet improvement threshold")
    
    # Remove new model file
    if NEW_MODEL_PATH.exists():
        NEW_MODEL_PATH.unlink()
    
    return False


# ============================================================================
# DAG Definition
# ============================================================================

dag = DAG(
    'model_retraining',
    default_args=default_args,
    description='Weekly automated model retraining for flight fare prediction',
    schedule_interval='0 2 * * 0',  # Every Sunday at 2 AM
    catchup=False,
    tags=['ml', 'retraining', 'flight-fare'],
)

# Task 1: Check data availability
check_data = PythonOperator(
    task_id='check_data_availability',
    python_callable=check_data_availability,
    dag=dag,
)

# Task 2: Backup current model
backup_model = PythonOperator(
    task_id='backup_current_model',
    python_callable=backup_current_model,
    dag=dag,
)

# Task 3: Run model training in isolated virtual environment
def run_training_in_venv():
    """
    Train the model in an isolated virtual environment.
    This function will be executed in a separate virtualenv with ML dependencies.
    """
    import sys
    import subprocess
    from pathlib import Path
    
    project_root = Path("/opt/airflow")
    main_py = project_root / "main.py"
    output_model = project_root / "models" / "new_model.pkl"
    
    # Change to project directory
    import os
    os.chdir(str(project_root))
    
    # Run main.py with arguments
    # Note: This runs in the virtualenv created by PythonVirtualenvOperator
    sys.path.insert(0, str(project_root))
    
    # Import and run main
    import main
    main.main()
    
    # Rename output to new_model.pkl
    best_model = project_root / "models" / "best_model.pkl"
    if best_model.exists():
        import shutil
        shutil.copy2(best_model, output_model)
        print(f"✓ Model trained and saved to: {output_model}")
    else:
        raise FileNotFoundError("Training did not produce best_model.pkl")
    
    return True

from airflow.operators.python import PythonVirtualenvOperator

train_model = PythonVirtualenvOperator(
    task_id='run_model_training',
    python_callable=run_training_in_venv,
    requirements=[
        'pandas==2.1.4',
        'numpy==1.26.3',
        'scikit-learn==1.4.0',
        'matplotlib==3.8.2',
        'seaborn==0.13.1',
        'joblib==1.3.2',
        'imbalanced-learn==0.12.0',
        'statsmodels==0.14.1',
    ],
    system_site_packages=False,  # Isolate from Airflow env
    dag=dag,
)

# Task 4: Evaluate and decide
evaluate_and_decide = BranchPythonOperator(
    task_id='evaluate_new_model',
    python_callable=evaluate_new_model,
    dag=dag,
)

# Task 5a: Deploy new model
deploy_model = PythonOperator(
    task_id='deploy_new_model',
    python_callable=deploy_new_model,
    dag=dag,
)

# Task 5b: Skip deployment
skip_deploy = PythonOperator(
    task_id='skip_deployment',
    python_callable=skip_deployment,
    dag=dag,
)

# Task 6: Send success notification
send_success_notification = EmailOperator(
    task_id='send_success_notification',
    to=NOTIFICATION_EMAIL,
    subject='✓ Model Retraining Successful - {{ ds }}',
    html_content="""
    <h3>Model Retraining Completed Successfully</h3>
    <p><strong>Date:</strong> {{ ds }}</p>
    <p><strong>New Model R² Score:</strong> {{ ti.xcom_pull(task_ids='evaluate_new_model', key='new_r2') }}</p>
    <p><strong>Performance Improvement:</strong> {{ ti.xcom_pull(task_ids='evaluate_new_model', key='improvement') }}</p>
    <p><strong>Status:</strong> New model deployed</p>
    """,
    trigger_rule=TriggerRule.ONE_SUCCESS,
    dag=dag,
)

# Task 7: Send skip notification
send_skip_notification = EmailOperator(
    task_id='send_skip_notification',
    to=NOTIFICATION_EMAIL,
    subject='ℹ Model Retraining - No Deployment - {{ ds }}',
    html_content="""
    <h3>Model Retraining Completed - Current Model Retained</h3>
    <p><strong>Date:</strong> {{ ds }}</p>
    <p><strong>Current Model R² Score:</strong> {{ ti.xcom_pull(task_ids='evaluate_new_model', key='old_r2') }}</p>
    <p><strong>New Model R² Score:</strong> {{ ti.xcom_pull(task_ids='evaluate_new_model', key='new_r2') }}</p>
    <p><strong>Improvement:</strong> {{ ti.xcom_pull(task_ids='evaluate_new_model', key='improvement') }}</p>
    <p><strong>Status:</strong> Improvement below threshold - current model retained</p>
    """,
    trigger_rule=TriggerRule.ONE_SUCCESS,
    dag=dag,
)


# ============================================================================
# Task Dependencies
# ============================================================================

check_data >> backup_model >> train_model >> evaluate_and_decide
evaluate_and_decide >> deploy_model >> send_success_notification
evaluate_and_decide >> skip_deploy >> send_skip_notification
