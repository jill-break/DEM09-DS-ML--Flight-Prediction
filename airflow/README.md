# Airflow Setup for Automated Model Retraining

## Overview

This directory contains the Airflow DAG for automated weekly retraining of the flight fare prediction model.

## Directory Structure

```
airflow/
├── dags/
│   └── model_retraining_dag.py    # Main retraining DAG
├── logs/                          # Airflow execution logs (auto-generated)
├── plugins/                       # Custom Airflow plugins (if needed)
└── config/
    └── retraining_config.yaml     # Retraining parameters
```

## Quick Start

### 1. Start Airflow Services

```bash
# From project root
docker-compose up -d
```

This will start:
- **PostgreSQL** (Airflow metadata database)
- **Airflow Webserver** (UI at http://localhost:8080)
- **Airflow Scheduler** (DAG execution engine)
- **MySQL** (staging database)
- **PostgreSQL Analytics** (analytics database)

### 2. Access Airflow UI

- **URL:** http://localhost:8080
- **Username:** `admin`
- **Password:** `admin`

### 3. Enable the DAG

1. Navigate to the DAGs page
2. Find `model_retraining` DAG
3. Toggle the switch to enable it
4. The DAG will run automatically every Sunday at 2:00 AM

### 4. Manual Trigger (Optional)

To trigger retraining manually:
1. Click on the DAG name
2. Click the "Play" button (Trigger DAG)
3. Confirm the trigger

## DAG Workflow

The `model_retraining` DAG follows this workflow:

```
check_data_availability
         ↓
backup_current_model
         ↓
run_model_training
         ↓
evaluate_new_model
     ↙        ↘
deploy        skip_deployment
new_model
     ↓              ↓
send_success    send_skip
notification    notification
```

### Task Descriptions

1. **check_data_availability** - Validates training data exists and has sufficient records (min: 1000)
2. **backup_current_model** - Creates timestamped backup of current model
3. **run_model_training** - Executes `main.py` to train new model
4. **evaluate_new_model** - Compares new vs current model performance
5. **deploy_new_model** - Deploys new model if improvement ≥ 2%
6. **skip_deployment** - Keeps current model if improvement < 2%
7. **send notifications** - Emails status report

## Configuration

### Retraining Parameters

Edit `airflow/config/retraining_config.yaml`:

```yaml
model:
  min_data_size: 1000                          # Minimum training records
  performance_improvement_threshold: 0.02      # 2% improvement required
  backup_retention_days: 30                    # Keep last 30 days of backups

schedule:
  retraining: "0 2 * * 0"                     # Cron: Sunday 2 AM
```

### Email Notifications

To enable email notifications, add these environment variables to `docker-compose.yml`:

```yaml
environment:
  AIRFLOW__SMTP__SMTP_HOST: smtp.gmail.com
  AIRFLOW__SMTP__SMTP_USER: your-email@gmail.com
  AIRFLOW__SMTP__SMTP_PASSWORD: your-app-password
  AIRFLOW__SMTP__SMTP_PORT: 587
  AIRFLOW__SMTP__SMTP_MAIL_FROM: your-email@gmail.com
  AIRFLOW_NOTIFICATION_EMAIL: admin@flight.com
```

**Note:** For Gmail, you need to create an [App Password](https://support.google.com/accounts/answer/185833).

### Modify Schedule

Change the cron expression in `airflow/dags/model_retraining_dag.py`:

```python
dag = DAG(
    'model_retraining',
    schedule_interval='0 2 * * 0',  # Modify this line
    ...
)
```

**Examples:**
- `0 2 * * 0` - Every Sunday at 2 AM
- `0 0 * * *` - Daily at midnight
- `0 0 1 * *` - Monthly on the 1st at midnight
- `0 */6 * * *` - Every 6 hours

## Monitoring

### View DAG Execution

1. Go to http://localhost:8080
2. Click on the `model_retraining` DAG
3. View recent run history and logs

### Check Logs

**Docker Logs:**
```bash
docker-compose logs airflow-scheduler
docker-compose logs airflow-server
```

**Task Logs:**
- UI: DAGs → model_retraining → Graph View → Click task → View logs
- Files: `airflow/logs/model_retraining/`

### Model Backups

Backups are stored in: `models/backups/`

Format: `model_backup_YYYYMMDD_HHMMSS.joblib`

The system automatically keeps the last 5 backups.

## Troubleshooting

### DAG Not Showing Up

```bash
# Check for syntax errors
python airflow/dags/model_retraining_dag.py

# Restart scheduler
docker-compose restart airflow-scheduler
```

### DAG Import Errors

View logs:
```bash
docker-compose logs airflow-scheduler | grep ERROR
```

### Training Fails

1. Check task logs in Airflow UI
2. Verify data file exists: `data/raw/Flight_Price_Dataset_of_Bangladesh.csv`
3. Check disk space for model output
4. Review main.py logs

### Email Not Sending

1. Verify SMTP settings in docker-compose.yml
2. Check Airflow logs for SMTP errors
3. Test with manual trigger

## Advanced Usage

### Modify Training Command

Edit `airflow/dags/model_retraining_dag.py`:

```python
train_model = BashOperator(
    task_id='run_model_training',
    bash_command='cd /opt/airflow && python main.py --custom-args',
    dag=dag,
)
```

### Add Custom Tasks

Add new PythonOperator or BashOperator tasks and update task dependencies:

```python
my_task = PythonOperator(
    task_id='my_custom_task',
    python_callable=my_function,
    dag=dag,
)

# Update dependencies
evaluate_and_decide >> my_task >> deploy_model
```

### Rollback Model

If you need to rollback to a previous model:

```bash
# Copy backup to current
cp models/backups/model_backup_20240210_020000.joblib models/best_model.joblib

# Restart Streamlit app
docker-compose restart airflow-server
```

## Stopping Airflow

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes all data)
docker-compose down -v
```

## Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [DAG Best Practices](https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html)
- [Cron Expression Generator](https://crontab.guru/)
