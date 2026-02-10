# Airflow image for model retraining orchestration
FROM apache/airflow:2.8.1-python3.11

USER root

# Install system dependencies and Python dev tools for virtualenv
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    procps \
    python3-dev \
    python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER airflow

# Install only minimal dependencies for Airflow DAG parsing
# pandas is needed for data validation tasks
RUN pip install --no-cache-dir pandas==2.1.4

# Note: ML training dependencies will be installed in a separate virtualenv
# by PythonVirtualenvOperator at runtime to avoid conflicts with Airflow
