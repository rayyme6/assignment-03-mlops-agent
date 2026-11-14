#!/usr/bin/env bash
# Run from the repository root. Airflow has its own environment and constraints.
set -euo pipefail
repo_root="$(pwd)"
airflow_version=3.1.3
python3 -m venv .venv-airflow
.venv-airflow/bin/python -m pip install --upgrade pip
python_version="$(.venv-airflow/bin/python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
constraints="https://raw.githubusercontent.com/apache/airflow/constraints-${airflow_version}/constraints-${python_version}.txt"
.venv-airflow/bin/python -m pip install "apache-airflow==${airflow_version}" --constraint "$constraints"
.venv-airflow/bin/python -m pip install "apache-airflow==${airflow_version}" pandas scikit-learn joblib PyYAML mlflow --constraint "$constraints"
export AIRFLOW_HOME="$repo_root/.airflow"
export AIRFLOW__CORE__DAGS_FOLDER="$repo_root/dags"
export AIRFLOW__CORE__LOAD_EXAMPLES=False
export AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS="admin:admin"
export PYTHONPATH="$repo_root${PYTHONPATH:+:$PYTHONPATH}"
export APP_ROOT="$repo_root"
exec .venv-airflow/bin/airflow standalone
