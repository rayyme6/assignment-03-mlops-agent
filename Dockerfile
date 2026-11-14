# Build the Airflow target only for the local Compose demonstration.
FROM apache/airflow:3.1.3-python3.11 AS airflow
USER root
WORKDIR /opt/mlops
COPY --chown=airflow:root training /opt/mlops/training
COPY --chown=airflow:root dags /opt/mlops/dags
COPY --chown=airflow:root params.yaml /opt/mlops/params.yaml
USER airflow
RUN pip install --no-cache-dir 'apache-airflow==3.1.3' mlflow pandas scikit-learn joblib PyYAML \
    --constraint https://raw.githubusercontent.com/apache/airflow/constraints-3.1.3/constraints-3.11.txt
ENV PYTHONPATH=/opt/mlops APP_ROOT=/opt/mlops

FROM python:3.11-slim AS inference
WORKDIR /app
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
RUN pip install --no-cache-dir fastapi==0.162.1 uvicorn==0.46.0 \
    scikit-learn==1.7.2 pandas==2.3.3 joblib==1.5.3
COPY --chown=app:app api /app/api
COPY --chown=app:app artifacts/model.joblib artifacts/metadata.json /app/artifacts/
ENV MODEL_PATH=/app/artifacts/model.joblib PYTHONUNBUFFERED=1
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=15s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
