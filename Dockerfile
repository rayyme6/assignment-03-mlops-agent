FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY agent agent
COPY api api
COPY training training
RUN pip install --no-cache-dir . && useradd --uid 10001 --create-home app
COPY artifacts/model.joblib /app/artifacts/model.joblib
USER 10001
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
