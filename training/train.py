import subprocess
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from mlflow.models import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


ROOT = Path(__file__).resolve().parents[1]

# Load dataset
df = pd.read_csv(ROOT / "data" / "iris.csv")

X = df.drop(columns=["target"])
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Retrieve actual dataset provenance
git_commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    cwd=ROOT,
    text=True
).strip()

with open(ROOT / "data" / "iris.csv.dvc") as file:
    dvc_metadata = yaml.safe_load(file)

dvc_hash = dvc_metadata["outs"][0]["md5"]

# MLflow configuration
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Iris-Classification")

# Train multiple experiments
for n_estimators in [50, 100, 200]:

    with mlflow.start_run(
        run_name=f"RF-{n_estimators}"
    ) as run:

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        f1 = f1_score(
            y_test,
            predictions,
            average="macro"
        )

        # Log parameters
        mlflow.log_param(
            "n_estimators",
            n_estimators
        )

        # Log evaluation metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Connect experiment to DVC provenance
        mlflow.set_tags({
            "git_commit": git_commit,
            "dvc_hash": dvc_hash,
            "dataset_path": "data/iris.csv"
        })

        # Save model
        signature = infer_signature(
            X_train,
            model.predict(X_train)
        )

        mlflow.sklearn.log_model(
            model,
            name="model",
            signature=signature
        )

        print(
            f"Run: {run.info.run_id} | "
            f"Trees: {n_estimators} | "
            f"Accuracy: {accuracy:.4f} | "
            f"F1: {f1:.4f}"
        )