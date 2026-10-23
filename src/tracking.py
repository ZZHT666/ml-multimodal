"""MLflow setup shared by the training scripts.

Tracking store: MLFLOW_TRACKING_URI (e.g. sqlite:///mlflow.db -- run metadata, small).
Artifact store: MLFLOW_ARTIFACT_ROOT (e.g. gs://<bucket>/mlflow-artifacts -- checkpoints,
large). On the VM, artifacts therefore land in GCS next to the data, not on a local disk.
"""
import os


def set_experiment(name: str):
    import mlflow

    root = os.environ.get("MLFLOW_ARTIFACT_ROOT")
    if root and mlflow.get_experiment_by_name(name) is None:
        # The artifact location is fixed when an experiment is created.
        mlflow.create_experiment(name, artifact_location=f"{root.rstrip('/')}/{name}")
    mlflow.set_experiment(name)
    return mlflow
