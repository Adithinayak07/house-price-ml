import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump
import os
import warnings
import numpy as np

from ..config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME,
    PATHS,
)

from ..data.preprocess import load_raw, preprocess_and_split

# configure mlflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def _safe_rmse(y_true, y_pred):
    """Compute RMSE without relying on sklearn's squared kwarg (compatibility)."""
    mse = mean_squared_error(y_true, y_pred)
    return float(mse ** 0.5)


def train(config: dict):
    """Trains Linear Regression model with MLflow tracking."""
    
    # Load dataset
    df = load_raw(config["raw_path"])
    target_col = config["target_column"]

    # Preprocess + split
    X_train, X_test, y_train, y_test, preprocessor = preprocess_and_split(df, target_col)

    # Store original test values BEFORE log-transform
    y_test_original = y_test.copy()

    # Apply log transform
    y_train = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    model = LinearRegression()

    with mlflow.start_run(run_name="linear_regression_house_rent") as run:

        # Train on log targets
        model.fit(X_train, y_train)

        # Predict log values
        log_preds = model.predict(X_test)

        # Convert back to real values
        preds = np.expm1(log_preds)

        # Evaluate using REAL y_test values
        mae = float(mean_absolute_error(y_test_original, preds))
        rmse = _safe_rmse(y_test_original, preds)
        r2 = float(r2_score(y_test_original, preds))

        # Log parameters & metrics
        mlflow.log_param("model_type", "LinearRegression (log1p transformed)")
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)

        # Log preprocessing pipeline
        try:
            if os.path.exists(PATHS.pipeline_artifact):
                mlflow.log_artifact(PATHS.pipeline_artifact, artifact_path="preprocessing")
        except Exception as e:
            warnings.warn(f"Could not log preprocessing pipeline to mlflow: {e}")

        # Log model
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=REGISTERED_MODEL_NAME
            )
        except Exception:
            mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        # Save local copy
        os.makedirs("models", exist_ok=True)
        dump(model, "models/linear_model.joblib")
        dump(preprocessor, "models/preprocessing_pipeline.joblib")


        print(f"Model saved locally at: models/linear_model.joblib")
        experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
        exp_id = experiment.experiment_id
        print(f"Model saved locally at: models/linear_model.joblib")
        print(f"🏃 View run {run.info.run_name} at: {MLFLOW_TRACKING_URI}/#/experiments/{exp_id}/runs/{run.info.run_id}")
        print(f"🧪 View experiment at: {MLFLOW_TRACKING_URI}/#/experiments/{exp_id}")
        return run.info.run_id


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)
