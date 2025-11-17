from dataclasses import dataclass
import os

# MLflow settings
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "house_rent_experiment")
REGISTERED_MODEL_NAME = os.getenv("REGISTERED_MODEL_NAME", "HouseRentLinearModel")

# Preprocessing settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

@dataclass
class Paths:
    raw_data: str = "data/raw/house_rent.csv"
    processed_train: str = "data/processed/train.csv"
    processed_test: str = "data/processed/test.csv"
    pipeline_artifact: str = "models/preprocessing_pipeline.joblib"

PATHS = Paths()
