import joblib
import pandas as pd
from typing import Dict, Any
import numpy as np

PIPELINE_PATH = "models/preprocessing_pipeline.joblib"
MODEL_PATH = "models/linear_model.joblib"


def load_pipeline_and_model(pipeline_path: str = PIPELINE_PATH, model_path: str = MODEL_PATH):
    """Loads preprocessing pipeline and model."""
    pipeline = joblib.load(pipeline_path)
    model = joblib.load(model_path)
    return pipeline, model


def predict_from_row(model, pipeline, row: Dict[str, Any]) -> float:
    """
    Predict rent from raw feature input using log-transform model.
    """
    df = pd.DataFrame([row])

    # Ensure categorical columns remain string type
    categorical_cols = ["locality", "facing", "parking"]
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)

    # Transform using the SAME pipeline used during training
    X = pipeline.transform(df)

    # Predict log(rent + 1)
    log_pred = model.predict(X)

    # Convert back to real rent (ensures no negative values)
    final_pred = np.expm1(log_pred)

    # Final safety check (never negative)
    return float(max(final_pred[0], 0.0))
