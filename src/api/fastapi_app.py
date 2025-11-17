from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any
import joblib
from ..models.predict import load_pipeline_and_model, predict_from_row

app = FastAPI(
    title="House Rent Prediction API",
    description="API for predicting house rent",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    data: Dict[str, Any]

pipeline = None
model = None


@app.on_event("startup")
def startup_event():
    global pipeline, model
    try:
        pipeline, model = load_pipeline_and_model(
            pipeline_path="models/preprocessing_pipeline.joblib",
            model_path="models/linear_model.joblib"   # rename your model to this
        )
        print("🚀 Model & pipeline loaded successfully!")
    except Exception as e:
        print(f"❌ Failed to load model or pipeline: {e}")


@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        result = predict_from_row(model, pipeline, request.data)
        return {"prediction": round(result, 2)}
    except Exception as e:
        return {"error": f"Prediction failed: {e}"}



@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None
    }
