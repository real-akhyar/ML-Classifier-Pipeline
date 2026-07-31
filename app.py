import os
import joblib
import numpy as np
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI App
app = FastAPI(
    title="ML Wine Classifier API",
    description="Production-ready REST API for Scikit-Learn Wine Classification Model (Roptal ready)",
    version="1.0.0"
)

# Global variables for model artifacts
model = None
scaler = None
feature_names = None
CLASS_NAMES = ['class_0', 'class_1', 'class_2']

class PredictRequest(BaseModel):
    features: List[float] = Field(
        ...,
        example=[13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050.0],
        description="List of 13 numeric features matching wine dataset"
    )

class PredictResponse(BaseModel):
    status: str
    class_index: int
    class_name: str
    probabilities: dict

@app.on_event("startup")
def load_artifacts():
    global model, scaler, feature_names
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "models", "model.joblib")
    scaler_path = os.path.join(base_dir, "models", "scaler.joblib")
    feature_path = os.path.join(base_dir, "models", "feature_names.joblib")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        # Auto-train if missing
        from src.train import train_and_export
        train_and_export()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    if os.path.exists(feature_path):
        feature_names = joblib.load(feature_path)

@app.get("/")
def root():
    return {
        "service": "ML Wine Classifier API",
        "status": "online",
        "version": "1.0.0",
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health")
def health_check():
    is_ready = model is not None and scaler is not None
    if not is_ready:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "scaler_loaded": True,
        "num_features_expected": len(feature_names) if feature_names else 13
    }

@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(payload: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    
    expected_dim = len(feature_names) if feature_names else 13
    if len(payload.features) != expected_dim:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {expected_dim} features, but got {len(payload.features)}"
        )

    try:
        arr = np.array(payload.features).reshape(1, -1)
        scaled = scaler.transform(arr)
        pred_idx = int(model.predict(scaled)[0])
        probs = model.predict_proba(scaled)[0]

        prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}

        return PredictResponse(
            status="success",
            class_index=pred_idx,
            class_name=CLASS_NAMES[pred_idx],
            probabilities=prob_dict
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
