from fastapi import FastAPI
import uuid
import json

from app.predict import predict_extension
from app.schemas import PredictionRequest, PredictionResponse

from logs.logging import log_decision_event
from datetime import datetime

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Basketball decision API running"}
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": True,
        "service": "basketball_decision_api"
    }
@app.get("/model-info")
def model_info():
    with open("models/v1/metadata.json") as f:
        metadata = json.load(f)
    return metadata


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    probability, model_version = predict_extension(request)

    decision_id = str(uuid.uuid4())

    decision_event = {
        "decision_id": decision_id,
        "timestamp": datetime.utcnow().isoformat(),
        "model_version": model_version,
        "inputs": request.dict(),
        "probability": probability
    }

    log_decision_event(decision_event)

    return PredictionResponse(
        decision_id=decision_id,
        extend_probability=probability,
        model_version=model_version
    )
