## Minutes Extension ML System

Machine learning system that predicts whether a basketball player's minutes
should be extended based on fatigue, fouls, and game context.

## System Architecture

Training Pipeline
↓
Serialized Model Artifact
↓
FastAPI Inference Service
↓
Decision Logging (JSONL)
↓
Docker Container

## API Endpoints

POST /predict  
GET /health  
GET /model-info

## Example Request

POST /predict

{
 "minutes_played": 30,
 "fatigue_index": 0.55,
 "fouls": 3,
 "time_left": 240,
 "score_margin": 2,
 "timeouts_left": 1
}

## Example Response

{
 "decision_id": "abc123",
 "extend_probability": 0.64,
 "model_version": "minutes_extension_v1"
}