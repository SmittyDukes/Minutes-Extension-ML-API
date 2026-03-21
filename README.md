## Minutes Extension ML System

Machine learning system that predicts whether a basketball player's minutes
should be extended based on fatigue, fouls, and game context.This project demonstrates a **production-style machine learning service**, including model training, artifact versioning, API inference, decision logging, and Docker deployment.


## Overview
This system goes beyond simple prediction by converting model outputs into structured, policy-governed decisions.
A logistic regression model generates a probability of extending minutes.
A decision layer applies threshold-based policy logic and constraint checks to produce a final decision.
Each prediction is validated, structured, and logged for traceability.

## How Decisions Are Made
The API receives and validates inputs using Pydantic to enforce schema and type correctness.
Validated features are passed to a logistic regression model, which outputs a probability of extending minutes.
A decision layer applies policy constraints and a threshold to convert this probability into a discrete decision (extend or do_not_extend).
If constraints are violated or inputs are invalid, the system abstains and returns a structured failure response.
All outputs follow a consistent schema including decision, reason, and confidence.
Each prediction is logged as a full decision event containing both input features and model outputs for traceability.

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

## Repository Structure

app/        FastAPI service and inference logic  
training/   Model training pipeline  
models/     Versioned model artifacts  
logs/       Decision logging utilities  
data/       Training dataset  

Dockerfile  Container runtime definition  
requirements.txt  Python dependencies

## Model

Algorithm: Logistic Regression  
Framework: scikit-learn  

Features:
- minutes_played
- fatigue_index
- fouls
- time_left
- score_margin
- timeouts_left

Output:
Probability that a player's minutes should be extended.

The model artifact is stored in:
model/v1/model.pkl

Model metadata is stored in:
models/v1/metadata.json

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
  "decision_id": "uuid",
  "extend_probability": 0.64,
  "threshold": 0.6,
  "decision": "extend",
  "reason": "probability_above_threshold",
  "confidence": 0.64,
  "model_version": "v1"
}

## GET/health
{
  "status": "ok"
}



## Decision Logging
Every prediction event is logged to a JSONL decision log.

{
 "decision_id": "abc123",
 "model_version": "minutes_extension_v1",
 "input_features": {...},
 "extend_probability": 0.64,
 "timestamp": "2026-03-04T20:11:22"
}
This enables:
traceability
debugging
auditing model decisions



## Running the API

Install dependencies:

pip install -r requirements.txt

Start the service:

uvicorn app.api:app --reload

Open the API docs:

http://127.0.0.1:8000/docs

## Docker

Build container:

docker build -t minutes-extension-ml .

Run container:

docker run -p 8000:8000 minutes-extension-ml


## Engineering Decisions
Logistic Regression was chosen for its interpretability and well-calibrated probability outputs, which are critical for threshold-based decision systems.
Features were selected based on domain relevance to player fatigue, game context, and foul risk, capturing key factors that influence real-world coaching decisions.
JSONL logging was used to record each decision event as a structured, append-only record, enabling efficient auditing, traceability, and post-hoc analysis.

## Failure Handling
The system is designed to fail safely:
Invalid or missing inputs → abstain decision
Probability outside valid range → abstain
Missing required features → abstain
All failure cases return structured responses with clear reasoning.
## Project Goals

This repository demonstrates the core components of a production-style machine learning system:
model training pipeline
versioned model artifacts
inference service
request validation
decision logging
containerized deployment
