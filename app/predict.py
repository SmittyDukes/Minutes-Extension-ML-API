import joblib
import json
import pandas as pd
from logs.logging import log_decision_event

# load models once when the service starts
model = joblib.load("models/v1/model.pkl")

with open("models/v1/metadata.json") as f:
    metadata = json.load(f)

MODEL_VERSION = metadata["model_version"]
FEATURE_ORDER = metadata["feature_order"]

def predict_extension(request):
    feature_map = {"minutes_played": request.minutes_played,
                   "fatigue_index": request.fatigue_index,
                   "fouls": request.fouls,
                   "time_left": request.time_left,
                   "score_margin": request.score_margin,
                   "timeouts_left": request.timeouts_left,}
    features = [[feature_map[f] for f in FEATURE_ORDER]]
    probability = model.predict_proba(features)[0][1]


    decision_event = {
        "decision_id": decision_id,
        "model_version": model_version,
        "extend_probability": probability
    }
    assert set(feature_map.keys()) == set(FEATURE_ORDER)
    log_decision_event(decision_event)

    return probability, MODEL_VERSION