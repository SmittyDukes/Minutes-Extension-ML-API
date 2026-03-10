from fastapi import FastAPI
import pickle

app = FastAPI()

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict():
    features = [[minutes_played, fatigue_index, fouls, time_left, score_margin, timeouts_left]]
    probability = model.predict_proba(features)[0][1]
    return {"extend_probability": probability}