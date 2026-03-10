from fastapi import FastAPI
import uuid

from inference.predict import predict_extension

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Basketball decision API running"}


@app.get("/predict")
def predict(minutes_played: int,
            fatigue_index: float,
            fouls: int,
            time_left: int,
            score_margin: int,
            timeouts_left: int):

    probability = predict_extension(
        minutes_played,
        fatigue_index,
        fouls,
        time_left,
        score_margin,
        timeouts_left
    )

    decision_id = str(uuid.uuid4())

    return {
        "decision_id": decision_id,
        "extend_probability": probability
    }