from pydantic import BaseModel

class PredictionRequest(BaseModel):
    minutes_played: int
    fatigue_index: float
    fouls: int
    time_left: int
    score_margin: int
    timeouts_left: int

class PredictionResponse(BaseModel):
    decision_id: str
    model_version: str
    extend_probability: float