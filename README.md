## Inference API

Start server

uvicorn inference.api:app --reload

Example request

GET /predict?minutes_played=30&fatigue_index=0.55&fouls=3&time_left=240&score_margin=2&timeouts_left=1

Example response

{
  "decision_id": "...",
  "extend_probability": 0.63
}