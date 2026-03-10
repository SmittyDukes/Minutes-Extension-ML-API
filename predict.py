import pandas as pd
import pickle

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

features = pd.DataFrame([{
    "minutes_played": 30,
    "fatigue_index": 0.55,
    "fouls": 3,
    "time_left": 240,
    "score_margin": 2,
    "timeouts_left": 1
}])
probability = model.predict_proba(features)[0][1]
print(probability)
