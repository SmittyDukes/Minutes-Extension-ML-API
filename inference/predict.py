import joblib
import pandas as pd

# load model once when the service starts
model = joblib.load("model/model.pkl")


def predict_extension(minutes_played,
                      fatigue_index,
                      fouls,
                      time_left,
                      score_margin,
                      timeouts_left):

    game_state = pd.DataFrame([{
        "minutes_played": minutes_played,
        "fatigue_index": fatigue_index,
        "fouls": fouls,
        "time_left": time_left,
        "score_margin": score_margin,
        "timeouts_left": timeouts_left
    }])

    probability = model.predict_proba(game_state)[0][1]

    return probability