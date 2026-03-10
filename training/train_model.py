import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

df = pd.read_csv("../data/game_data.csv")
X = df[[ "minutes_played", "fatigue_index", "fouls", "time_left", "score_margin", "timeouts_left"]]
y = df["points_scored"]

X_train, X_test, y_train, y_test= train_test_split(X, y, test_size=0.2, random_state=42)
model= LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mse= mean_squared_error(y_test, predictions)
print("MOdel MSE:", mse)

joblib.dump(model, '../models/points_model.pkl')
print("Model saved to models/points_model.pkl")