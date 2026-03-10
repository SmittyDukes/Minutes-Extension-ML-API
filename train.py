import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle


data = {"minutes_played":[30,31,32,33,34,35,36,37], "fatigue_index":[1,2,3,1,2,3,1,2], "fouls":[1,2,3,4,5,1,2,3],"time_left":[120, 120, 120, 120, 120, 120, 120, 120], "score_margin":[1,2,3,4,5,6,7,8],"timeouts_left":[1,2,1,2,1,2,1,2], "extend_minutes":[1,0,1,0,1,0,1,0]}
df = pd.DataFrame(data)

X = df[['minutes_played','fatigue_index','fouls','time_left','score_margin','timeouts_left']]
y = df['extend_minutes']

model = LogisticRegression()
model.fit(X,y)

with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)