import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np

df = pd.read_csv("features.csv")

test_ids = [1630558, 1630202, 1629675]
test_df  = df[df['Player_ID'].isin([1630558, 1630202, 1629675])]
train_df = df[~df['Player_ID'].isin([1630558, 1630202, 1629675])]

features = ['days_rest', 'is_away', 'rolling_avg_min', 'prev_min']

X_train = train_df[features]
y_train = train_df['label']
X_test  = test_df[features]
y_test  = test_df['label']

print(len(X_train), len(X_test))   # should be 968 605

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

model = LogisticRegression()
model.fit(X_train_scaled, y_train)
predictions = model.predict(X_test_scaled)

from policy import print_threshold_table

y_true = list(y_test)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # probability of class 1 per game

pairs_to_try = [
    (0.50, 0.50),  # no abstention, same as your current model.predict()
    (0.40, 0.60),
    (0.35, 0.65),
    (0.30, 0.70),
    (0.25, 0.75),
]

y_prob = model.predict_proba(X_test_scaled)[:, 1]

print_threshold_table(y_true, y_prob, pairs_to_try)
print(predictions[:20])
print(dict(zip(features, model.coef_[0])))
print(dict(zip(X_train.columns, model.coef_[0])))
print(train_df['label'].value_counts(normalize=True))
print(test_df['label'].value_counts(normalize=True))
print(len(df))
print(len(train_df))
print(len(test_df))
print(predictions[:20])
print(dict(zip(X_train.columns, model.coef_[0])))
print (classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print((y_prob > 0.7).sum())
print(y_prob.max())
print(y_prob.min())