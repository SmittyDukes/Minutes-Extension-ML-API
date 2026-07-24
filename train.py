import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from policy import print_threshold_table

# Load the feature set from build_features.py
df = pd.read_csv("features.csv")


# Player-based split: test on 3 players the model never trains on
test_ids = [1630558, 1630202, 1629675]
test_df  = df[df['Player_ID'].isin([1630558, 1630202, 1629675])]
train_df = df[~df['Player_ID'].isin([1630558, 1630202, 1629675])]

features = ['days_rest', 'is_away', 'rolling_avg_min', 'prev_min']
X_train = train_df[features]
y_train = train_df['label']
X_test  = test_df[features]
y_test  = test_df['label']

# Scale features (fit on train only, so no leakage happens)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Train - class_weights='balanced' since positives anr the minority class
model = LogisticRegression(class_weight='balanced')
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

# Evaluation
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(dict(zip(features, model.coef_[0])))

# Abstention threshold sweep - precision/recall as the confidence band widens
pairs_to_try = [
    (0.50, 0.50),  # no abstention, same as your current model.predict()
    (0.40, 0.60),
    (0.35, 0.65),
    (0.30, 0.70),
    (0.25, 0.75),
]


print_threshold_table(list(y_test), y_prob, pairs_to_try)
