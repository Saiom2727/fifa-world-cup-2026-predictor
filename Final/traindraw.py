import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from catboost import CatBoostClassifier

# Load data
df = pd.read_csv("/home/sai/Downloads/football/Final/data/features_with_elo.csv")

# Target
def get_result(row):
    if row["home_score"] > row["away_score"]:
        return 0
    elif row["home_score"] == row["away_score"]:
        return 1
    else:
        return 2

df["result"] = df.apply(get_result, axis=1)

# Extra draw-friendly feature
df["close_match"] = abs(df["elo_diff"])

# Features
X = df[[
    "home_team",
    "away_team",
    "neutral",
    "home_form",
    "away_form",
    "form_diff",
    "elo_home",
    "elo_away",
    "elo_diff",
    "close_match"
]]

y = df["result"]

# Time-based split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    shuffle=False
)

# Categorical columns
cat_features = ["home_team", "away_team"]

# Model
model = CatBoostClassifier(
    iterations=700,
    depth=7,
    learning_rate=0.04,
    loss_function="MultiClass",
    eval_metric="Accuracy",

    # Draw boosted
    class_weights=[1.0, 1.4, 1.1],

    verbose=100
)

# Train
model.fit(X_train, y_train, cat_features=cat_features)

# Predict
pred = model.predict(X_test)

# Metrics
acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)
print(classification_report(y_test, pred))

# Save
joblib.dump(model, "catboost_draw_model.pkl")

print("Saved catboost_draw_model.pkl")