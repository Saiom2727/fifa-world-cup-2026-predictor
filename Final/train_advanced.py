import pandas as pd
import numpy as np
import joblib

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------------
# LOAD DATA
# -----------------------------------
df = pd.read_csv("/home/sai/Downloads/football/Final/data/cleaned_matches_1990.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# -----------------------------------
# INITIAL ELO TIERS
# -----------------------------------
elite = {"Brazil","Argentina","France","Spain","England","Germany"}
strong = {"Portugal","Netherlands","Belgium","Italy","Uruguay"}
good = {"Mexico","United States","Japan","Croatia","Denmark","Switzerland"}

elo = {}

def start_elo(team):
    if team in elite:
        return 1850
    elif team in strong:
        return 1750
    elif team in good:
        return 1650
    else:
        return 1500

# -----------------------------------
# COMPETITION IMPORTANCE
# -----------------------------------
def get_k(tournament):

    t = str(tournament).lower()

    if "friendly" in t:
        return 12

    elif "qualif" in t:
        return 22

    elif "nations league" in t:
        return 24

    elif "cup" in t or "euro" in t or "copa" in t or "afcon" in t:
        return 28

    elif "world cup" in t:
        return 38

    else:
        return 20

# -----------------------------------
# ELO FUNCTIONS
# -----------------------------------
def expected(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1)/400))

def update_elo(r1, r2, s1, s2, k):
    e1 = expected(r1, r2)
    e2 = expected(r2, r1)

    nr1 = r1 + k*(s1-e1)
    nr2 = r2 + k*(s2-e2)

    return nr1, nr2

# -----------------------------------
# FEATURE CREATION
# -----------------------------------
rows = []
history = {}

for _, row in df.iterrows():

    home = row["home_team"]
    away = row["away_team"]

    if home not in elo:
        elo[home] = start_elo(home)
    if away not in elo:
        elo[away] = start_elo(away)

    if home not in history:
        history[home] = []
    if away not in history:
        history[away] = []

    # Recent form
    home_form = sum(history[home][-5:])
    away_form = sum(history[away][-5:])

    # Current Elo before match
    eh = elo[home]
    ea = elo[away]

    # Result class
    if row["home_score"] > row["away_score"]:
        result = 0
        s_home, s_away = 1,0
    elif row["home_score"] < row["away_score"]:
        result = 2
        s_home, s_away = 0,1
    else:
        result = 1
        s_home, s_away = 0.5,0.5

    rows.append({
        "date": row["date"],
        "home_team": home,
        "away_team": away,
        "neutral": int(row["neutral"]),
        "home_form": home_form,
        "away_form": away_form,
        "form_diff": home_form-away_form,
        "elo_home": eh,
        "elo_away": ea,
        "elo_diff": eh-ea,
        "abs_elo_diff": abs(eh-ea),
        "result": result
    })

    # Update recent form
    history[home].append(3 if s_home==1 else 1 if s_home==0.5 else 0)
    history[away].append(3 if s_away==1 else 1 if s_away==0.5 else 0)

    # Update Elo
    k = get_k(row["tournament"])
    newh, newa = update_elo(eh, ea, s_home, s_away, k)

    elo[home] = newh
    elo[away] = newa

# -----------------------------------
# DATAFRAME
# -----------------------------------
data = pd.DataFrame(rows)

# -----------------------------------
# TIME BASED SPLIT
# -----------------------------------
split_index = int(len(data)*0.8)

train = data.iloc[:split_index]
test  = data.iloc[split_index:]

X_train = train.drop(columns=["date","result"])
y_train = train["result"]

X_test = test.drop(columns=["date","result"])
y_test = test["result"]

cat_cols = ["home_team","away_team"]

# -----------------------------------
# CATBOOST TUNED
# -----------------------------------
model = CatBoostClassifier(
    iterations=1200,
    depth=8,
    learning_rate=0.03,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    l2_leaf_reg=5,
    random_strength=1,
    bagging_temperature=1,
    verbose=100
)

model.fit(
    X_train, y_train,
    cat_features=cat_cols,
    eval_set=(X_test,y_test),
    use_best_model=True,
    early_stopping_rounds=100
)

# -----------------------------------
# EVALUATE
# -----------------------------------
pred = model.predict(X_test)

acc = accuracy_score(y_test,pred)

print("Final Accuracy:", acc)
print(classification_report(y_test,pred))

# -----------------------------------
# SAVE
# -----------------------------------
joblib.dump(model,"advanced_model.pkl")
print("Saved advanced_model.pkl")