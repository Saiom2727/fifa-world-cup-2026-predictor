import pandas as pd

# Load existing features file
df = pd.read_csv("/home/sai/Downloads/football/Final/data/features.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").reset_index(drop=True)

# ---------- INITIAL ELO RATINGS ----------

elite = {
    "Brazil", "Argentina", "France", "Spain",
    "England", "Germany"
}

strong = {
    "Portugal", "Netherlands", "Belgium",
    "Italy", "Uruguay"
}

good = {
    "Mexico", "United States", "Japan",
    "Croatia", "Denmark", "Switzerland"
}

weak = {
    "San Marino", "Andorra", "Liechtenstein",
    "Guam", "Bhutan", "Mongolia"
}

elo = {}

def get_start_elo(team):
    if team in elite:
        return 1850
    elif team in strong:
        return 1750
    elif team in good:
        return 1650
    elif team in weak:
        return 1400
    else:
        return 1500

# ---------- ELO FUNCTIONS ----------

def expected_score(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))

def update_elo(r1, r2, score1, score2, k=20):
    exp1 = expected_score(r1, r2)
    exp2 = expected_score(r2, r1)

    new_r1 = r1 + k * (score1 - exp1)
    new_r2 = r2 + k * (score2 - exp2)

    return new_r1, new_r2

# ---------- PROCESS MATCHES ----------

elo_home = []
elo_away = []
elo_diff = []

for _, row in df.iterrows():

    home = row["home_team"]
    away = row["away_team"]

    if home not in elo:
        elo[home] = get_start_elo(home)

    if away not in elo:
        elo[away] = get_start_elo(away)

    home_rating = elo[home]
    away_rating = elo[away]

    elo_home.append(home_rating)
    elo_away.append(away_rating)
    elo_diff.append(home_rating - away_rating)

    # Match result
    if row["home_score"] > row["away_score"]:
        s_home = 1
        s_away = 0
    elif row["home_score"] < row["away_score"]:
        s_home = 0
        s_away = 1
    else:
        s_home = 0.5
        s_away = 0.5

    new_home, new_away = update_elo(
        home_rating, away_rating, s_home, s_away
    )

    elo[home] = new_home
    elo[away] = new_away

# ---------- SAVE ----------

df["elo_home"] = elo_home
df["elo_away"] = elo_away
df["elo_diff"] = elo_diff

df.to_csv("/home/sai/Downloads/football/Final/data/features_with_elo.csv", index=False)

print("Saved: data/features_with_elo.csv")
print(df[[
    "home_team","away_team",
    "elo_home","elo_away","elo_diff"
]].head())