import pandas as pd

df = pd.read_csv("/home/sai/Downloads/football/Final/data/cleaned_matches_1990.csv")
df["date"] = pd.to_datetime(df["date"])

df = df.sort_values("date").reset_index(drop=True)

# store team stats
team_history = {}

rows = []

for i, row in df.iterrows():

    home = row["home_team"]
    away = row["away_team"]

    # initialize
    if home not in team_history:
        team_history[home] = []
    if away not in team_history:
        team_history[away] = []

    # previous 5 results
    home_last5 = team_history[home][-5:]
    away_last5 = team_history[away][-5:]

    home_form = sum(home_last5)
    away_form = sum(away_last5)

    rows.append({
        "date": row["date"],
        "home_team": home,
        "away_team": away,
        "home_score": row["home_score"],
        "away_score": row["away_score"],
        "neutral": int(row["neutral"]),
        "home_form": home_form,
        "away_form": away_form,
        "form_diff": home_form - away_form
    })

    # update points after match
    if row["home_score"] > row["away_score"]:
        team_history[home].append(3)
        team_history[away].append(0)
    elif row["home_score"] < row["away_score"]:
        team_history[home].append(0)
        team_history[away].append(3)
    else:
        team_history[home].append(1)
        team_history[away].append(1)

new_df = pd.DataFrame(rows)

new_df.to_csv("/home/sai/Downloads/football/Final/data/features.csv", index=False)

print(new_df.head())
print("Saved features.csv")