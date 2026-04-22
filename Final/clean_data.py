import pandas as pd

# Load datasets
intl = pd.read_csv("/home/sai/Downloads/football/Final/data/matches.csv")
fifa = pd.read_csv("/home/sai/Downloads/football/Final/data/fifamatches.csv")

# Select useful columns from fifa
fifa = fifa[
    [
        "match_date",
        "home_team_name",
        "away_team_name",
        "home_team_score",
        "away_team_score",
        "tournament_name",
        "country_name"
    ]
]

# Rename columns
fifa.columns = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "country"
]

# Add missing columns
fifa["city"] = fifa["country"]
fifa["neutral"] = True

# Reorder columns
fifa = fifa[
    [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral"
    ]
]

# Combine datasets
df = pd.concat([intl, fifa], ignore_index=True)

# Convert date
df["date"] = pd.to_datetime(df["date"], errors="coerce")

# Remove invalid dates
df = df.dropna(subset=["date"])

# Keep only 1990 onwards
df = df[df["date"].dt.year >= 1990]

# Remove duplicates
df.drop_duplicates(
    subset=["date", "home_team", "away_team"],
    keep="first",
    inplace=True
)

# Sort by date
df = df.sort_values("date")

# Save
df.to_csv("/home/sai/Downloads/football/Final/data/cleaned_matches_1990.csv", index=False)

print("✅ Saved: /home/sai/Downloads/football/Final/data/cleaned_matches_1990.csv")
print("Total matches:", len(df))
print(df.head())