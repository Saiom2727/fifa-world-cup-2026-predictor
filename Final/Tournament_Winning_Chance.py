import pandas as pd
import numpy as np
import joblib
from collections import defaultdict

# =====================================================
# LOAD MODELS
# =====================================================

group_model = joblib.load("catboost_draw_model.pkl")   # draw-friendly model
ko_model    = joblib.load("advanced_model.pkl")        # knockout model

# =====================================================
# TEAM STRENGTHS (EDIT IF NEEDED)
# Higher = stronger
# =====================================================

team_elo = {
    "Mexico":1680, "South Africa":1500, "Korea Republic":1650, "Czechia":1700,
    "Canada":1720, "Bosnia-Herzegovina":1580, "Qatar":1550, "Switzerland":1780,
    "Brazil":1900, "Morocco":1760, "Haiti":1450, "Scotland":1710,
    "USA":1740, "Paraguay":1660, "Australia":1640, "Türkiye":1720,
    "Germany":1860, "Curaçao":1480, "Côte d'Ivoire":1650, "Ecuador":1730,
    "Netherlands":1840, "Japan":1740, "Sweden":1710, "Tunisia":1600,
    "Belgium":1800, "Egypt":1600, "IR Iran":1640, "New Zealand":1500,
    "Spain":1880, "Cabo Verde":1500, "Saudi Arabia":1580, "Uruguay":1810,
    "France":1890, "Senegal":1720, "Iraq":1500, "Norway":1690,
    "Argentina":1910, "Algeria":1650, "Austria":1710, "Jordan":1480,
    "Portugal":1840, "Congo DR":1540, "Uzbekistan":1560, "Colombia":1790,
    "England":1870, "Croatia":1760, "Ghana":1610, "Panama":1540
}

# =====================================================
# GROUPS
# =====================================================

groups = {
"A":["Mexico","South Africa","Korea Republic","Czechia"],
"B":["Canada","Bosnia-Herzegovina","Qatar","Switzerland"],
"C":["Brazil","Morocco","Haiti","Scotland"],
"D":["USA","Paraguay","Australia","Türkiye"],
"E":["Germany","Curaçao","Côte d'Ivoire","Ecuador"],
"F":["Netherlands","Japan","Sweden","Tunisia"],
"G":["Belgium","Egypt","IR Iran","New Zealand"],
"H":["Spain","Cabo Verde","Saudi Arabia","Uruguay"],
"I":["France","Senegal","Iraq","Norway"],
"J":["Argentina","Algeria","Austria","Jordan"],
"K":["Portugal","Congo DR","Uzbekistan","Colombia"],
"L":["England","Croatia","Ghana","Panama"]
}

# =====================================================
# HELPERS
# =====================================================

def get_elo(team):
    return team_elo.get(team, 1500)

def get_form(team):
    # placeholder
    return 10

# -----------------------------------------------------

def make_group_features(team1, team2):

    elo1 = get_elo(team1)
    elo2 = get_elo(team2)

    f1 = get_form(team1)
    f2 = get_form(team2)

    diff = elo1 - elo2

    return pd.DataFrame([{
        "home_team": team1,
        "away_team": team2,
        "neutral": 1,
        "home_form": f1,
        "away_form": f2,
        "form_diff": f1-f2,
        "elo_home": elo1,
        "elo_away": elo2,
        "elo_diff": diff,
        "close_match": abs(diff)
    }])

# -----------------------------------------------------

def make_ko_features(team1, team2):

    elo1 = get_elo(team1)
    elo2 = get_elo(team2)

    f1 = get_form(team1)
    f2 = get_form(team2)

    diff = elo1 - elo2

    return pd.DataFrame([{
        "home_team": team1,
        "away_team": team2,
        "neutral": 1,
        "home_form": f1,
        "away_form": f2,
        "form_diff": f1-f2,
        "elo_home": elo1,
        "elo_away": elo2,
        "elo_diff": diff,
        "abs_elo_diff": abs(diff)
    }])

# =====================================================
# MATCH FUNCTIONS
# =====================================================

def play_group_match(team1, team2):

    X = make_group_features(team1, team2)

    probs = group_model.predict_proba(X)[0]

    result = np.random.choice([0,1,2], p=probs)

    # proxy goals
    if result == 0:
        return (3,0,1,0)
    elif result == 1:
        return (1,1,1,1)
    else:
        return (0,3,0,1)

# -----------------------------------------------------

def play_ko_match(team1, team2):

    X = make_ko_features(team1, team2)

    probs = ko_model.predict_proba(X)[0]

    p1 = probs[0]
    p2 = probs[2]

    total = p1 + p2

    p1 /= total
    p2 /= total

    return np.random.choice([team1,team2], p=[p1,p2])

# =====================================================
# GROUP SIMULATION
# =====================================================

def simulate_group(teams):

    table = {}

    for t in teams:
        table[t] = {
            "team":t,
            "pts":0,
            "gd":0,
            "gf":0
        }

    for i in range(len(teams)):
        for j in range(i+1,len(teams)):

            t1 = teams[i]
            t2 = teams[j]

            p1,p2,g1,g2 = play_group_match(t1,t2)

            table[t1]["pts"] += p1
            table[t2]["pts"] += p2

            table[t1]["gf"] += g1
            table[t2]["gf"] += g2

            table[t1]["gd"] += (g1-g2)
            table[t2]["gd"] += (g2-g1)

    standings = list(table.values())

    standings.sort(
        key=lambda x:(x["pts"],x["gd"],x["gf"]),
        reverse=True
    )

    return standings

# =====================================================
# THIRD PLACE SORT
# =====================================================

def rank_third_places(thirds):

    thirds.sort(
        key=lambda x:(x["pts"],x["gd"],x["gf"]),
        reverse=True
    )

    return thirds[:8]

# =====================================================
# TOURNAMENT
# =====================================================

def simulate_once():

    winners = {}
    runners = {}
    thirds = []

    # -----------------------------------------
    # GROUPS
    # -----------------------------------------

    for g in groups:

        table = simulate_group(groups[g])

        winners[g] = table[0]["team"]
        runners[g] = table[1]["team"]

        third = table[2]
        third["group"] = g
        thirds.append(third)

    best_thirds = rank_third_places(thirds)

    best_third_teams = [x["team"] for x in best_thirds]

    # 32 qualified teams
    qualified = []

    for g in groups:
        qualified.append(winners[g])
        qualified.append(runners[g])

    qualified.extend(best_third_teams)

    # -----------------------------------------
    # ROUND OF 32
    # Randomized bracket version
    # -----------------------------------------

    np.random.shuffle(qualified)

    r32_winners = []

    for i in range(0,32,2):
        r32_winners.append(
            play_ko_match(qualified[i], qualified[i+1])
        )

    # ROUND OF 16
    r16 = []

    for i in range(0,16,2):
        r16.append(play_ko_match(r32_winners[i], r32_winners[i+1]))

    # QUARTERS
    qf = []

    for i in range(0,8,2):
        qf.append(play_ko_match(r16[i], r16[i+1]))

    # SEMIS
    sf = []

    for i in range(0,4,2):
        sf.append(play_ko_match(qf[i], qf[i+1]))

    # FINAL
    champion = play_ko_match(sf[0], sf[1])

    return champion

# =====================================================
# MONTE CARLO
# =====================================================

def monte_carlo(n=1000):

    wins = defaultdict(int)

    for _ in range(n):
        champ = simulate_once()
        wins[champ]+=1

    results = sorted(
        wins.items(),
        key=lambda x:x[1],
        reverse=True
    )

    print("\n🏆 2026 FIFA WORLD CUP WIN PROBABILITIES\n")

    for team,count in results:
        pct = round(count*100/n,2)
        print(f"{team:<20} {pct}%")

# =====================================================
# MAIN
# =====================================================

monte_carlo(1000)