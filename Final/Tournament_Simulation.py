import pandas as pd
import numpy as np
import joblib
from collections import defaultdict

# =====================================================
# LOAD MODELS
# =====================================================

group_model = joblib.load("catboost_draw_model.pkl")
ko_model    = joblib.load("advanced_model.pkl")

# =====================================================
# TEAM STRENGTHS
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
    return 10

def make_group_features(team1, team2):
    e1 = get_elo(team1)
    e2 = get_elo(team2)
    f1 = get_form(team1)
    f2 = get_form(team2)
    diff = e1 - e2
    return pd.DataFrame([{
        "home_team": team1,
        "away_team": team2,
        "neutral": 1,
        "home_form": f1,
        "away_form": f2,
        "form_diff": f1-f2,
        "elo_home": e1,
        "elo_away": e2,
        "elo_diff": diff,
        "close_match": abs(diff)
    }])

def make_ko_features(team1, team2):
    e1 = get_elo(team1)
    e2 = get_elo(team2)
    f1 = get_form(team1)
    f2 = get_form(team2)
    diff = e1 - e2
    return pd.DataFrame([{
        "home_team": team1,
        "away_team": team2,
        "neutral": 1,
        "home_form": f1,
        "away_form": f2,
        "form_diff": f1-f2,
        "elo_home": e1,
        "elo_away": e2,
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
    return result

def play_ko_match(team1, team2):
    X = make_ko_features(team1, team2)
    probs = ko_model.predict_proba(X)[0]
    p1 = probs[0]
    p2 = probs[2]
    total = p1+p2
    p1 /= total
    p2 /= total
    winner = np.random.choice([team1,team2], p=[p1,p2])
    return winner

# =====================================================
# GROUP STAGE
# =====================================================

def simulate_group(group_name, teams):
    table = {}
    for t in teams:
        table[t] = {"team":t, "pts":0, "gd":0, "gf":0}

    matches_rows = []

    for i in range(len(teams)):
        for j in range(i+1, len(teams)):
            t1 = teams[i]
            t2 = teams[j]
            result = play_group_match(t1, t2)

            if result == 0:
                table[t1]["pts"] += 3
                outcome1, outcome2 = "W", "L"
            elif result == 1:
                table[t1]["pts"] += 1
                table[t2]["pts"] += 1
                outcome1, outcome2 = "D", "D"
            else:
                table[t2]["pts"] += 3
                outcome1, outcome2 = "L", "W"

            matches_rows.append((t1, outcome1, outcome2, t2))

    standings = list(table.values())
    standings.sort(key=lambda x:(x["pts"],x["gd"],x["gf"]), reverse=True)
    return standings, matches_rows

# =====================================================
# TOURNAMENT
# =====================================================

def simulate_worldcup():
    html = ""
    winners = {}
    runners = {}
    thirds = []

    # GROUPS
    for g in groups:
        standings, match_rows = simulate_group(g, groups[g])
        winners[g] = standings[0]["team"]
        runners[g] = standings[1]["team"]
        third = standings[2]
        third["group"] = g
        thirds.append(third)

        html += f"""
        <div class='card'>
          <div class='card-header'><span class='group-badge'>Group {g}</span></div>
          <div class='card-body'>
            <div class='two-col'>
              <div>
                <h3 class='section-title'>Standings</h3>
                <table>
                  <tr><th>#</th><th>Team</th><th>Pts</th><th>GD</th></tr>
        """
        for idx, row in enumerate(standings):
            qual_class = "qualify" if idx < 2 else ""
            html += f"<tr class='{qual_class}'><td>{idx+1}</td><td>{row['team']}</td><td><b>{row['pts']}</b></td><td>{row['gd']}</td></tr>"

        html += """
                </table>
              </div>
              <div>
                <h3 class='section-title'>Results</h3>
                <table>
                  <tr><th>Home</th><th></th><th>Away</th></tr>
        """
        for t1, o1, o2, t2 in match_rows:
            badge1 = f"<span class='badge badge-{o1}'>{o1}</span>"
            badge2 = f"<span class='badge badge-{o2}'>{o2}</span>"
            html += f"<tr><td class='team-name'>{t1} {badge1}</td><td class='vs'>vs</td><td class='team-name'>{badge2} {t2}</td></tr>"

        html += "</table></div></div></div></div>"

    # best thirds
    thirds.sort(key=lambda x:(x["pts"],x["gd"],x["gf"]), reverse=True)
    best8 = thirds[:8]

    qualified = []
    for g in groups:
        qualified.append(winners[g])
        qualified.append(runners[g])
    for t in best8:
        qualified.append(t["team"])

    np.random.shuffle(qualified)

    current = qualified
    round_names = ["Round of 32","Round of 16","Quarterfinals","Semifinals","Final"]

    for rnd in round_names:
        html += f"""
        <div class='card ko-card'>
          <div class='card-header'><span class='round-badge'>{rnd}</span></div>
          <div class='card-body'>
            <table>
              <tr><th>Team 1</th><th></th><th>Team 2</th><th>Winner</th></tr>
        """
        nxt = []
        for i in range(0, len(current), 2):
            t1 = current[i]
            t2 = current[i+1]
            winner = play_ko_match(t1, t2)
            loser = t2 if winner == t1 else t1
            w1 = "<b>" + t1 + "</b>" if winner == t1 else f"<span class='lost'>{t1}</span>"
            w2 = "<b>" + t2 + "</b>" if winner == t2 else f"<span class='lost'>{t2}</span>"
            html += f"<tr><td>{w1}</td><td class='vs'>vs</td><td>{w2}</td><td class='winner-cell'>🏅 {winner}</td></tr>"
            nxt.append(winner)

        html += "</table></div></div>"
        current = nxt
        if len(current) == 1:
            break

    champion = current[0]
    html = f"<div class='champion'>🏆 {champion}</div>" + html
    return html, champion

# =====================================================
# HTML PAGE
# =====================================================

body, champion = simulate_worldcup()

page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>2026 FIFA World Cup Simulation</title>
<style>
  :root {{
    --bg: #070d1f;
    --surface: #0f1829;
    --surface2: #162038;
    --accent: #c9a84c;
    --accent2: #1a7a3e;
    --text: #e8eaf6;
    --muted: #7986a8;
    --win: #1a7a3e;
    --draw: #5a6a8a;
    --loss: #8a1a1a;
    --qualify: rgba(26,122,62,0.18);
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: 'Segoe UI', Arial, sans-serif;
    padding: 30px 20px;
    min-height: 100vh;
  }}
  h1 {{
    text-align: center;
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: 1px;
    margin-bottom: 8px;
    color: var(--accent);
    text-shadow: 0 0 30px rgba(201,168,76,0.3);
  }}
  .subtitle {{
    text-align: center;
    color: var(--muted);
    margin-bottom: 40px;
    font-size: 0.95rem;
    letter-spacing: 2px;
    text-transform: uppercase;
  }}
  .champion {{
    font-size: 2rem;
    text-align: center;
    background: linear-gradient(135deg, #1a5e30, #1a7a3e);
    padding: 28px;
    border-radius: 16px;
    margin-bottom: 36px;
    font-weight: 800;
    border: 2px solid var(--accent);
    letter-spacing: 1px;
    box-shadow: 0 0 40px rgba(201,168,76,0.2);
  }}
  .section-label {{
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 3px;
    font-size: 0.8rem;
    color: var(--muted);
    margin: 40px 0 16px;
  }}
  .card {{
    background: var(--surface);
    border-radius: 14px;
    margin: 18px 0;
    border: 1px solid #1e2d4a;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  }}
  .ko-card {{ border-color: #2a3a5a; }}
  .card-header {{
    background: var(--surface2);
    padding: 14px 20px;
    border-bottom: 1px solid #1e2d4a;
  }}
  .group-badge {{
    font-size: 1rem;
    font-weight: 700;
    color: var(--accent);
    letter-spacing: 1px;
    text-transform: uppercase;
  }}
  .round-badge {{
    font-size: 1rem;
    font-weight: 700;
    color: #7eb8f7;
    letter-spacing: 1px;
    text-transform: uppercase;
  }}
  .card-body {{ padding: 20px; }}
  .two-col {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }}
  @media(max-width:700px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
  .section-title {{
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: var(--muted);
    margin-bottom: 10px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
  }}
  td, th {{
    padding: 9px 12px;
    text-align: center;
    border-bottom: 1px solid #1a2540;
  }}
  th {{
    background: #111c35;
    color: var(--muted);
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
  }}
  tr:last-child td {{ border-bottom: none; }}
  tr:hover td {{ background: rgba(255,255,255,0.03); }}
  .qualify td {{ background: var(--qualify); }}
  .qualify:hover td {{ background: rgba(26,122,62,0.25); }}
  .team-name {{ text-align: left; }}
  .vs {{ color: var(--muted); font-size: 0.8rem; }}
  .lost {{ color: var(--muted); }}
  .winner-cell {{ color: #7eb8f7; font-weight: 600; }}
  .badge {{
    display: inline-block;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 700;
    margin-left: 4px;
    vertical-align: middle;
  }}
  .badge-W {{ background: var(--win); color: #fff; }}
  .badge-D {{ background: var(--draw); color: #fff; }}
  .badge-L {{ background: var(--loss); color: #fff; }}
  .groups-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(480px, 1fr));
    gap: 18px;
  }}
  .groups-grid .card {{ margin: 0; }}
</style>
</head>
<body>

<h1>⚽ 2026 FIFA World Cup</h1>
<p class="subtitle">Full Tournament Simulation</p>

{body}

</body>
</html>
"""

with open("UIworldcup_simulation.html", "w", encoding="utf-8") as f:
    f.write(page)

print("Saved: worldcup_simulation.html")