# ⚽ 2026 FIFA World Cup Predictor

Machine Learning based project to predict football match outcomes and simulate the full 2026 FIFA World Cup tournament using historical international football data.

---

## 🚀 Project Features

✅ Predict Match Results (Home Win / Draw / Away Win)

✅ Dynamic Elo Rating System

✅ Recent Team Form Features

✅ CatBoost ML Model

✅ Separate Models for:

- Group Stage (draw-sensitive)
- Knockout Stage (winner-focused)

✅ Full 2026 FIFA World Cup Simulation

- 48 Teams
- 12 Groups
- Round of 32
- Final Champion Prediction

✅ HTML Visual Tournament Output

---

## 📊 Final Model Accuracy

### Knockout Model

Accuracy: **60.23%**

### Group Stage Model

Accuracy: **59.76%**

Better draw handling for standings realism.

---

## 🧠 Machine Learning Features Used

- home_team
- away_team
- neutral venue
- recent form
- Elo rating
- Elo difference
- close match indicator

---

## 📂 Dataset Used

Historical International Football Matches + FIFA World Cup Matches

Final cleaned dataset:

**32,000+ matches**

---

## ⚙️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- CatBoost
- XGBoost
- Joblib
- HTML/CSS

---

## 📁 Project Structure

```bash
Final/
│── data/
│   ├── cleaned_matches_1990.csv
│   ├── features_with_elo.csv
│
│── clean_data.py
│── feature_engineering.py
│── elo_features.py
│── train_advanced.py
│── traindraw.py
│── Tournament_Winning_Chance.py
│── Tournament_Simulation.py

advanced_model.pkl
catboost_draw_model.pkl
worldcup_simulation.html
README.md
requirements.txt
.gitignore
```

---

## ▶️ How To Run

### Install requirements

```bash
pip install -r requirements.txt
```

### Train Models

```bash
python Final/train_advanced.py
python Final/traindraw.py
```

### Run Tournament Simulation

```bash
python Final/Tournament_Simulation.py  //For tournament simulation
```
python Final/Tournament_Simulation.py  //For tournament winning probabilities

### Open HTML Output

```text
Example_Predicition_Run.html
```

---

## 🏆 Sample Output

- Group Stage Results
- Knockout Bracket
- Final Winner
- Champion Probabilities

---

## 🔥 Future Improvements

- Poisson Goal Prediction
- Live FIFA Rankings API
- Player Injury Data
- Streamlit Web App
- Better Bracket Visualization

---

## 👨‍💻 Author

Sai Gunjal