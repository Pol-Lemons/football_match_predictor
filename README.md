# Premier League Match Predictor

A machine-learning system that predicts Premier League matches, from goals, corners and cards to who scores, and shows the predictions in a sports-data web app.

It is built as a two-stage pipeline: a **stats engineer** model turns raw data into expected match statistics, and an **analyst** model turns those statistics into a written assessment, probabilities for every betting market, a confidence level and a final pick. Every prediction is backtested on matches the models never saw.

## Features

- **Match predictions**: home / draw / away chances, most likely scores, over/under, both teams to score, odd/even goals, Asian handicap, corners and cards, each with fair odds
- **Goalscorer model**: chance of each player scoring, built from player xG per 90 minutes, expected minutes and injury news
- **Written assessment**: strength (Elo), form, game flow, referee strictness, missing players and rest days, in plain sentences
- **Confidence levels**: the model says how sure it is, and switches to a safer double chance when a match is too close
- **Value bets**: markets where the model's price beats the bookmaker's, with a bet and parlay calculator
- **Web app** (Streamlit): fixture list, match pages with head-to-head history since 2000/01, real match stats, player ratings, model performance and past rounds

## Results (backtest)

465 matches from August 2025 to November 2026. Each season was predicted by models trained only on earlier seasons.

| Prediction | Right |
|---|---|
| Main pick, all matches | **67%** |
| Main pick, high confidence (one result 60%+) | **77%** |
| Over/under 2.5 goals | 54% |
| Both teams to score | 56% |

| Confidence | Share of matches | Main pick | Right |
|---|---|---|---|
| High (60%+) | 15% | Match result | 77% |
| Medium (50%+) | 29% | Match result | 55% |
| Low (<50%) | 57% | Double chance | 71% |

The goalscorer model is well calibrated: players given a 10-20% chance scored 14% of the time, players given 20-30% scored 25%.

Value bets (edge of 5%+, odds up to 5.0) returned +9.5% on match results (167 bets) and +11.8% on Asian handicaps (134 bets). The samples are still small.

## How it works

```
Football-Data.co.uk ─┐                     ┌─ match_stats.csv ─┐
FPL (players, injuries) ─┼─> Model 1: stats engineer ─┤                    ├─> Model 2: analyst ─> web app
Open-Meteo (weather) ─┘   (Poisson regressions)  └─ player_predictions.csv ┘   (markets, picks)
```

1. **Features** (`lesson10_stats_engineer.py`): form over the last 10 matches (points, goals, shots on target, corners, cards, fouls), Elo ratings that also reward shots on target, rest days, referee card rate and bookmaker odds, all computed only from matches before kick-off.
2. **Model 1**: one Poisson regression per stat and side predicts expected goals, shots on target, corners and cards.
3. **Goalscorers** (`lesson12_goalscorers.py`): splits each team's expected goals between its players by xG per 90 and expected minutes.
4. **Model 2** (`lesson11_match_analyst.py`): turns expected goals into the chance of every score, then into every market, the assessment and the pick.

The `lesson*.py` files show how the project grew step by step, from a baseline classifier to the full pipeline.

## Run it

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# Download the latest results, players and weather, then run all models
.venv/bin/python run_predictions.py --update

# Open the web app
.venv/bin/streamlit run app.py
```

`python run_predictions.py "Man United"` shows every market for one team in the terminal. `python test_app.py` checks that every page of the app loads.

## Project structure

| Path | What it does |
|---|---|
| `run_predictions.py` | Runs the whole pipeline with one command |
| `lesson10_stats_engineer.py` | Model 1: features and expected match stats |
| `lesson12_goalscorers.py` | Goalscorer model |
| `lesson11_match_analyst.py` | Model 2: assessment, markets, picks, backtest |
| `fetch_*.py` | Data downloads (FPL, weather, head-to-head history) |
| `app.py`, `pages/`, `app_data.py` | Streamlit web app |
| `data/processed/app/` | Model outputs the web app reads |

## Data and credits

- Match results, stats and odds: [Football-Data.co.uk](https://www.football-data.co.uk/)
- Players, xG, minutes and injuries: [Fantasy Premier League](https://fantasy.premierleague.com/) (unofficial API)
- Weather data by [Open-Meteo.com](https://open-meteo.com/) ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/))
- Club badges: [Premier League](https://www.premierleague.com/), property of the clubs
- Animations: [GSAP](https://gsap.com/), smooth scrolling: [Lenis](https://github.com/darkroomengineering/lenis) (MIT)
