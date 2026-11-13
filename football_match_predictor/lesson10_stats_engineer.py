from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# MODEL 1: THE STATS ENGINEER
# Builds pre-match features for every game (played and upcoming), then predicts
# the match stats: goals, shots on target, corners and cards for each team.
# Output: data/processed/match_stats.csv, which Model 2 (the analyst) reads.

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "match_stats.csv"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

FORM_GAMES = 10  # tested 5 vs 10: 10 is more stable for corners and cards
MIN_FORM_GAMES = 5  # promoted teams can be used once they have 5 games
MAX_REST_DAYS = 14

INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 80
SHOTS_WEIGHT = 0.3  # Elo = 70% result + 30% shots on target share (lesson 7)

# A referee with few games is pulled toward the league average,
# as if he also had this many "average" games.
REFEREE_PRIOR_GAMES = 10

BACKTEST_SEASONS = ["2025-2026", "2026-2027"]

# Per-team numbers kept for the last FORM_GAMES games
TEAM_STATS = [
    "points", "goals_for", "goals_against",
    "shots_for", "shots_against",
    "corners_for", "corners_against",
    "cards_for", "cards_against",
    "fouls_for", "fouls_against",
]

FEATURES = (
    [f"home_form_{stat}" for stat in TEAM_STATS]
    + [f"away_form_{stat}" for stat in TEAM_STATS]
    + [
        "elo_difference",
        "home_rest_days", "away_rest_days",
        "referee_cards",
        "odds_home_probability", "odds_away_probability",
        "odds_over_2_5_probability", "asian_handicap_line",
    ]
)

# Weather at the home stadium on match day (from fetch_weather.py, Open-Meteo)
WEATHER_PATH = BASE_DIR / "data" / "external" / "weather_daily.csv"
WEATHER_FEATURES = ["weather_temperature", "weather_rain_mm", "weather_wind_kmh"]
# Tested: adding weather made every prediction 0.1-1% worse (it only adds noise),
# so the model does not use it. It is still saved for Model 2's assessment.
USE_WEATHER = False

if USE_WEATHER:
    FEATURES = FEATURES + WEATHER_FEATURES

# What Model 1 predicts: stat name -> (home column, away column)
TARGETS = {
    "goals": ("home_goals", "away_goals"),
    "shots_on_target": ("home_shots_on_target", "away_shots_on_target"),
    "corners": ("home_corners", "away_corners"),
    "cards": ("home_cards", "away_cards"),
}

fixture_columns = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "Referee": "referee",
    "AvgH": "avg_home_odds",
    "AvgD": "avg_draw_odds",
    "AvgA": "avg_away_odds",
    "Avg>2.5": "avg_over_2_5_odds",
    "Avg<2.5": "avg_under_2_5_odds",
    "AHh": "asian_handicap_line",
    "AvgAHH": "avg_asian_handicap_home_odds",
    "AvgAHA": "avg_asian_handicap_away_odds",
}


# ---------------------------------------------------------------------------
# LOAD DATA: played games + upcoming fixtures
# ---------------------------------------------------------------------------

def load_upcoming_fixtures(played):
    try:
        fixtures = pd.read_csv(FIXTURES_URL, encoding="utf-8-sig")
    except Exception as error:
        print(f"Could not download fixtures: {error}")
        return pd.DataFrame()

    fixtures = fixtures[fixtures["Div"] == "E0"]
    fixtures = fixtures.rename(columns=fixture_columns)[list(fixture_columns.values())]
    fixtures["date"] = pd.to_datetime(fixtures["date"], dayfirst=True)

    # Skip games that are already in the results data
    key = ["date", "home_team", "away_team"]
    already_played = fixtures.merge(
        played[key], on=key, how="left", indicator=True
    )["_merge"] == "both"
    fixtures = fixtures[~already_played.values]

    fixtures["season"] = played["season"].iloc[-1]
    return fixtures


played = pd.read_csv(
    PROCESSED_DIR / "premier_league_matches.csv",
    parse_dates=["date"]
)
played["home_cards"] = played["home_yellow_cards"] + played["home_red_cards"]
played["away_cards"] = played["away_yellow_cards"] + played["away_red_cards"]

fixtures = load_upcoming_fixtures(played)

games = (
    pd.concat([played, fixtures], ignore_index=True)
    .sort_values("date", kind="stable")
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------------
# FEATURES: everything is calculated from games BEFORE each match
# ---------------------------------------------------------------------------

def build_features(games):
    recent = defaultdict(lambda: deque(maxlen=FORM_GAMES))
    elo = {}
    last_played = {}
    referee_cards = defaultdict(lambda: [0.0, 0])  # referee -> [total cards, games]
    league_cards = [0.0, 0]
    rows = []

    for game in games.itertuples(index=False):
        home, away = game.home_team, game.away_team
        row = {}

        # Team form: average of the last FORM_GAMES games (at least MIN_FORM_GAMES)
        for side, team in [("home", home), ("away", away)]:
            if len(recent[team]) >= MIN_FORM_GAMES:
                averages = np.mean(recent[team], axis=0)
            else:
                averages = [np.nan] * len(TEAM_STATS)
            for stat, value in zip(TEAM_STATS, averages):
                row[f"{side}_form_{stat}"] = value

            if team in last_played:
                rest = (game.date - last_played[team]).days
                row[f"{side}_rest_days"] = min(rest, MAX_REST_DAYS)
            else:
                row[f"{side}_rest_days"] = MAX_REST_DAYS

        # Elo
        home_elo = elo.get(home, INITIAL_ELO)
        away_elo = elo.get(away, INITIAL_ELO)
        row["home_elo"] = home_elo
        row["away_elo"] = away_elo
        row["elo_difference"] = home_elo - away_elo

        # Referee: average cards per game, pulled toward the league average
        league_average = league_cards[0] / league_cards[1] if league_cards[1] else 4.0
        total, count = referee_cards[game.referee] if pd.notna(game.referee) else (0.0, 0)
        row["referee_cards"] = (
            (total + REFEREE_PRIOR_GAMES * league_average)
            / (count + REFEREE_PRIOR_GAMES)
        )
        row["referee_games"] = count

        rows.append(row)

        # Upcoming game: no stats yet, nothing to learn from it
        if pd.isna(game.home_goals):
            continue

        home_goals, away_goals = game.home_goals, game.away_goals
        home_shots, away_shots = game.home_shots_on_target, game.away_shots_on_target

        expected_home = 1 / (1 + 10 ** ((away_elo - home_elo - HOME_ADVANTAGE) / 400))
        actual_home = 1.0 if home_goals > away_goals else 0.5 if home_goals == away_goals else 0.0
        total_shots = home_shots + away_shots
        shots_share = home_shots / total_shots if total_shots else 0.5
        score = (1 - SHOTS_WEIGHT) * actual_home + SHOTS_WEIGHT * shots_share
        change = K_FACTOR * (score - expected_home)
        elo[home] = home_elo + change
        elo[away] = away_elo - change

        home_points = 3 if home_goals > away_goals else 1 if home_goals == away_goals else 0
        away_points = 3 if away_goals > home_goals else 1 if home_goals == away_goals else 0

        # Same order as TEAM_STATS
        recent[home].append((
            home_points, home_goals, away_goals, home_shots, away_shots,
            game.home_corners, game.away_corners, game.home_cards, game.away_cards,
            game.home_fouls, game.away_fouls,
        ))
        recent[away].append((
            away_points, away_goals, home_goals, away_shots, home_shots,
            game.away_corners, game.home_corners, game.away_cards, game.home_cards,
            game.away_fouls, game.home_fouls,
        ))

        last_played[home] = game.date
        last_played[away] = game.date

        match_cards = game.home_cards + game.away_cards
        if pd.notna(game.referee):
            referee_cards[game.referee][0] += match_cards
            referee_cards[game.referee][1] += 1
        league_cards[0] += match_cards
        league_cards[1] += 1

    return pd.concat([games, pd.DataFrame(rows)], axis=1)


def add_odds_probabilities(data):
    # 1 / odds, then remove the bookmaker margin so chances add up to 100%
    inverse_home = 1 / data["avg_home_odds"]
    inverse_draw = 1 / data["avg_draw_odds"]
    inverse_away = 1 / data["avg_away_odds"]
    inverse_total = inverse_home + inverse_draw + inverse_away
    data["odds_home_probability"] = inverse_home / inverse_total
    data["odds_draw_probability"] = inverse_draw / inverse_total
    data["odds_away_probability"] = inverse_away / inverse_total

    inverse_over = 1 / data["avg_over_2_5_odds"]
    inverse_under = 1 / data["avg_under_2_5_odds"]
    data["odds_over_2_5_probability"] = inverse_over / (inverse_over + inverse_under)
    return data


def add_weather(data):
    if not WEATHER_PATH.exists():
        if USE_WEATHER:
            raise FileNotFoundError(f"{WEATHER_PATH} not found - run fetch_weather.py first.")
        print("No weather file - run fetch_weather.py to add weather to the assessment.")
        return data

    weather = pd.read_csv(WEATHER_PATH, parse_dates=["date"]).rename(columns={
        "team": "home_team",
        "temperature": "weather_temperature",
        "rain_mm": "weather_rain_mm",
        "wind_kmh": "weather_wind_kmh",
    })
    data = data.merge(weather, on=["home_team", "date"], how="left")

    # A game too far ahead for the forecast: use the average for that month
    missing = data[WEATHER_FEATURES].isna().any(axis=1)
    if missing.any():
        month_average = data.groupby(data["date"].dt.month)[WEATHER_FEATURES].transform("mean")
        data[WEATHER_FEATURES] = data[WEATHER_FEATURES].fillna(month_average)
        print(f"Weather missing for {missing.sum()} games - used the monthly average.")
    return data


games = add_odds_probabilities(build_features(games))
games = add_weather(games)

# Drop games where a team has fewer than MIN_FORM_GAMES previous games, or odds are missing
games = games.dropna(subset=FEATURES).reset_index(drop=True)

played_games = games[games["home_goals"].notna()].reset_index(drop=True)
upcoming_games = games[games["home_goals"].isna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# MODELS: one Poisson model per stat and per team side
# ---------------------------------------------------------------------------

def train_stat_models(train):
    models = {}
    for home_column, away_column in TARGETS.values():
        for column in [home_column, away_column]:
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("model", PoissonRegressor(alpha=1.0, max_iter=1000)),
            ])
            model.fit(train[FEATURES], train[column])
            models[column] = model
    return models


def predict_stats(models, data):
    data = data.copy()
    for column, model in models.items():
        data[f"exp_{column}"] = model.predict(data[FEATURES])
    return data


# Backtest: each season predicted by models trained only on earlier seasons
tested = []
for season in BACKTEST_SEASONS:
    train = played_games[played_games["season"] < season]
    test = played_games[played_games["season"] == season]
    tested.append(predict_stats(train_stat_models(train), test))
tested = pd.concat(tested, ignore_index=True)
tested["set"] = "backtest"

# Upcoming games: models trained on everything played so far
if upcoming_games.empty:
    upcoming = upcoming_games
else:
    upcoming = predict_stats(train_stat_models(played_games), upcoming_games)
    upcoming["set"] = "upcoming"


# ---------------------------------------------------------------------------
# HOW ACCURATE IS MODEL 1? Compare with just guessing the league average
# ---------------------------------------------------------------------------

# Poisson deviance = the fair error measure for "expected count" predictions.
# Baseline = the average from the seasons before each test season.

def improvement(actual, predicted, columns):
    baseline_values = {
        season: played_games.loc[played_games["season"] < season, columns].sum(axis=1).mean()
        for season in BACKTEST_SEASONS
    }
    baseline = tested["season"].map(baseline_values)
    model_error = mean_poisson_deviance(actual, predicted)
    baseline_error = mean_poisson_deviance(actual, baseline)
    return f"{(baseline_error - model_error) / baseline_error * 100:+.1f}%"


accuracy_rows = []
for stat, (home_column, away_column) in TARGETS.items():
    accuracy_rows.append({
        "Stat": stat.replace("_", " ").title(),
        "Avg per match": round((tested[home_column] + tested[away_column]).mean(), 2),
        "Home team": improvement(
            tested[home_column], tested[f"exp_{home_column}"], [home_column]
        ),
        "Away team": improvement(
            tested[away_column], tested[f"exp_{away_column}"], [away_column]
        ),
        "Match total": improvement(
            tested[home_column] + tested[away_column],
            tested[f"exp_{home_column}"] + tested[f"exp_{away_column}"],
            [home_column, away_column],
        ),
    })

print("=" * 78)
print("MODEL 1 - STATS ENGINEER")
print(f"Backtest: {len(tested)} matches, {tested['date'].min():%b %Y} to {tested['date'].max():%b %Y}")
print("=" * 78)
print("How much better than guessing the league average (higher = better):")
print(pd.DataFrame(accuracy_rows).to_string(index=False))
print("Each team's own numbers are easier to predict than match totals.")

APP_DIR = PROCESSED_DIR / "app"  # results for the website (app.py)
APP_DIR.mkdir(parents=True, exist_ok=True)
pd.DataFrame(accuracy_rows).to_csv(APP_DIR / "stats_accuracy.csv", index=False)


# ---------------------------------------------------------------------------
# SAVE for Model 2
# ---------------------------------------------------------------------------

output_columns = (
    ["set", "date", "season", "home_team", "away_team", "referee", "referee_games"]
    + ["home_elo", "away_elo", "elo_difference", "home_rest_days", "away_rest_days", "referee_cards"]
    + ["home_form_points", "away_form_points"]
    + [column for column in WEATHER_FEATURES if column in games.columns]
    + list(fixture_columns.values())[4:]  # bookmaker odds and handicap line
    + ["odds_home_probability", "odds_draw_probability", "odds_away_probability",
       "odds_over_2_5_probability"]
    + [f"exp_{column}" for pair in TARGETS.values() for column in pair]
    + [column for pair in TARGETS.values() for column in pair]  # actual stats
    + ["result"]
)

output = pd.concat([tested, upcoming], ignore_index=True)[output_columns]
output.to_csv(OUTPUT_PATH, index=False)

print(f"\nUpcoming matches predicted: {len(upcoming)}")
print(f"Saved {len(output)} rows to {OUTPUT_PATH}")
