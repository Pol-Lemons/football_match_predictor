from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# One model for everything:
#   features = form + shots on target + Elo + rest days + bookmaker odds
#   model    = predict goals for each team (Poisson)
#   output   = chance of every score -> 1X2, over/under 2.5, Asian handicap

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

FORM_GAMES = 5
MAX_REST_DAYS = 14

INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 80
SHOTS_WEIGHT = 0.3  # lesson 7 test: Elo = 70% result + 30% shots on target share

MAX_GOALS = 10
MIN_EDGE = 0.05  # only call it a value bet if expected profit is 5% or more
BACKTEST_SEASONS = ["2025-2026", "2026-2027"]

FEATURES = [
    "home_form_points", "home_form_goals_for", "home_form_goals_against",
    "home_form_shots_for", "home_form_shots_against",
    "away_form_points", "away_form_goals_for", "away_form_goals_against",
    "away_form_shots_for", "away_form_shots_against",
    "elo_difference",
    "home_rest_days", "away_rest_days",
    "odds_home_probability", "odds_away_probability",
    "odds_over_2_5_probability", "asian_handicap_line",
]

fixture_columns = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
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
# LOAD DATA
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
    # Last 5 games per team: (points, goals for, goals against, shots on target for, against)
    recent = defaultdict(lambda: deque(maxlen=FORM_GAMES))
    elo = {}
    last_played = {}
    rows = []

    for game in games.itertuples(index=False):
        home, away = game.home_team, game.away_team
        row = {}

        for side, team in [("home", home), ("away", away)]:
            if len(recent[team]) == FORM_GAMES:
                points, goals_for, goals_against, shots_for, shots_against = (
                    np.mean(recent[team], axis=0)
                )
            else:
                points = goals_for = goals_against = shots_for = shots_against = np.nan

            row[f"{side}_form_points"] = points
            row[f"{side}_form_goals_for"] = goals_for
            row[f"{side}_form_goals_against"] = goals_against
            row[f"{side}_form_shots_for"] = shots_for
            row[f"{side}_form_shots_against"] = shots_against

            if team in last_played:
                rest = (game.date - last_played[team]).days
                row[f"{side}_rest_days"] = min(rest, MAX_REST_DAYS)
            else:
                row[f"{side}_rest_days"] = MAX_REST_DAYS

        home_elo = elo.get(home, INITIAL_ELO)
        away_elo = elo.get(away, INITIAL_ELO)
        row["home_elo"] = home_elo
        row["away_elo"] = away_elo
        row["elo_difference"] = home_elo - away_elo
        rows.append(row)

        # Upcoming game: no score yet, nothing to learn from it
        if pd.isna(game.home_goals):
            continue

        home_goals, away_goals = game.home_goals, game.away_goals
        home_shots, away_shots = game.home_shots_on_target, game.away_shots_on_target

        # Elo update (lesson 7), with "played well" credit from shots on target
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
        recent[home].append((home_points, home_goals, away_goals, home_shots, away_shots))
        recent[away].append((away_points, away_goals, home_goals, away_shots, home_shots))

        last_played[home] = game.date
        last_played[away] = game.date

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


games = add_odds_probabilities(build_features(games))

# Drop games where a team has fewer than 5 previous games, or odds are missing
games = games.dropna(subset=FEATURES).reset_index(drop=True)

played_games = games[games["home_goals"].notna()].reset_index(drop=True)
upcoming_games = games[games["home_goals"].isna()].reset_index(drop=True)


# ---------------------------------------------------------------------------
# MODEL: expected goals for each team -> chance of every score
# ---------------------------------------------------------------------------

def train_goal_models(train):
    models = {}
    for target in ["home_goals", "away_goals"]:
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", PoissonRegressor(alpha=0.01, max_iter=1000)),
        ])
        model.fit(train[FEATURES], train[target])
        models[target] = model
    return models


def predict_expected_goals(models, data):
    data = data.copy()
    data["home_xg"] = models["home_goals"].predict(data[FEATURES])
    data["away_xg"] = models["away_goals"].predict(data[FEATURES])
    return data


goal_range = np.arange(MAX_GOALS + 1)

# Grid of every score: HOME[i, j] = i, AWAY[i, j] = j
HOME, AWAY = np.meshgrid(goal_range, goal_range, indexing="ij")
MARGIN = HOME - AWAY
TOTAL = HOME + AWAY


def score_matrix(home_xg, away_xg):
    # chance[i, j] = chance the score ends i - j
    chance = np.outer(poisson.pmf(goal_range, home_xg), poisson.pmf(goal_range, away_xg))
    return chance / chance.sum()


def handicap_payoff(margin, line, odds):
    # What 1 unit returns for each score. margin = goals for - goals against.
    # Quarter lines (-0.25, -0.75, ...) split the stake over the two nearest lines,
    # e.g. -0.75 = half on -0.5 and half on -1.
    if round(line * 4) % 2 == 1:
        lines = [line - 0.25, line + 0.25]
    else:
        lines = [line]

    payoff = 0
    for part in lines:
        adjusted = margin + part
        payoff = payoff + np.where(adjusted > 0, odds, np.where(adjusted == 0, 1.0, 0.0))
    return payoff / len(lines)


def list_bets(data):
    # One row for every bet in every match, with model chance and edge
    rows = []

    for game in data.itertuples(index=False):
        chance = score_matrix(game.home_xg, game.away_xg)
        home, away, line = game.home_team, game.away_team, game.asian_handicap_line

        # (market, pick, what 1 unit returns for each score)
        markets = [
            ("1X2", f"{home} win", np.where(MARGIN > 0, game.avg_home_odds, 0.0)),
            ("1X2", "Draw", np.where(MARGIN == 0, game.avg_draw_odds, 0.0)),
            ("1X2", f"{away} win", np.where(MARGIN < 0, game.avg_away_odds, 0.0)),
            ("Over/Under", "Over 2.5", np.where(TOTAL > 2, game.avg_over_2_5_odds, 0.0)),
            ("Over/Under", "Under 2.5", np.where(TOTAL < 3, game.avg_under_2_5_odds, 0.0)),
        ]
        if pd.notna(game.avg_asian_handicap_home_odds):
            markets += [
                ("Handicap", f"{home} {line:+g}",
                 handicap_payoff(MARGIN, line, game.avg_asian_handicap_home_odds)),
                ("Handicap", f"{away} {-line:+g}",
                 handicap_payoff(-MARGIN, -line, game.avg_asian_handicap_away_odds)),
            ]

        for market, pick, payoff in markets:
            expected_return = (chance * payoff).sum()
            odds = payoff.max()

            row = {
                "date": game.date,
                "match": f"{home} vs {away}",
                "market": market,
                "pick": pick,
                "odds": odds,
                "model_chance": chance[payoff > 1].sum(),
                "edge": expected_return - 1,
            }

            if pd.notna(game.home_goals):
                home_goals = min(int(game.home_goals), MAX_GOALS)
                away_goals = min(int(game.away_goals), MAX_GOALS)
                row["profit"] = payoff[home_goals, away_goals] - 1

            rows.append(row)

    return pd.DataFrame(rows)


def outcome_probabilities(data):
    # Model chances for home / draw / away and over 2.5, per match
    rows = []
    for game in data.itertuples(index=False):
        chance = score_matrix(game.home_xg, game.away_xg)
        best_score = np.unravel_index(chance.argmax(), chance.shape)
        rows.append({
            "H": chance[MARGIN > 0].sum(),
            "D": chance[MARGIN == 0].sum(),
            "A": chance[MARGIN < 0].sum(),
            "over_2_5": chance[TOTAL > 2].sum(),
            "likely_score": f"{best_score[0]}-{best_score[1]}",
        })
    return pd.DataFrame(rows, index=data.index)


def percent(value):
    return f"{value * 100:.0f}%"


# ---------------------------------------------------------------------------
# 1. BACKTEST: train on earlier seasons only, then test the next season
# ---------------------------------------------------------------------------

tested = []
for season in BACKTEST_SEASONS:
    train = played_games[played_games["season"] < season]
    test = played_games[played_games["season"] == season]
    models = train_goal_models(train)
    tested.append(predict_expected_goals(models, test))

tested = pd.concat(tested, ignore_index=True)
probabilities = outcome_probabilities(tested)

# Are the model's chances better than the bookmaker's? (lower = better)
over_actual = (tested["home_goals"] + tested["away_goals"] > 2).astype(int)
quality = pd.DataFrame([
    {
        "Market": "1X2",
        "Model": log_loss(tested["result"], probabilities[["A", "D", "H"]], labels=["A", "D", "H"]),
        "Bookmaker": log_loss(
            tested["result"],
            tested[["odds_away_probability", "odds_draw_probability", "odds_home_probability"]],
            labels=["A", "D", "H"],
        ),
    },
    {
        "Market": "Over/Under 2.5",
        "Model": log_loss(over_actual, probabilities["over_2_5"], labels=[0, 1]),
        "Bookmaker": log_loss(over_actual, tested["odds_over_2_5_probability"], labels=[0, 1]),
    },
])

print("=" * 78)
print(f"BACKTEST: {len(tested)} matches, {tested['date'].min():%b %Y} to {tested['date'].max():%b %Y}")
print("(each season predicted by a model trained only on earlier seasons)")
print("=" * 78)

print("\nHow good are the chances? (Log Loss, lower = better)")
print(quality.round(4).to_string(index=False))

backtest_bets = list_bets(tested)
value_bets = backtest_bets[backtest_bets["edge"] >= MIN_EDGE]

best_per_match = (
    value_bets.sort_values("edge", ascending=False)
    .drop_duplicates(["date", "match"])
)


def summarise(name, bets):
    count = max(len(bets), 1)
    return {
        "Strategy": name,
        "Bets": len(bets),
        "Win %": percent((bets["profit"] > 0).sum() / count),
        "Avg odds": round(bets["odds"].mean(), 2) if len(bets) else 0,
        "Profit (units)": round(bets["profit"].sum(), 1),
        "ROI": f"{bets['profit'].sum() / count * 100:+.1f}%",
    }


strategies = [
    summarise(f"Value bets ({market})", value_bets[value_bets["market"] == market])
    for market in ["1X2", "Over/Under", "Handicap"]
]
strategies += [
    summarise("All value bets", value_bets),
    summarise("Best value bet per match", best_per_match),
]

print(f"\nBetting 1 unit on every bet with edge >= {MIN_EDGE:.0%}:")
print(pd.DataFrame(strategies).to_string(index=False))
print("ROI = profit per 100 units staked. Below 0 means you lose money.")


# ---------------------------------------------------------------------------
# 2. PREVIEW: upcoming fixtures (or replay the latest 10 games if none)
# ---------------------------------------------------------------------------

replay_mode = upcoming_games.empty
preview = played_games.tail(10) if replay_mode else upcoming_games

models = train_goal_models(played_games[played_games["date"] < preview["date"].min()])
preview = predict_expected_goals(models, preview).reset_index(drop=True)
preview_chances = outcome_probabilities(preview)
preview_bets = list_bets(preview)

print("\n" + "=" * 78)
if replay_mode:
    print("No upcoming fixtures with odds yet. REPLAY of the latest 10 games:")
else:
    print("PREVIEW: upcoming Premier League fixtures")
print("=" * 78)

preview_rows = []
for index, game in preview.iterrows():
    match = f"{game['home_team']} vs {game['away_team']}"
    chances = preview_chances.loc[index]
    bets = preview_bets[preview_bets["match"] == match].sort_values("edge", ascending=False)
    best = bets.iloc[0]

    row = {
        "Date": f"{game['date']:%a %d %b}",
        "Match": match,
        "Exp. goals": f"{game['home_xg']:.1f} - {game['away_xg']:.1f}",
        "Home": percent(chances["H"]),
        "Draw": percent(chances["D"]),
        "Away": percent(chances["A"]),
        "Over 2.5": percent(chances["over_2_5"]),
        "Likely score": chances["likely_score"],
        "Best bet": best["pick"] if best["edge"] >= MIN_EDGE else "-",
        "Odds": f"{best['odds']:.2f}" if best["edge"] >= MIN_EDGE else "",
        "Edge": f"{best['edge']:+.0%}" if best["edge"] >= MIN_EDGE else "",
    }
    if replay_mode:
        row["Result"] = f"{game['home_goals']:.0f}-{game['away_goals']:.0f}"
    preview_rows.append(row)

print(pd.DataFrame(preview_rows).to_string(index=False))

print(f"\nAll bets with edge >= {MIN_EDGE:.0%}, best first:")
preview_value = preview_bets[preview_bets["edge"] >= MIN_EDGE].sort_values("edge", ascending=False)
if preview_value.empty:
    print("None - the model agrees with the bookmaker on every game.")
else:
    table = preview_value[["match", "market", "pick", "odds", "model_chance", "edge"]].copy()
    table["odds"] = table["odds"].map("{:.2f}".format)
    table["model_chance"] = table["model_chance"].map(percent)
    table["edge"] = table["edge"].map("{:+.0%}".format)
    print(table.to_string(index=False))
