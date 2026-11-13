from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

FIXTURES_URL = "https://www.football-data.co.uk/fixtures.csv"

# Keo "co gia tri" (value bet): model chance x odds phai lon hon 1 it nhat 5%
MIN_EDGE = 0.05

# Mua dau tien dung de kiem tra lai chien thuat (backtest)
BACKTEST_FROM_SEASON = "2025-2026"

form_features = [
    "home_form_points_5",
    "home_form_goals_for_5",
    "home_form_goals_against_5",
    "away_form_points_5",
    "away_form_goals_for_5",
    "away_form_goals_against_5",
]

odds_features = [
    "odds_home_probability",
    "odds_draw_probability",
    "odds_away_probability",
    "asian_handicap_line",
    "odds_over_2_5_probability",
]

# Mo hinh 1X2 (thang / hoa / thua)
result_features = form_features + odds_features

# Mo hinh tai xiu 2.5
goals_features = [
    "home_form_goals_for_5",
    "home_form_goals_against_5",
    "away_form_goals_for_5",
    "away_form_goals_against_5",
    "odds_over_2_5_probability",
]

# Same names as lesson 1, for reading the fixtures file
fixture_columns = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "AvgH": "avg_home_odds",
    "AvgD": "avg_draw_odds",
    "AvgA": "avg_away_odds",
    "AHh": "asian_handicap_line",
    "Avg>2.5": "avg_over_2_5_odds",
    "Avg<2.5": "avg_under_2_5_odds",
}


# ---------------------------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------------------------

matches = pd.read_csv(
    PROCESSED_DIR / "premier_league_matches.csv",
    parse_dates=["date"]
).sort_values("date").reset_index(drop=True)

feature_data = pd.read_csv(
    PROCESSED_DIR / "premier_league_form_features.csv",
    parse_dates=["date"]
).sort_values("date").reset_index(drop=True)

feature_data["over_2_5"] = (
    feature_data["home_goals"] + feature_data["away_goals"] > 2
).astype(int)


# MOI TRAN THANH 2 DONG: 1 DONG CHO DOI NHA, 1 DONG CHO DOI KHACH

home_rows = pd.DataFrame({
    "date": matches["date"],
    "team": matches["home_team"],
    "opponent": matches["away_team"],
    "venue": "Home",
    "goals_for": matches["home_goals"],
    "goals_against": matches["away_goals"],
    "outcome": matches["result"].map({"H": "W", "D": "D", "A": "L"}),
})

away_rows = pd.DataFrame({
    "date": matches["date"],
    "team": matches["away_team"],
    "opponent": matches["home_team"],
    "venue": "Away",
    "goals_for": matches["away_goals"],
    "goals_against": matches["home_goals"],
    "outcome": matches["result"].map({"A": "W", "D": "D", "H": "L"}),
})

team_history = pd.concat(
    [home_rows, away_rows],
    ignore_index=True
).sort_values("date").reset_index(drop=True)

team_history["points"] = team_history["outcome"].map({"W": 3, "D": 1, "L": 0})


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def add_odds_probabilities(data):
    # Same formula as lesson 3: 1 / odds, then remove the bookmaker margin
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


def team_form(team, before_date, venue=None, games=5):
    # Last 5 matches of a team, only matches played BEFORE this date
    rows = team_history[
        (team_history["team"] == team)
        & (team_history["date"] < before_date)
    ]
    if venue:
        rows = rows[rows["venue"] == venue]
    rows = rows.tail(games)

    if len(rows) < games:
        return None

    return {
        "last_5": " ".join(rows["outcome"]),
        "points": rows["points"].mean(),
        "goals_for": rows["goals_for"].mean(),
        "goals_against": rows["goals_against"].mean(),
        "over_2_5": ((rows["goals_for"] + rows["goals_against"]) > 2).mean(),
        "clean_sheets": (rows["goals_against"] == 0).mean(),
    }


def head_to_head(home_team, away_team, before_date, games=5):
    rows = team_history[
        (team_history["team"] == home_team)
        & (team_history["opponent"] == away_team)
        & (team_history["date"] < before_date)
    ].tail(games)

    if rows.empty:
        return "No meetings in the data"

    wins = (rows["outcome"] == "W").sum()
    draws = (rows["outcome"] == "D").sum()
    losses = (rows["outcome"] == "L").sum()
    goals = (rows["goals_for"] + rows["goals_against"]).mean()

    return (
        f"{home_team} {wins}W {draws}D {losses}L "
        f"in the last {len(rows)}, {goals:.1f} goals per game"
    )


def train_model(data, feature_columns, target_column):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(data[feature_columns], data[target_column])
    return model


def list_bets(data, result_model, goals_model):
    # One row for every possible bet: home, draw, away, over 2.5, under 2.5
    result_probability = pd.DataFrame(
        result_model.predict_proba(data[result_features]),
        columns=result_model.classes_,
        index=data.index
    )
    over_probability = pd.Series(
        goals_model.predict_proba(data[goals_features])[:, 1],
        index=data.index
    )

    total_goals = data["home_goals"] + data["away_goals"]
    played = data["home_goals"].notna()

    markets = [
        # (market, pick name, model chance, odds column, bet won?)
        ("1X2", data["home_team"] + " win", result_probability["H"],
         "avg_home_odds", data["result"] == "H"),
        ("1X2", pd.Series("Draw", index=data.index), result_probability["D"],
         "avg_draw_odds", data["result"] == "D"),
        ("1X2", data["away_team"] + " win", result_probability["A"],
         "avg_away_odds", data["result"] == "A"),
        ("Goals", pd.Series("Over 2.5 goals", index=data.index), over_probability,
         "avg_over_2_5_odds", total_goals > 2),
        ("Goals", pd.Series("Under 2.5 goals", index=data.index), 1 - over_probability,
         "avg_under_2_5_odds", total_goals <= 2),
    ]

    parts = []
    for market, pick, probability, odds_column, won in markets:
        parts.append(pd.DataFrame({
            "match_key": data.index,
            "date": data["date"],
            "match": data["home_team"] + " vs " + data["away_team"],
            "market": market,
            "pick": pick,
            "odds": data[odds_column],
            "model_probability": probability,
            "won": won.where(played),
        }))

    bets = pd.concat(parts, ignore_index=True)

    # Edge = how much you expect to win per 1 unit staked (0.05 = +5%)
    bets["edge"] = bets["model_probability"] * bets["odds"] - 1
    return bets


def summarise(name, bets):
    # bets needs two columns: odds and won (True / False)
    won = bets["won"].astype(bool)
    profit = (bets["odds"] - 1).where(won, -1)
    count = max(len(bets), 1)
    return {
        "Strategy": name,
        "Bets": len(bets),
        "Win %": won.sum() / count * 100,
        "Avg odds": bets["odds"].sum() / count,
        "Profit (units)": profit.sum(),
        "ROI %": profit.sum() / count * 100,
    }


def build_parlays(value_bets, legs):
    # Xien: moi tuan lay `legs` keo co edge cao nhat, moi tran 1 keo
    value_bets = value_bets.assign(week=value_bets["date"].dt.to_period("W"))
    parlays = []
    for week, week_bets in value_bets.groupby("week"):
        picks = (
            week_bets.sort_values("edge", ascending=False)
            .drop_duplicates("match_key")
            .head(legs)
        )
        if len(picks) < legs:
            continue
        parlays.append({
            "week": week,
            "odds": picks["odds"].prod(),
            "won": picks["won"].astype(bool).all(),
        })
    return pd.DataFrame(parlays, columns=["week", "odds", "won"])


def pick_value_bets(bets):
    # Keep bets with enough edge, and only the best one per match per market
    # (so we never back both "Draw" and "Home win" in the same game)
    return (
        bets[bets["edge"] >= MIN_EDGE]
        .sort_values("edge", ascending=False)
        .drop_duplicates(["match_key", "market"])
        .reset_index(drop=True)
    )


def percent(value):
    return f"{value * 100:.0f}%"


# ---------------------------------------------------------------------------
# 1. BACKTEST: WOULD THESE SUGGESTIONS HAVE MADE MONEY?
# Train on old seasons, then "bet" on every later match and count the profit.
# ---------------------------------------------------------------------------

past = feature_data[feature_data["season"] < BACKTEST_FROM_SEASON]
later = feature_data[feature_data["season"] >= BACKTEST_FROM_SEASON]

backtest_result_model = train_model(past, result_features, "result")
backtest_goals_model = train_model(past, goals_features, "over_2_5")

backtest_bets = list_bets(later, backtest_result_model, backtest_goals_model)
backtest_value = pick_value_bets(backtest_bets)

favourites = (
    backtest_bets[backtest_bets["market"] == "1X2"]
    .sort_values("odds")
    .drop_duplicates("match_key")
)

strategies = [
    summarise("Value bets: 1X2", backtest_value[backtest_value["market"] == "1X2"]),
    summarise("Value bets: over/under 2.5", backtest_value[backtest_value["market"] == "Goals"]),
    summarise("All value bets", backtest_value),
    summarise("Parlay of 2 (best 2 value bets each week)", build_parlays(backtest_value, 2)),
    summarise("Parlay of 3 (best 3 value bets each week)", build_parlays(backtest_value, 3)),
    summarise("Reference: always back the favourite", favourites),
]

backtest_table = pd.DataFrame(strategies)

print("=" * 78)
print(
    f"BACKTEST: betting 1 unit on each suggestion, "
    f"{later['date'].min():%b %Y} to {later['date'].max():%b %Y}"
)
print(f"(Model trained only on seasons before {BACKTEST_FROM_SEASON})")
print("=" * 78)
print(
    backtest_table.to_string(
        index=False,
        justify="left",
        formatters={
            "Strategy": "{:<42}".format,
            "Win %": "{:.1f}".format,
            "Avg odds": "{:.2f}".format,
            "Profit (units)": "{:+.1f}".format,
            "ROI %": "{:+.1f}".format,
        }
    )
)
print("ROI = profit per 100 units staked. Below 0 means you lose money.")


# ---------------------------------------------------------------------------
# 2. MATCHES TO PREVIEW: upcoming fixtures, or replay the latest round
# ---------------------------------------------------------------------------

def load_upcoming_fixtures():
    try:
        fixtures = pd.read_csv(FIXTURES_URL, encoding="utf-8-sig")
    except Exception as error:
        print(f"\nCould not download fixtures: {error}")
        return pd.DataFrame()

    fixtures = fixtures[fixtures["Div"] == "E0"]
    fixtures = fixtures.rename(columns=fixture_columns)[list(fixture_columns.values())]
    fixtures["date"] = pd.to_datetime(fixtures["date"], dayfirst=True)

    # Not played yet, so no score
    fixtures["home_goals"] = float("nan")
    fixtures["away_goals"] = float("nan")
    fixtures["result"] = None
    return fixtures


preview = load_upcoming_fixtures()
replay_mode = preview.empty

if replay_mode:
    # No Premier League games in the fixtures file yet (odds usually appear
    # a few days before). Replay the latest 10 games as if before kick-off.
    preview = matches.tail(10).copy()

preview = add_odds_probabilities(preview.reset_index(drop=True))

# Form as it was before each match
skipped = []
for index, game in preview.iterrows():
    for side in ["home", "away"]:
        form = team_form(game[f"{side}_team"], game["date"])
        if form is None:
            skipped.append(game[f"{side}_team"])
            continue
        preview.loc[index, f"{side}_form_points_5"] = form["points"]
        preview.loc[index, f"{side}_form_goals_for_5"] = form["goals_for"]
        preview.loc[index, f"{side}_form_goals_against_5"] = form["goals_against"]

preview = preview.dropna(subset=result_features).reset_index(drop=True)

# Train on every match played before the first preview game
history = feature_data[feature_data["date"] < preview["date"].min()]
result_model = train_model(history, result_features, "result")
goals_model = train_model(history, goals_features, "over_2_5")

bets = list_bets(preview, result_model, goals_model)


# ---------------------------------------------------------------------------
# 3. PRE-MATCH STATISTICS, ONE CARD PER MATCH
# ---------------------------------------------------------------------------

print("\n" + "=" * 78)
if replay_mode:
    print("No upcoming Premier League fixtures with odds yet.")
    print("REPLAY of the latest round: only data from before kick-off is used,")
    print("and the real result is shown at the bottom of each card.")
else:
    print("PRE-MATCH PREVIEW: upcoming Premier League fixtures")
print("=" * 78)

if skipped:
    print(f"Skipped (fewer than 5 games in the data): {', '.join(sorted(set(skipped)))}")

for index, game in preview.iterrows():
    home, away, date = game["home_team"], game["away_team"], game["date"]

    home_all = team_form(home, date)
    away_all = team_form(away, date)
    home_venue = team_form(home, date, venue="Home")
    away_venue = team_form(away, date, venue="Away")

    def show(form, key, style="{:.2f}"):
        return "-" if form is None else style.format(form[key])

    stat_rows = [
        ("Last 5 (oldest -> newest)", home_all["last_5"], away_all["last_5"]),
        ("Last 5 at home / away", show(home_venue, "last_5", "{}"), show(away_venue, "last_5", "{}")),
        ("Points per game", show(home_all, "points"), show(away_all, "points")),
        ("Goals scored per game", show(home_all, "goals_for", "{:.1f}"), show(away_all, "goals_for", "{:.1f}")),
        ("Goals conceded per game", show(home_all, "goals_against", "{:.1f}"), show(away_all, "goals_against", "{:.1f}")),
        ("Games with 3+ goals", percent(home_all["over_2_5"]), percent(away_all["over_2_5"])),
        ("Clean sheets", percent(home_all["clean_sheets"]), percent(away_all["clean_sheets"])),
    ]

    match_bets = bets[bets["match_key"] == index].set_index("pick")
    over_probability = match_bets.loc["Over 2.5 goals", "model_probability"]

    print(f"\n{date:%a %d %b %Y}   {home} vs {away}")
    print("-" * 78)
    print(f"{'':<28}{home:<20}{away:<20}")
    for label, home_value, away_value in stat_rows:
        print(f"{label:<28}{home_value:<20}{away_value:<20}")
    print(f"{'Head-to-head':<28}{head_to_head(home, away, date)}")

    line = game["asian_handicap_line"]
    if line == 0:
        handicap_text = "Level (0), no team gives goals"
    else:
        favourite = home if line < 0 else away
        handicap_text = f"{favourite} gives {abs(line):g} goal{'' if abs(line) == 1 else 's'}"
    print(f"{'Asian handicap':<28}{handicap_text}")

    print()
    print(f"{'':<28}{'Home':<9}{'Draw':<9}{'Away':<9}{'Over 2.5':<10}{'Under 2.5':<10}")
    print(
        f"{'Odds':<28}"
        f"{game['avg_home_odds']:<9.2f}{game['avg_draw_odds']:<9.2f}{game['avg_away_odds']:<9.2f}"
        f"{game['avg_over_2_5_odds']:<10.2f}{game['avg_under_2_5_odds']:<10.2f}"
    )
    print(
        f"{'Bookmaker chance':<28}"
        f"{percent(game['odds_home_probability']):<9}{percent(game['odds_draw_probability']):<9}"
        f"{percent(game['odds_away_probability']):<9}"
        f"{percent(game['odds_over_2_5_probability']):<10}{percent(1 - game['odds_over_2_5_probability']):<10}"
    )
    print(
        f"{'Model chance':<28}"
        + "".join(
            f"{percent(match_bets.loc[pick, 'model_probability']):<{width}}"
            for pick, width in [
                (f"{home} win", 9), ("Draw", 9), (f"{away} win", 9),
                ("Over 2.5 goals", 10), ("Under 2.5 goals", 10),
            ]
        )
    )
    print(
        f"{'Edge (value)':<28}"
        + "".join(
            f"{match_bets.loc[pick, 'edge'] * 100:+.0f}%".ljust(width)
            for pick, width in [
                (f"{home} win", 9), ("Draw", 9), (f"{away} win", 9),
                ("Over 2.5 goals", 10), ("Under 2.5 goals", 10),
            ]
        )
    )

    if replay_mode:
        goals = int(game["home_goals"] + game["away_goals"])
        print(
            f"{'Real result':<28}{int(game['home_goals'])}-{int(game['away_goals'])}, "
            f"{'over' if goals > 2 else 'under'} 2.5 goals"
        )


# ---------------------------------------------------------------------------
# 4. SUGGESTED BETS
# ---------------------------------------------------------------------------

value_bets = pick_value_bets(bets)

print("\n" + "=" * 78)
print(f"SUGGESTED BETS (model edge of at least {MIN_EDGE:.0%})")
print("=" * 78)

if value_bets.empty:
    print("No bet has enough edge this round. Best choice: don't bet.")
else:
    suggestion_table = pd.DataFrame({
        "Match": value_bets["match"].str.ljust(value_bets["match"].str.len().max()),
        "Bet": value_bets["pick"].str.ljust(value_bets["pick"].str.len().max()),
        "Odds": value_bets["odds"].map("{:.2f}".format),
        "Model %": (value_bets["model_probability"] * 100).map("{:.0f}".format),
        "Break-even %": (100 / value_bets["odds"]).map("{:.0f}".format),
        "Edge %": (value_bets["edge"] * 100).map("{:+.1f}".format),
    })
    if replay_mode:
        suggestion_table["Result"] = value_bets["won"].map({True: "won", False: "lost"})
    print(suggestion_table.to_string(index=False, justify="left"))
    print("Break-even % = how often the bet must win just to not lose money.")
    print(
        f"In the backtest, bets like these returned "
        f"{backtest_table.loc[2, 'ROI %']:+.1f}% per bet."
    )

    # Xien 2 / xien 3 from different matches
    best_per_match = value_bets.drop_duplicates("match_key")
    for legs in [2, 3]:
        if len(best_per_match) < legs:
            continue
        picks = best_per_match.head(legs)
        odds = picks["odds"].prod()
        chance = picks["model_probability"].prod()
        print(f"\nParlay of {legs}:")
        for _, leg in picks.iterrows():
            print(f"  - {leg['pick']:<18} ({leg['match']}) @ {leg['odds']:.2f}")
        print(
            f"  Combined odds {odds:.2f}, model chance {chance:.0%}, "
            f"edge {chance * odds - 1:+.0%}"
        )
        print(
            f"  In the backtest, parlays of {legs} returned "
            f"{backtest_table.loc[2 + legs - 1, 'ROI %']:+.1f}% per bet."
        )
        if replay_mode:
            print(f"  Result: {'won' if picks['won'].astype(bool).all() else 'lost'}")

print(
    "\nRead the backtest table at the top before betting: the bookmaker's odds\n"
    "already contain most of what this model knows, so small edges are\n"
    "often just noise. A parlay multiplies the bookmaker's margin, it does not\n"
    "remove it."
)
