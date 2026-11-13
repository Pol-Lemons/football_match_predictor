from pathlib import Path

import numpy as np
import pandas as pd


# MODEL 1b: THE PLAYER ENGINEER (goalscorers)
# For every match, splits the team's expected goals (from Model 1) between
# its players:
#
#   player weight = player xG per 90 minutes x expected minutes / 90
#   player goals  = team expected goals x (player weight / all player weights)
#   chance to score at least once = 1 - e^(-player goals)
#
# Expected minutes come from the last 5 matches. For upcoming matches they are
# also multiplied by the FPL "chance of playing" (injuries, suspensions).
#
# Run fetch_fpl.py and lesson10_stats_engineer.py first.
# Output: data/processed/player_predictions.csv, which Model 2 reads.

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
EXTERNAL_DIR = BASE_DIR / "data" / "external"
OUTPUT_PATH = PROCESSED_DIR / "player_predictions.csv"

SEASON = "2026-2027"  # FPL only has the current season
RECENT_MATCHES = 5  # expected minutes = average of the last 5 team matches
MIN_TEAM_MATCHES = 3  # need a few matches before predicting
PRIOR_MINUTES = 270  # a player with few minutes is pulled toward his position's average
PLAYER_GOAL_SHARE = 0.95  # about 5% of goals are own goals (checked on FPL data)


match_stats = pd.read_csv(PROCESSED_DIR / "match_stats.csv", parse_dates=["date"])
players = pd.read_csv(EXTERNAL_DIR / "fpl_players.csv")
player_matches = pd.read_csv(EXTERNAL_DIR / "fpl_player_matches.csv", parse_dates=["date"])
player_matches = player_matches.merge(players[["player_id", "position"]], on="player_id")

matches = match_stats[match_stats["season"] == SEASON].reset_index(drop=True)


def availability(player):
    # FPL chance of playing: empty means no news, so available
    if pd.notna(player["chance_of_playing"]):
        return player["chance_of_playing"] / 100
    return 1.0 if player["status"] == "a" else 0.0


def predict_team(team, team_expected_goals, before_date, use_availability):
    # Only information from before the match
    history = player_matches[
        (player_matches["team"] == team) & (player_matches["date"] < before_date)
    ]
    team_dates = sorted(history["date"].unique())
    if len(team_dates) < MIN_TEAM_MATCHES:
        return pd.DataFrame()

    # Average xG per 90 for each position (all teams), used for players with few minutes
    league_history = player_matches[player_matches["date"] < before_date]
    position_totals = league_history.groupby("position")[["expected_goals", "minutes"]].sum()
    position_xg90 = position_totals["expected_goals"] / position_totals["minutes"] * 90

    season = history.groupby("player_id")[["minutes", "expected_goals", "goals"]].sum()
    recent = history[history["date"].isin(team_dates[-RECENT_MATCHES:])]
    usual_minutes = recent.groupby("player_id")["minutes"].mean()

    # Player info only - FPL's season totals include later matches, so they are not used
    info_columns = ["player", "team", "position", "status", "chance_of_playing", "news", "penalties_order"]
    table = players.loc[players["team"] == team, ["player_id", *info_columns]].set_index("player_id")
    table = table.join(season.add_suffix("_season")).join(usual_minutes.rename("usual_minutes"))
    table[["minutes_season", "expected_goals_season", "goals_season", "usual_minutes"]] = (
        table[["minutes_season", "expected_goals_season", "goals_season", "usual_minutes"]].fillna(0)
    )

    prior = table["position"].map(position_xg90).fillna(0)
    table["xg_per_90"] = (
        (table["expected_goals_season"] + prior * PRIOR_MINUTES / 90)
        / (table["minutes_season"] + PRIOR_MINUTES)
        * 90
    )

    table["availability"] = table.apply(availability, axis=1) if use_availability else 1.0
    table["expected_minutes"] = table["usual_minutes"] * table["availability"]

    weight = table["xg_per_90"] * table["expected_minutes"] / 90
    if weight.sum() == 0:
        return pd.DataFrame()

    table["expected_goals_match"] = team_expected_goals * PLAYER_GOAL_SHARE * weight / weight.sum()
    table["chance"] = 1 - np.exp(-table["expected_goals_match"])

    # Comparison: the same method, but with real goals per 90 instead of xG per 90
    position_goals90 = position_totals.join(
        league_history.groupby("position")["goals"].sum()
    ).eval("goals / minutes * 90")
    goals_prior = table["position"].map(position_goals90).fillna(0)
    goals_per_90 = (
        (table["goals_season"] + goals_prior * PRIOR_MINUTES / 90)
        / (table["minutes_season"] + PRIOR_MINUTES)
        * 90
    )
    goals_weight = goals_per_90 * table["expected_minutes"] / 90
    table["goals_based_chance"] = 1 - np.exp(
        -team_expected_goals * PLAYER_GOAL_SHARE * goals_weight / goals_weight.sum()
    )

    # Share of the team's usual xG that is missing (injured / suspended / doubtful)
    usual_weight = table["xg_per_90"] * table["usual_minutes"] / 90
    table["xg_share_missing"] = (
        (usual_weight * (1 - table["availability"])).sum() / usual_weight.sum()
        if usual_weight.sum() > 0 else 0.0
    )

    table["team"] = team
    return table.reset_index()


rows = []
for match in matches.itertuples(index=False):
    use_availability = match.set == "upcoming"  # injury news is only known for the next games
    for side in ["home", "away"]:
        team = getattr(match, f"{side}_team")
        prediction = predict_team(
            team, getattr(match, f"exp_{side}_goals"), match.date, use_availability
        )
        if prediction.empty:
            continue
        prediction["set"] = match.set
        prediction["date"] = match.date
        prediction["home_team"] = match.home_team
        prediction["away_team"] = match.away_team
        rows.append(prediction)

predictions = pd.concat(rows, ignore_index=True)

# Actual goals in the match (backtest only)
actual = player_matches[["date", "home_team", "away_team", "player_id", "goals", "minutes"]].rename(
    columns={"goals": "goals_in_match", "minutes": "minutes_in_match"}
)
predictions = predictions.merge(actual, on=["date", "home_team", "away_team", "player_id"], how="left")
predictions["scored"] = (predictions["goals_in_match"] > 0).where(predictions["goals_in_match"].notna())


# ---------------------------------------------------------------------------
# HOW GOOD IS IT? Backtest on this season's played matches
# ---------------------------------------------------------------------------

def log_loss(scored, chance):
    chance = chance.clip(1e-6, 1 - 1e-6)
    return -(scored * np.log(chance) + (1 - scored) * np.log(1 - chance)).mean()


tested = predictions[
    (predictions["set"] == "backtest")
    & predictions["scored"].notna()
    & (predictions["expected_minutes"] > 0)
].copy()
tested["scored"] = tested["scored"].astype(float)

print("=" * 78)
print("MODEL 1b - GOALSCORERS")
print(f"Backtest: {tested[['date', 'home_team']].drop_duplicates().shape[0]} matches, "
      f"{len(tested)} player predictions ({SEASON})")
print("=" * 78)

model_loss = log_loss(tested["scored"], tested["chance"])
goals_loss = log_loss(tested["scored"], tested["goals_based_chance"])
print("Log loss (lower = better):")
print(f"  using xG per 90:    {model_loss:.4f}")
print(f"  using goals per 90: {goals_loss:.4f}  "
      f"(xG is {(goals_loss - model_loss) / goals_loss * 100:+.1f}% better)")

tested["bucket"] = pd.cut(
    tested["chance"], [0, 0.1, 0.2, 0.3, 0.45, 1],
    labels=["0-10%", "10-20%", "20-30%", "30-45%", "45%+"],
)
calibration = tested.groupby("bucket", observed=True).agg(
    Players=("chance", "size"),
    Model_said=("chance", "mean"),
    Scored=("scored", "mean"),
)
calibration["Model_said"] = calibration["Model_said"].map("{:.0%}".format)
calibration["Scored"] = calibration["Scored"].map("{:.0%}".format)
calibration = calibration.rename(columns={"Model_said": "Model said"})
print("\nWhen the model gives a player this chance, how often did he score?")
print(calibration.to_string())

APP_DIR = PROCESSED_DIR / "app"  # results for the website (app.py)
APP_DIR.mkdir(parents=True, exist_ok=True)
calibration.to_csv(APP_DIR / "goalscorer_calibration.csv", index_label="Model chance")

top_picks = (
    tested.sort_values("chance", ascending=False)
    .drop_duplicates(["date", "home_team", "away_team"])
)
print(f"\nTop pick of each match scored in {top_picks['scored'].mean():.0%} of matches "
      f"(model said {top_picks['chance'].mean():.0%} on average).")


output_columns = [
    "set", "date", "home_team", "away_team", "team", "player_id", "player", "position",
    "status", "news", "availability", "usual_minutes", "expected_minutes",
    "xg_per_90", "goals_season", "penalties_order", "expected_goals_match", "chance",
    "xg_share_missing", "scored",
]
predictions[output_columns].to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {len(predictions)} rows to {OUTPUT_PATH}")
