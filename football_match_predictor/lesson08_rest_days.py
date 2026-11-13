from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Cap rest days: anything over 14 days (summer break, international break)
# counts as "fully rested".
MAX_REST_DAYS = 14


matches = pd.read_csv(
    PROCESSED_DIR / "premier_league_matches.csv",
    parse_dates=["date"]
).sort_values("date").reset_index(drop=True)

feature_data = pd.read_csv(
    PROCESSED_DIR / "premier_league_features_elo.csv",
    parse_dates=["date"]
)


# 1. Days since each team's previous league match.
# Note: data only has Premier League games, so cup / European games are missing.
last_played = {}
rest_rows = []

for match in matches.itertuples(index=False):
    rest = {}

    for side, team in [("home", match.home_team), ("away", match.away_team)]:
        if team in last_played:
            days = (match.date - last_played[team]).days
            rest[side] = min(days, MAX_REST_DAYS)
        else:
            rest[side] = MAX_REST_DAYS

    rest_rows.append({
        "date": match.date,
        "home_team": match.home_team,
        "away_team": match.away_team,
        "home_rest_days": rest["home"],
        "away_rest_days": rest["away"],
        "rest_difference": rest["home"] - rest["away"]
    })

    # Update after saving, so we only use information from before the match.
    last_played[match.home_team] = match.date
    last_played[match.away_team] = match.date


rest_data = pd.DataFrame(rest_rows)

feature_data = feature_data.merge(
    rest_data,
    on=["date", "home_team", "away_team"],
    how="left",
    validate="one_to_one"
)

feature_data.to_csv(
    PROCESSED_DIR / "premier_league_features_rest.csv",
    index=False
)


# 2. Do rested teams actually do better?
def percent(value):
    return f"{value * 100:.1f}%"


feature_data["rest_group"] = pd.cut(
    feature_data["rest_difference"],
    bins=[-100, -2, 1, 100],
    labels=["Home team more tired", "Similar rest", "Home team more rested"]
)

rest_summary = (
    feature_data
    .groupby("rest_group", observed=True)["result"]
    .value_counts(normalize=True)
    .unstack()[["H", "D", "A"]]
)
rest_summary["Matches"] = feature_data["rest_group"].value_counts()
rest_summary[["H", "D", "A"]] = rest_summary[["H", "D", "A"]].map(percent)
rest_summary = rest_summary.rename(columns={
    "H": "Home Win", "D": "Draw", "A": "Away Win"
})

print("\nResults by rest difference:")
print(rest_summary.to_string())


# 3. Does adding rest days improve the model?
# Train only on seasons BEFORE the test season, so the model never sees the future.
test_season = "2025-2026"
train_data = feature_data[feature_data["season"] < test_season]
test_data = feature_data[feature_data["season"] == test_season]

form_elo_features = [
    "home_form_points_5",
    "home_form_goals_for_5",
    "home_form_goals_against_5",
    "away_form_points_5",
    "away_form_goals_for_5",
    "away_form_goals_against_5",
    "elo_difference"
]

rest_features = ["home_rest_days", "away_rest_days", "rest_difference"]

odds_features = ["odds_home_probability", "odds_away_probability"]

feature_sets = {
    "Form + Elo": form_elo_features,
    "Form + Elo + Rest": form_elo_features + rest_features,
    "Form + Elo + Odds": form_elo_features + odds_features,
    "Form + Elo + Odds + Rest": form_elo_features + odds_features + rest_features
}

comparison_rows = []

for name, columns in feature_sets.items():
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=2000, random_state=42))
    ])
    model.fit(train_data[columns], train_data["result"])

    y_pred = model.predict(test_data[columns])
    y_probability = model.predict_proba(test_data[columns])

    comparison_rows.append({
        "Features": name,
        "Accuracy": percent(accuracy_score(test_data["result"], y_pred)),
        "Log Loss": round(
            log_loss(test_data["result"], y_probability, labels=model.classes_),
            4
        )
    })

print(f"\nModel comparison (test season {test_season}, {len(test_data)} matches):")
print(pd.DataFrame(comparison_rows).to_string(index=False))
print("\nLower Log Loss = better probabilities.")
