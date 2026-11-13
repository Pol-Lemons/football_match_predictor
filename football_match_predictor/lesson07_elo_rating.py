from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

raw_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "premier_league_matches.csv"
)

feature_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "premier_league_form_features.csv"
)

output_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "premier_league_features_elo.csv"
)


def normalize_columns(data):
    data = data.copy()

    data.columns = (
        data.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    return data.rename(columns={
        "match_date": "date",
        "hometeam": "home_team",
        "awayteam": "away_team",
        "fthg": "home_goals",
        "ftag": "away_goals",
        "ftr": "result"
    })


matches = normalize_columns(
    pd.read_csv(raw_path)
)

feature_data = normalize_columns(
    pd.read_csv(feature_path)
)


matches["date"] = pd.to_datetime(
    matches["date"],
    errors="coerce"
)

feature_data["date"] = pd.to_datetime(
    feature_data["date"],
    errors="coerce"
)


matches = (
    matches
    .dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "result"
        ]
    )
    .sort_values("date")
    .reset_index(drop=True)
)


INITIAL_ELO = 1500
K_FACTOR = 20
HOME_ADVANTAGE = 80


ratings = {}
elo_rows = []


for match in matches.itertuples(index=False):
    home_team = match.home_team
    away_team = match.away_team

    home_elo = ratings.get(
        home_team,
        INITIAL_ELO
    )

    away_elo = ratings.get(
        away_team,
        INITIAL_ELO
    )


    # Xác suất kỳ vọng của đội nhà.
    expected_home = 1 / (
        1
        + 10 ** (
            (
                away_elo
                - (
                    home_elo
                    + HOME_ADVANTAGE
                )
            )
            / 400
        )
    )


    if match.result == "H":
        actual_home = 1.0

    elif match.result == "D":
        actual_home = 0.5

    else:
        actual_home = 0.0


    # Lưu rating trước trận để tránh data leakage.
    elo_rows.append({
        "date": match.date,
        "season": match.season,
        "home_team": home_team,
        "away_team": away_team,
        "home_elo": home_elo,
        "away_elo": away_elo,
        "elo_difference": (
            home_elo - away_elo
        ),
        "expected_home_elo": expected_home
    })


    rating_change = K_FACTOR * (
        actual_home - expected_home
    )


    ratings[home_team] = (
        home_elo + rating_change
    )

    ratings[away_team] = (
        away_elo - rating_change
    )


elo_data = pd.DataFrame(elo_rows)


merge_columns = [
    "date",
    "season",
    "home_team",
    "away_team"
]


feature_data = feature_data.merge(
    elo_data,
    on=merge_columns,
    how="left",
    validate="one_to_one"
)


elo_columns = [
    "home_elo",
    "away_elo",
    "elo_difference",
    "expected_home_elo"
]


missing_elo = (
    feature_data[elo_columns]
    .isna()
    .sum()
)


print("\nMissing Elo values:")
print(missing_elo)


if missing_elo.sum() > 0:
    raise ValueError(
        "Some matches could not be matched "
        "with Elo data."
    )


feature_data.to_csv(
    output_path,
    index=False
)


display_columns = [
    "date",
    "home_team",
    "away_team",
    "home_elo",
    "away_elo",
    "elo_difference",
    "expected_home_elo",
    "result"
]


print("\nLast 10 matches with pre-match Elo:")

print(
    feature_data[display_columns]
    .tail(10)
    .to_string(
        index=False,
        float_format=lambda value: f"{value:.2f}"
    )
)


final_ratings = pd.DataFrame({
    "team": ratings.keys(),
    "elo_rating": ratings.values()
})

final_ratings = (
    final_ratings
    .sort_values(
        "elo_rating",
        ascending=False
    )
    .reset_index(drop=True)
)


print("\nCurrent team Elo ratings:")

print(
    final_ratings
    .head(20)
    .to_string(
        index=False,
        float_format=lambda value: f"{value:.2f}"
    )
)


print("\nSaved data to:")
print(output_path)