from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

data_path = (
    BASE_DIR / "data" / "processed" / "premier_league_matches.csv"
)

output_path = (
    BASE_DIR / "data" / "processed" / "premier_league_form_features.csv"
)

matches = pd.read_csv(data_path)

#CHUAN HOA TEN COT DE CODE CHAY VOI CA FTHG/FTAG va HomeGoals/AwayGoals

matches.columns = (
    matches.columns.str.strip().str.lower().str.replace(" ", "_")


)

matches = matches.rename(columns={
     "match_date": "date",
    "hometeam": "home_team",
    "awayteam": "away_team",
    "fthg": "home_goals",
    "ftag": "away_goals",
    "ftr": "result",
    "hs": "home_shots",
    "as": "away_shots",
    "hst": "home_shots_on_target",
    "ast": "away_shots_on_target"
})

required_columns = [
    "date",
    "home_team", 
    "away_team",
    "home_goals",
    "away_goals",
    "result",
    "season"
]

missing_columns = [
    column 
    for column in required_columns
    if column not in matches.columns
]
if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"


    )

matches["date"] = pd.to_datetime(
    matches["date"],
    errors="coerce"
)
matches = (
    matches
    .dropna(subset=required_columns)
    .sort_values("date")
    .reset_index(drop=True)
)

matches["match_id"] = matches.index

# DIEM CUA DOI NHA TRONG TUNG TRAN

home_points = np.select(
    [
        matches["result"].eq("H"),
        matches["result"].eq("D"),  
    ],
    [3,1],
    default=0    
)


# DIEM CUA DOI KHACH TRONG TUNG TRAN DAU

away_points = np.select(
    [
        matches["result"].eq("A"),
        matches["result"].eq("D")
    ],
    [3,1],
    default=0
)

# BIEN MOI TRAN THANH MOT DONG LICH SU CUA HOME TEAM

home_history = pd.DataFrame({
    "match_id": matches["match_id"],
    "date": matches["date"],
    "team": matches["home_team"],
    "opponent": matches["away_team"],
    "venue": "Home",
    "goals_for": matches["home_goals"],
    "goals_against": matches["away_goals"],
    "points": home_points
})

#BIEN MOI TRAN DAU THANH MOT DONG LICH CUA AWAY TEAM
away_history = pd.DataFrame({
    "match_id": matches["match_id"],
    "date": matches["date"],
    "team": matches["away_team"],
    "opponent": matches["home_team"],
    "venue": "Away",
    "goals_for": matches["away_goals"],
    "goals_against": matches["home_goals"],
    "points": away_points
})

team_history = pd.concat(
    [home_history, away_history],
    ignore_index=True   

)

team_history = team_history.sort_values(
    ["date", "match_id", "venue"]
).reset_index(drop=True)

# SO TRAN DOI DA CHOI TRUOC TRAN HIEN TAI

team_history["previous_matches"] = (
    team_history.groupby("team").cumcount()
)

#shift(1) bo tran hien khoi phep tinh

team_history["form_points_5"] = (
    team_history.groupby("team")["points"]
    .transform(
        lambda values:(
            values.shift(1).rolling(
                window=5,
                min_periods=5
            )
            .mean()
        )
    )
)

team_history["form_goals_for_5"] = (
    team_history.groupby("team")["goals_for"]
    .transform(
        lambda values: ( 
            values.shift(1).rolling(
                window=5,
                min_periods=5
            ).mean()
        )
    )
)

team_history["form_goals_against_5"] = (
    team_history.groupby("team")["goals_against"]
    .transform(
        lambda values: ( 
            values.shift(1).rolling(
                window=5,
                min_periods=5
            ).mean()
        )
    )
)

home_features = (
    team_history.loc[
        team_history["venue"].eq("Home"),
        [
            "match_id",
            "previous_matches",
            "form_points_5",
            "form_goals_for_5",
            "form_goals_against_5"
        ]
    ]
    .rename(columns={
     "previous_matches":
            "home_previous_matches",
     "form_points_5":
            "home_form_points_5",
     "form_goals_for_5":
            "home_form_goals_for_5",
     "form_goals_against_5":
            "home_form_goals_against_5"
})
)
away_features = (
    team_history.loc[
        team_history["venue"].eq("Away"),
        [
            "match_id",
            "previous_matches",
            "form_points_5",
            "form_goals_for_5",
            "form_goals_against_5"
        ]
    ].rename(columns={
        "previous_matches":
            "away_previous_matches",
        "form_points_5":
            "away_form_points_5",
        "form_goals_for_5":
            "away_form_goals_for_5",
        "form_goals_against_5":
            "away_form_goals_against_5"
    })
)

feature_data = matches.merge(
    home_features,
    on="match_id",
    how="left"
)

feature_data = feature_data.merge(
    away_features,
    on="match_id",
    how="left"
)

form_columns = [
    "home_form_points_5",
    "home_form_goals_for_5",
    "home_form_goals_against_5",
    "away_form_points_5",
    "away_form_goals_for_5",
    "away_form_goals_against_5"
]

# CHI GIU TRAN MA CA 2 DOI CO IT NHAT 5 TRAN CU
feature_data = (
    feature_data.dropna(subset=form_columns).reset_index(drop=True)
)

# SO SANH TRUC TIEP PHONG DO 2 DOI

feature_data["form_points_difference"] = (
    feature_data["home_form_points_5"] - feature_data["away_form_points_5"]
)

feature_data["form_attack_difference"] = (
    feature_data["home_form_goals_for_5"] - feature_data["away_form_goals_for_5"]
)

feature_data["form_defence_difference"] = (
    feature_data["home_form_goals_against_5"] - feature_data["away_form_goals_against_5"]
)

# DOI TY LE CUOC (ODDS) THANH XAC SUAT
# 1 / odds = xac suat nha cai ngam dinh.
# Chia cho tong de bo phan loi nhuan (margin) cua nha cai, cho tong = 1.

inverse_home = 1 / feature_data["avg_home_odds"]
inverse_draw = 1 / feature_data["avg_draw_odds"]
inverse_away = 1 / feature_data["avg_away_odds"]
inverse_total = inverse_home + inverse_draw + inverse_away

feature_data["odds_home_probability"] = inverse_home / inverse_total
feature_data["odds_draw_probability"] = inverse_draw / inverse_total
feature_data["odds_away_probability"] = inverse_away / inverse_total

# Keo tai xiu: xac suat tran co tu 3 ban tro len
inverse_over = 1 / feature_data["avg_over_2_5_odds"]
inverse_under = 1 / feature_data["avg_under_2_5_odds"]

feature_data["odds_over_2_5_probability"] = (
    inverse_over / (inverse_over + inverse_under)
)

odds_columns = [
    "odds_home_probability",
    "odds_draw_probability",
    "odds_away_probability",
    "odds_over_2_5_probability",
    "asian_handicap_line"
]

# Bo vai tran thieu du lieu keo
feature_data = (
    feature_data.dropna(subset=odds_columns).reset_index(drop=True)
)

feature_data.to_csv(
    output_path,
    index=False
)

columns_to_show = [
    "date",
    "home_team",
    "away_team",
    "home_form_points_5",
    "away_form_points_5",
    "form_points_difference",
    "result"
]

print("\nFeature data shape:")
print(feature_data.shape)

print("\nFirst 10 matches with form features:")
print(
    feature_data[columns_to_show]
    .head(10)
    .round(2)
    .to_string(index=False)
)

print("\nMissing form values:")
print(feature_data[form_columns].isna().sum())

print("\nSaved feature data to:")
print(output_path)
