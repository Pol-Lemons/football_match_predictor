from pathlib import Path

import pandas as pd

BASE_DIR =Path(__file__).resolve().parent
RAW_DIR = BASE_DIR/"data"/"raw"

PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)



season_urls = {
    "2021-2022": (
        "https://www.football-data.co.uk/"
        "mmz4281/2122/E0.csv"
    ),
    "2022-2023": (
        "https://www.football-data.co.uk/"
        "mmz4281/2223/E0.csv"
    ),
    "2023-2024": (
        "https://www.football-data.co.uk/"
        "mmz4281/2324/E0.csv"
    ),
    "2024-2025": (
        "https://www.football-data.co.uk/"
        "mmz4281/2425/E0.csv"
    ),
    "2025-2026": (
        "https://www.football-data.co.uk/"
        "mmz4281/2526/E0.csv"
    ),
    "2026-2027": (
        "https://www.football-data.co.uk/"
        "mmz4281/2627/E0.csv"
    ),
}


# Short code in the CSV -> readable name we use from now on
columns = {
    # Match info
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "Referee": "referee",

    # Full-time result
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",  # H = home win, D = draw, A = away win

    # Half-time result
    "HTHG": "home_goals_half_time",
    "HTAG": "away_goals_half_time",
    "HTR": "result_half_time",

    # Shots
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",

    # Corners and fouls
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",

    # Cards
    "HY": "home_yellow_cards",
    "AY": "away_yellow_cards",
    "HR": "home_red_cards",
    "AR": "away_red_cards",

    # Betting odds before the match (average across bookmakers)
    # Keo 1X2: thang nha / hoa / thang khach
    "AvgH": "avg_home_odds",
    "AvgD": "avg_draw_odds",
    "AvgA": "avg_away_odds",

    # Keo chau A (handicap): line for the home team, e.g. -1.5 = home gives 1.5 goals
    "AHh": "asian_handicap_line",
    "AvgAHH": "avg_asian_handicap_home_odds",
    "AvgAHA": "avg_asian_handicap_away_odds",

    # Keo tai xiu 2.5 ban
    "Avg>2.5": "avg_over_2_5_odds",
    "Avg<2.5": "avg_under_2_5_odds",
}

result_labels = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win",
}

all_seasons = []
for season, url in season_urls.items():
    print(f"Loading season {season}...")
    season_data = pd.read_csv(url)

    missing_columns = [
        column
        for column in columns
        if column not in season_data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{season} is missing columns: "
            f"{missing_columns}"
        )

    season_data = season_data[list(columns)].rename(columns=columns)
    season_data.insert(0, "season", season)

    season_data["date"] = pd.to_datetime(
        season_data["date"],
        dayfirst=True,
        errors="coerce"
    )

    season_data = season_data.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "result"
        ]
    )

    season_data["result_label"] = season_data["result"].map(result_labels)

    raw_filename = (
        f"premier_league_"
        f"{season.replace('-','_')}.csv"
    )

    season_data.to_csv(
        RAW_DIR/ raw_filename,
        index=False
    )

    all_seasons.append(season_data)

matches = pd.concat(
    all_seasons,
    ignore_index=True
)

matches = matches.sort_values(
    "date"
).reset_index(drop=True)

processed_path = (
    PROCESSED_DIR / "premier_league_matches.csv"
)

matches.to_csv(
    processed_path,
    index=False
)

print("\nRows and Columns: ")
print(matches.shape)

print("\nDate range: ")
print(matches["date"].min())
print(matches["date"].max())

print("\nMatches per season: ")
print(matches.groupby("season").size())

print("\nResult distribution: ")
print(matches["result_label"].value_counts())

print("\nMissing Values: ")
missing = matches.isna().sum()
missing = missing[missing > 0]
print(missing if len(missing) else "None")


def print_match_table(rows, title):
    # One line per match, home stat on the left, away stat on the right
    header = (
        f"{'Date':<12} {'Home':>16}  {'Score':^5}  {'Away':<16} "
        f"{'HT':^5} {'Result':<9} {'Shots (on target)':^19} "
        f"{'Corners':^7} {'Yellow':^6} {'Red':^5}"
    )
    print(f"\n{title}")
    print(header)
    print("-" * len(header))

    for _, m in rows.iterrows():
        score = f"{m.home_goals}-{m.away_goals}"
        half_time = f"{m.home_goals_half_time}-{m.away_goals_half_time}"
        shots = (
            f"{m.home_shots} ({m.home_shots_on_target})"
            f" - {m.away_shots} ({m.away_shots_on_target})"
        )
        corners = f"{m.home_corners}-{m.away_corners}"
        yellow = f"{m.home_yellow_cards}-{m.away_yellow_cards}"
        red = f"{m.home_red_cards}-{m.away_red_cards}"

        print(
            f"{m.date:%d %b %Y}  {m.home_team:>16}  {score:^5}  {m.away_team:<16} "
            f"{half_time:^5} {m.result_label:<9} {shots:^19} "
            f"{corners:^7} {yellow:^6} {red:^5}"
        )


print_match_table(matches.head(5), "First 5 matches:")
print_match_table(matches.tail(5), "Latest 5 matches:")

first_season = next(iter(season_urls))
first_season_matches = matches[matches["season"] == first_season]
print_match_table(
    first_season_matches,
    f"All {len(first_season_matches)} matches of season {first_season}:"
)

print("\nSaved combined data to: ")
print(processed_path)

def team_performance(matches, team, season=None):
    # Lọc tất cả trận đội này đá (sân nhà hoặc sân khách)
    games = matches[
        (matches["home_team"] == team) | (matches["away_team"] == team)
    ]
    if season:
        games = games[games["season"] == season]

    if games.empty:
        print(f"\nKhông tìm thấy trận nào của '{team}'.")
        print("Tên đội hợp lệ:", ", ".join(sorted(matches["home_team"].unique())))
        return

    # Nhìn từ góc độ của đội này
    is_home = games["home_team"] == team
    goals_for = games["home_goals"].where(is_home, games["away_goals"])
    goals_against = games["away_goals"].where(is_home, games["home_goals"])
    won = (games["result"] == "H") & is_home | (games["result"] == "A") & ~is_home
    drawn = games["result"] == "D"
    lost = ~won & ~drawn

    summary = pd.DataFrame({
        "season": games["season"],
        "venue": is_home.map({True: "Home", False: "Away"}),
        "W": won.astype(int),
        "D": drawn.astype(int),
        "L": lost.astype(int),
        "GF": goals_for,
        "GA": goals_against,
    })
    summary["Pts"] = summary["W"] * 3 + summary["D"]

    def record(group):
        table = group.sum(numeric_only=True)
        table.insert(0, "P", group.size().values)
        table["GD"] = table["GF"] - table["GA"]
        table["Win %"] = (table["W"] / table["P"] * 100).round(1)
        return table

    title = f"{team}" + (f" - {season}" if season else " - all seasons")
    print(f"\n===== {title} =====")

    print("\nTheo mùa giải:")
    print(record(summary.groupby("season")).to_string())

    print("\nSân nhà vs sân khách:")
    print(record(summary.groupby("venue")).to_string())

    print_match_table(games, f"PERFORMANCE OF {team} IN {season or 'ALL SEASONS'}:")


# Đổi 2 dòng này để xem đội / mùa khác.
# selected_season = None -> xem tất cả các mùa
selected_season = "2023-2024"
selected_team = "Liverpool"  # tên phải giống trong dữ liệu, vd "Man City", "Tottenham"

team_performance(matches, selected_team, selected_season)
