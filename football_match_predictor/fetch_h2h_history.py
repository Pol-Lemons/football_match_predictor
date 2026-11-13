import csv
import io
from pathlib import Path

import pandas as pd
import requests


# Older Premier League seasons (2000/01 to 2020/21) from Football-Data.co.uk,
# only for the website's head-to-head history - the models do not use them.
# Past seasons never change, so the file is downloaded once.
#
# Output: data/external/h2h_history.csv

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "data" / "external" / "h2h_history.csv"
URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"
FIRST_SEASON, LAST_SEASON = 2000, 2020  # the main data starts in 2021/22

# Same names as lesson01_load_data.py (match stats only, no odds)
COLUMNS = {
    "Date": "date", "HomeTeam": "home_team", "AwayTeam": "away_team", "Referee": "referee",
    "FTHG": "home_goals", "FTAG": "away_goals", "FTR": "result",
    "HTHG": "home_goals_half_time", "HTAG": "away_goals_half_time",
    "HS": "home_shots", "AS": "away_shots", "HST": "home_shots_on_target", "AST": "away_shots_on_target",
    "HC": "home_corners", "AC": "away_corners", "HF": "home_fouls", "AF": "away_fouls",
    "HY": "home_yellow_cards", "AY": "away_yellow_cards", "HR": "home_red_cards", "AR": "away_red_cards",
}

if OUTPUT_PATH.exists():
    print(f"{OUTPUT_PATH.name} already downloaded - delete it to download again.")
    raise SystemExit

seasons = []
for year in range(FIRST_SEASON, LAST_SEASON + 1):
    code = f"{year % 100:02d}{(year + 1) % 100:02d}"
    response = requests.get(URL.format(code=code), timeout=60)
    response.raise_for_status()
    # Old files use Windows encoding, and some rows have extra odds columns past the
    # header, so cut every row to the header's length instead of dropping it
    rows = list(csv.reader(io.StringIO(response.content.decode("latin-1"))))
    header = rows[0]
    data = pd.DataFrame([row[:len(header)] for row in rows[1:] if any(row)], columns=header)
    data = data[[column for column in COLUMNS if column in data.columns]].rename(columns=COLUMNS)
    data["date"] = pd.to_datetime(data["date"], dayfirst=True, format="mixed", errors="coerce")
    numbers = [column for column in data.columns if column.startswith(("home_", "away_"))
               and column not in ("home_team", "away_team")]
    data[numbers] = data[numbers].apply(pd.to_numeric, errors="coerce")
    data = data.dropna(subset=["date", "home_team", "away_team", "home_goals"])
    data.insert(0, "season", f"{year}-{year + 1}")
    seasons.append(data)
    print(f"{year}-{year + 1}: {len(data)} matches")

history = pd.concat(seasons, ignore_index=True)
assert history.groupby("season").size().between(370, 380).all(), "a season is missing matches"

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
history.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {len(history)} matches to {OUTPUT_PATH}")
