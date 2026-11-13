import time
from pathlib import Path

import pandas as pd
import requests


# Downloads daily weather at each team's home stadium from Open-Meteo
# (free, no API key). Weather data by Open-Meteo.com, CC BY 4.0.
#
# - archive API: past days (available after a delay of a few days)
# - forecast API: the last 10 days + the next 16 days
#
# Output: data/external/weather_daily.csv  (team, date, temperature, rain, wind)

BASE_DIR = Path(__file__).resolve().parent
MATCHES_PATH = BASE_DIR / "data" / "processed" / "premier_league_matches.csv"
OUTPUT_PATH = BASE_DIR / "data" / "external" / "weather_daily.csv"
CACHE_DIR = BASE_DIR / "data" / "external" / "weather_cache"

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DAILY_VALUES = "temperature_2m_mean,precipitation_sum,wind_speed_10m_max"

# Home stadium (latitude, longitude). Weather is city-scale, so close is enough.
STADIUMS = {
    "Arsenal": (51.555, -0.108),
    "Aston Villa": (52.509, -1.885),
    "Bournemouth": (50.735, -1.838),
    "Brentford": (51.491, -0.289),
    "Brighton": (50.862, -0.083),
    "Burnley": (53.789, -2.230),
    "Chelsea": (51.482, -0.191),
    "Coventry": (52.448, -1.496),
    "Crystal Palace": (51.398, -0.086),
    "Everton": (53.425, -3.004),
    "Fulham": (51.475, -0.222),
    "Hull": (53.746, -0.368),
    "Ipswich": (52.055, 1.145),
    "Leeds": (53.778, -1.572),
    "Leicester": (52.620, -1.142),
    "Liverpool": (53.431, -2.961),
    "Luton": (51.884, -0.432),
    "Man City": (53.483, -2.200),
    "Man United": (53.463, -2.291),
    "Newcastle": (54.976, -1.622),
    "Norwich": (52.622, 1.309),
    "Nott'm Forest": (52.940, -1.133),
    "Sheffield United": (53.370, -1.471),
    "Southampton": (50.906, -1.391),
    "Sunderland": (54.915, -1.388),
    "Tottenham": (51.604, -0.066),
    "Watford": (51.650, -0.402),
    "West Ham": (51.539, -0.017),
    "Wolves": (52.590, -2.130),
}

COLUMN_NAMES = {
    "time": "date",
    "temperature_2m_mean": "temperature",
    "precipitation_sum": "rain_mm",
    "wind_speed_10m_max": "wind_kmh",
}


def download(url, latitude, longitude, **extra):
    # Open-Meteo counts long date ranges as many calls and limits calls per
    # minute, so on "429 Too Many Requests" wait a minute and try again.
    for attempt in range(6):
        response = requests.get(
            url,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "daily": DAILY_VALUES,
                "timezone": "Europe/London",
                **extra,
            },
            timeout=60,
        )
        if response.status_code != 429:
            break
        print("  rate limit reached, waiting 65 seconds...", flush=True)
        time.sleep(65)

    response.raise_for_status()
    data = pd.DataFrame(response.json()["daily"]).rename(columns=COLUMN_NAMES)
    data["date"] = pd.to_datetime(data["date"])
    return data


matches = pd.read_csv(MATCHES_PATH, parse_dates=["date"])
teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))

missing = [team for team in teams if team not in STADIUMS]
if missing:
    print(f"No stadium location for: {', '.join(missing)} - add them to STADIUMS.")

start_date = matches["date"].min().date().isoformat()
# The archive is a few days behind; the forecast API covers the gap.
archive_end = (pd.Timestamp.today().normalize() - pd.Timedelta(days=6)).date().isoformat()

parts = []
for team in teams:
    if team not in STADIUMS:
        continue
    latitude, longitude = STADIUMS[team]

    # Past weather never changes, so keep it in a cache file per team
    # and only download the days that are new since the last run.
    cache_path = CACHE_DIR / f"{team}.csv"
    if cache_path.exists():
        past = pd.read_csv(cache_path, parse_dates=["date"])
        fetch_from = (past["date"].max() + pd.Timedelta(days=1)).date().isoformat()
    else:
        past = pd.DataFrame()
        fetch_from = start_date

    if fetch_from <= archive_end:
        new_days = download(ARCHIVE_URL, latitude, longitude, start_date=fetch_from, end_date=archive_end)
        past = pd.concat([past, new_days], ignore_index=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        past.to_csv(cache_path, index=False)

    recent = download(FORECAST_URL, latitude, longitude, past_days=10, forecast_days=16)

    # Keep archive values where both exist (they are measured, not forecast)
    weather = pd.concat([past, recent]).drop_duplicates("date", keep="first")
    weather["team"] = team
    parts.append(weather)
    print(f"{team:<18} {len(weather)} days", flush=True)

weather = pd.concat(parts, ignore_index=True)[["team", "date", "temperature", "rain_mm", "wind_kmh"]]

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
weather.to_csv(OUTPUT_PATH, index=False)
print(f"\nSaved {len(weather)} rows to {OUTPUT_PATH}")
