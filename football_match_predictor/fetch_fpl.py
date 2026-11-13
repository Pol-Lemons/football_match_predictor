import json
from pathlib import Path

import pandas as pd
import requests


# Downloads player data for the current season from the Fantasy Premier League
# API (free, no key, unofficial - so we call it only a few times per run).
#
# Output:
#   data/external/fpl_players.csv         one row per player: team, position,
#                                         injury status, season minutes and xG
#   data/external/fpl_player_matches.csv  one row per player per team match:
#                                         minutes, goals, xG

BASE_DIR = Path(__file__).resolve().parent
EXTERNAL_DIR = BASE_DIR / "data" / "external"
CACHE_DIR = EXTERNAL_DIR / "fpl_cache"
SEASON = "2026-2027"

API_URL = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "football-match-predictor"}

# FPL team names -> names used by Football-Data (our match data)
TEAM_NAMES = {
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Man Utd": "Man United",
    "Spurs": "Tottenham",
}

POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

# Per-match player actions kept from FPL's live data.
# bps = FPL's Bonus Points System score, which the website turns into a 0-10 rating.
ACTION_STATS = [
    "assists", "saves", "tackles", "clearances_blocks_interceptions", "recoveries",
    "yellow_cards", "red_cards", "own_goals", "penalties_saved", "penalties_missed",
    "goals_conceded", "bonus", "bps",
]


def get(path):
    response = requests.get(f"{API_URL}/{path}", headers=HEADERS, timeout=60)
    response.raise_for_status()
    return response.json()


def get_gameweek(event_id, finished):
    # A finished gameweek never changes, so keep it in a cache file
    cache_path = CACHE_DIR / f"event_{event_id}.json"
    if finished and cache_path.exists():
        return json.loads(cache_path.read_text())

    data = get(f"event/{event_id}/live/")
    if finished:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data))
    return data


overview = get("bootstrap-static/")
fixtures = pd.DataFrame(get("fixtures/"))

teams = {
    team["id"]: TEAM_NAMES.get(team["name"], team["name"])
    for team in overview["teams"]
}

known_teams = set(
    pd.read_csv(BASE_DIR / "data" / "processed" / "premier_league_matches.csv",
                usecols=["home_team"])["home_team"]
)
unknown = sorted(set(teams.values()) - known_teams)
if unknown:
    print(f"FPL teams not found in match data: {unknown} - add them to TEAM_NAMES.")


# 1. PLAYERS: injury status and season totals
players = pd.DataFrame([
    {
        "player_id": player["id"],
        "player": player["web_name"],
        "team": teams[player["team"]],
        "position": POSITIONS[player["element_type"]],
        "status": player["status"],  # a = available, d = doubtful, i = injured, s = suspended, u/n = unavailable
        "chance_of_playing": player["chance_of_playing_next_round"],
        "news": player["news"],
        "penalties_order": player["penalties_order"],
        "minutes": player["minutes"],
        "goals": player["goals_scored"],
        "expected_goals": float(player["expected_goals"]),
    }
    for player in overview["elements"]
])


# 2. MATCHES: every finished fixture with date and teams
fixtures = fixtures[fixtures["finished"]].copy()
fixtures["date"] = (
    pd.to_datetime(fixtures["kickoff_time"], utc=True)
    .dt.tz_convert("Europe/London")
    .dt.tz_localize(None)
    .dt.normalize()
)
fixtures["home_team"] = fixtures["team_h"].map(teams)
fixtures["away_team"] = fixtures["team_a"].map(teams)
fixture_info = fixtures.set_index("id")[["event", "date", "home_team", "away_team"]]


# 3. PLAYER MATCHES: minutes, goals and xG per player per fixture
stats_rows = {}
skipped_double = 0
finished_events = [event for event in overview["events"] if event["finished"]]

for event in finished_events:
    live = get_gameweek(event["id"], finished=True)
    for element in live["elements"]:
        played_fixtures = {part["fixture"] for part in element["explain"]}
        if len(played_fixtures) > 1:
            # Two games in one gameweek: FPL only gives the total, so skip
            skipped_double += 1
            continue
        if len(played_fixtures) == 1:
            stats = element["stats"]
            stats_rows[(element["id"], played_fixtures.pop())] = {
                "minutes": stats["minutes"],
                "starts": stats["starts"],
                "goals": stats["goals_scored"],
                "expected_goals": float(stats["expected_goals"]),
                # Actions in the match, for the website's player table and rating
                **{name: stats[name] for name in ACTION_STATS},
                "expected_assists": float(stats["expected_assists"]),
            }

# One row for every player in every match his team played (0 minutes if he did not play)
rows = []
for fixture_id, fixture in fixture_info.iterrows():
    for side in ["home", "away"]:
        team = fixture[f"{side}_team"]
        for player in players[players["team"] == team].itertuples(index=False):
            stats = stats_rows.get(
                (player.player_id, fixture_id),
                {"minutes": 0, "starts": 0, "goals": 0, "expected_goals": 0.0, "expected_assists": 0.0,
                 **{name: 0 for name in ACTION_STATS}},
            )
            rows.append({
                "season": SEASON,
                "gameweek": fixture["event"],
                "date": fixture["date"],
                "home_team": fixture["home_team"],
                "away_team": fixture["away_team"],
                "team": team,
                "player_id": player.player_id,
                **stats,
            })

player_matches = pd.DataFrame(rows)

EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
players.to_csv(EXTERNAL_DIR / "fpl_players.csv", index=False)
# Team badge codes: the website builds logo links from them
pd.DataFrame([
    {"team": teams[team["id"]], "badge_code": team["code"]} for team in overview["teams"]
]).to_csv(EXTERNAL_DIR / "fpl_teams.csv", index=False)
player_matches.to_csv(EXTERNAL_DIR / "fpl_player_matches.csv", index=False)

unavailable = players[players["status"] != "a"]
print(f"Players: {len(players)} ({len(unavailable)} injured, doubtful or suspended)")
print(f"Gameweeks: {len(finished_events)} finished, {len(fixture_info)} matches, "
      f"{len(player_matches)} player-match rows")
if skipped_double:
    print(f"Skipped {skipped_double} players with two games in one gameweek.")
