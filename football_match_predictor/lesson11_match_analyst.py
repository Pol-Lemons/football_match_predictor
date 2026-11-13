import sys
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import poisson


# MODEL 2: THE MATCH ANALYST
# Reads Model 1's stats (data/processed/match_stats.csv) and turns them into:
#   - a written assessment of each match
#   - chance and fair odds for every market: 1X2, double chance, over/under,
#     both teams to score, odd/even goals, Asian handicap, corners, cards
#   - the final prediction and the best value bet against the bookmaker
#
# Run lesson10_stats_engineer.py first, then:
#   python lesson11_match_analyst.py               -> all upcoming games
#   python lesson11_match_analyst.py "Man United"  -> one team, with every market

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
STATS_PATH = PROCESSED_DIR / "match_stats.csv"
OUTPUT_PATH = PROCESSED_DIR / "match_predictions.csv"
APP_DIR = PROCESSED_DIR / "app"  # results for the website (app.py)

MIN_EDGE = 0.05
MAX_VALUE_ODDS = 5.0  # lesson 9: "edges" on long shots were mostly model mistakes

# Confidence = model chance of the most likely result (home / draw / away).
# Backtest: 50%+ was right about 62% of the time, under 50% only about 40%.
CONFIDENCE_LEVELS = [(0.60, "High"), (0.50, "Medium"), (0.0, "Low")]
CONFIDENT_CHANCE = 0.50  # below this, the pick becomes the best double chance

GOAL_LINES = [1.5, 2.5, 3.5]
CORNER_LINES = [8.5, 9.5, 10.5]
CARD_LINES = [3.5, 4.5]

# Markets shown on each match card (all markets are saved to the CSV)
KEY_MARKETS = ["1X2", "Goals 2.5", "Both teams score", "Odd/Even", "Handicap", "Corners 9.5", "Cards 3.5"]

# Grid of every score: HOME[i, j] = i, AWAY[i, j] = j
GOAL_RANGE = np.arange(11)
HOME, AWAY = np.meshgrid(GOAL_RANGE, GOAL_RANGE, indexing="ij")
MARGIN = HOME - AWAY
TOTAL = HOME + AWAY

# For corners and cards we only need the match total
COUNT_RANGE = np.arange(41)


stats = pd.read_csv(STATS_PATH, parse_dates=["date"])

# Goalscorers and missing players from lesson12 (optional)
PLAYERS_PATH = PROCESSED_DIR / "player_predictions.csv"
player_predictions = (
    pd.read_csv(PLAYERS_PATH, parse_dates=["date"]) if PLAYERS_PATH.exists() else pd.DataFrame()
)
backtest = stats[stats["set"] == "backtest"].reset_index(drop=True)
upcoming = stats[stats["set"] == "upcoming"].reset_index(drop=True)

# League averages, used as "normal" in the assessment
league = {
    "goals": (backtest["home_goals"] + backtest["away_goals"]).mean(),
    "corners": (backtest["home_corners"] + backtest["away_corners"]).mean(),
    "cards": (backtest["home_cards"] + backtest["away_cards"]).mean(),
}


# ---------------------------------------------------------------------------
# PROBABILITIES
# ---------------------------------------------------------------------------

def score_matrix(home_xg, away_xg):
    # chance[i, j] = chance the score ends i - j
    chance = np.outer(poisson.pmf(GOAL_RANGE, home_xg), poisson.pmf(GOAL_RANGE, away_xg))
    return chance / chance.sum()


def count_chances(expected):
    chance = poisson.pmf(COUNT_RANGE, expected)
    return chance / chance.sum()


def handicap_shares(margin, line):
    # For each score: share of the stake that WINS, and share that is REFUNDED.
    # Quarter lines (-0.25, -0.75, ...) split the stake over the two nearest lines.
    lines = [line - 0.25, line + 0.25] if round(line * 4) % 2 == 1 else [line]
    win = sum((margin + part > 0).astype(float) for part in lines) / len(lines)
    refund = sum((margin + part == 0).astype(float) for part in lines) / len(lines)
    return win, refund


def list_markets(game):
    # Every market: (group, market, pick, chances, win share, refund share, bookmaker odds, actual index)
    home, away = game.home_team, game.away_team
    score = score_matrix(game.exp_home_goals, game.exp_away_goals)
    no_refund = np.zeros_like(score)
    played = pd.notna(game.home_goals)
    actual_score = (min(int(game.home_goals), 10), min(int(game.away_goals), 10)) if played else None

    home_win, draw, away_win = MARGIN > 0, MARGIN == 0, MARGIN < 0
    nan = float("nan")

    def goal_market(market, pick, win, odds=nan, refund=no_refund):
        return (market, pick, score, win.astype(float), refund, odds, actual_score)

    markets = [
        goal_market("1X2", f"{home} win", home_win, game.avg_home_odds),
        goal_market("1X2", "Draw", draw, game.avg_draw_odds),
        goal_market("1X2", f"{away} win", away_win, game.avg_away_odds),
        goal_market("Double chance", f"{home} or draw", home_win | draw),
        goal_market("Double chance", f"{away} or draw", away_win | draw),
        goal_market("Double chance", f"{home} or {away}", home_win | away_win),
    ]
    for line in GOAL_LINES:
        over_odds = game.avg_over_2_5_odds if line == 2.5 else nan
        under_odds = game.avg_under_2_5_odds if line == 2.5 else nan
        markets += [
            goal_market(f"Goals {line}", f"Over {line} goals", TOTAL > line, over_odds),
            goal_market(f"Goals {line}", f"Under {line} goals", TOTAL < line, under_odds),
        ]
    markets += [
        goal_market("Both teams score", "Both teams score: Yes", (HOME > 0) & (AWAY > 0)),
        goal_market("Both teams score", "Both teams score: No", (HOME == 0) | (AWAY == 0)),
        goal_market("Odd/Even", "Odd total goals", TOTAL % 2 == 1),
        goal_market("Odd/Even", "Even total goals", TOTAL % 2 == 0),
    ]

    if pd.notna(game.asian_handicap_line):
        line = game.asian_handicap_line
        home_win_share, home_refund = handicap_shares(MARGIN, line)
        away_win_share, away_refund = handicap_shares(-MARGIN, -line)
        markets += [
            ("Handicap", f"{home} {line:+g}", score, home_win_share, home_refund,
             game.avg_asian_handicap_home_odds, actual_score),
            ("Handicap", f"{away} {0.0 - line:+g}", score, away_win_share, away_refund,
             game.avg_asian_handicap_away_odds, actual_score),
        ]

    for name, lines in [("Corners", CORNER_LINES), ("Cards", CARD_LINES)]:
        column = name.lower()
        chance = count_chances(game[f"exp_home_{column}"] + game[f"exp_away_{column}"])
        actual = (
            min(int(game[f"home_{column}"] + game[f"away_{column}"]), COUNT_RANGE[-1])
            if played else None
        )
        for line in lines:
            markets += [
                (f"{name} {line}", f"Over {line} {column}", chance,
                 (COUNT_RANGE > line).astype(float), np.zeros_like(chance), nan, actual),
                (f"{name} {line}", f"Under {line} {column}", chance,
                 (COUNT_RANGE < line).astype(float), np.zeros_like(chance), nan, actual),
            ]

    rows = []
    for market, pick, chance, win, refund, odds, actual in markets:
        win_chance = (chance * win).sum()
        refund_chance = (chance * refund).sum()
        row = {
            "date": game.date,
            "match": f"{home} vs {away}",
            "market": market,
            "pick": pick,
            "chance": win_chance,
            # Fair odds = odds where you break even in the long run
            "fair_odds": (1 - refund_chance) / win_chance if win_chance > 0 else np.inf,
            "bookmaker_odds": odds,
            "edge": odds * win_chance + refund_chance - 1 if pd.notna(odds) else nan,
        }
        if actual is not None:
            row["won"] = win[actual]
            row["profit"] = odds * win[actual] + refund[actual] - 1 if pd.notna(odds) else nan
        rows.append(row)

    return pd.DataFrame(rows)


def market_chance(markets, pick):
    return markets.loc[markets["pick"] == pick, "chance"].iloc[0]


def best_value_bet(markets):
    value = markets[
        (markets["edge"] >= MIN_EDGE)
        & (markets["bookmaker_odds"] <= MAX_VALUE_ODDS)
    ]
    return None if value.empty else value.sort_values("edge", ascending=False).iloc[0]


# ---------------------------------------------------------------------------
# ASSESSMENT: turn the numbers into sentences
# ---------------------------------------------------------------------------

def assess(game, markets):
    home, away = game.home_team, game.away_team
    notes = []

    # Strength (Elo + home advantage of 80 points)
    gap = game.home_elo - game.away_elo + 80
    if abs(gap) < 60:
        strength = "evenly matched"
    else:
        favourite = home if gap > 0 else away
        strength = f"{favourite} are {'clear' if abs(gap) >= 150 else 'slight'} favourites"
    notes.append(f"Strength: Elo {home} {game.home_elo:.0f} vs {away} {game.away_elo:.0f} -> {strength} (incl. home advantage).")

    # Form
    form_gap = game.home_form_points - game.away_form_points
    form_note = ""
    if abs(form_gap) >= 0.7:
        form_note = f" -> {home if form_gap > 0 else away} in much better form"
    notes.append(
        f"Form: {home} {game.home_form_points:.1f} vs {away} {game.away_form_points:.1f} "
        f"points per game (last 10){form_note}."
    )

    # Game flow: who creates more
    total_shots = game.exp_home_shots_on_target + game.exp_away_shots_on_target
    share = game.exp_home_shots_on_target / total_shots
    if share >= 0.6:
        flow = f"{home} should control the game"
    elif share <= 0.4:
        flow = f"{away} should control the game"
    else:
        flow = "balanced game"
    notes.append(
        f"Game flow: {flow} - expected shots on target "
        f"{game.exp_home_shots_on_target:.1f} vs {game.exp_away_shots_on_target:.1f}."
    )

    # Goals
    total_goals = game.exp_home_goals + game.exp_away_goals
    if total_goals >= league["goals"] + 0.4:
        goals_note = "open game, more goals than usual"
    elif total_goals <= league["goals"] - 0.4:
        goals_note = "tight game, fewer goals than usual"
    else:
        goals_note = "around the league average"
    notes.append(
        f"Goals: {total_goals:.1f} expected (league {league['goals']:.1f}) -> {goals_note}; "
        f"both teams score {market_chance(markets, 'Both teams score: Yes'):.0%}."
    )

    # Corners
    total_corners = game.exp_home_corners + game.exp_away_corners
    corner_note = (
        "more than usual" if total_corners >= league["corners"] + 0.8
        else "fewer than usual" if total_corners <= league["corners"] - 0.8
        else "about average"
    )
    notes.append(
        f"Corners: {game.exp_home_corners:.1f} vs {game.exp_away_corners:.1f} "
        f"(total {total_corners:.1f}, league {league['corners']:.1f}) -> {corner_note}."
    )

    # Cards and referee
    total_cards = game.exp_home_cards + game.exp_away_cards
    if game.referee_cards >= league["cards"] + 0.3:
        referee_note = "strict referee"
    elif game.referee_cards <= league["cards"] - 0.3:
        referee_note = "lenient referee"
    else:
        referee_note = "average referee"
    notes.append(
        f"Cards: {total_cards:.1f} expected. {game.referee} gives {game.referee_cards:.1f} "
        f"per game (league {league['cards']:.1f}) -> {referee_note}."
    )

    # Weather (only if Model 1 used it)
    if "weather_rain_mm" in game.index:
        conditions = []
        if game.weather_rain_mm >= 5:
            conditions.append("heavy rain")
        if game.weather_wind_kmh >= 40:
            conditions.append("strong wind")
        if game.weather_temperature <= 2:
            conditions.append("very cold")
        notes.append(
            f"Weather: {game.weather_temperature:.0f}°C, rain {game.weather_rain_mm:.1f} mm, "
            f"wind up to {game.weather_wind_kmh:.0f} km/h"
            + (f" -> {', '.join(conditions)}." if conditions else " -> normal conditions.")
        )

    # Rest days
    rest_gap = game.home_rest_days - game.away_rest_days
    if abs(rest_gap) >= 3:
        rested = home if rest_gap > 0 else away
        notes.append(f"Rest: {rested} have had {abs(rest_gap):.0f} more days of rest.")

    # Model vs bookmaker
    outcomes = [
        (f"{home} win", "odds_home_probability"),
        ("Draw", "odds_draw_probability"),
        (f"{away} win", "odds_away_probability"),
    ]
    differences = [
        (market_chance(markets, pick) - game[column], pick, market_chance(markets, pick), game[column])
        for pick, column in outcomes
    ]
    gap, pick, model, bookmaker = max(differences, key=lambda item: abs(item[0]))
    if abs(gap) >= 0.05:
        direction = "higher" if gap > 0 else "lower"
        notes.append(
            f"Market: model rates '{pick}' {direction} than the bookmaker "
            f"({model:.0%} vs {bookmaker:.0%})."
        )
    else:
        notes.append("Market: model and bookmaker agree on the result.")

    return notes


def players_in_match(game):
    if player_predictions.empty:
        return player_predictions
    return player_predictions[
        (player_predictions["date"] == game.date)
        & (player_predictions["home_team"] == game.home_team)
        & (player_predictions["away_team"] == game.away_team)
    ]


def missing_player_notes(players):
    # Regular players (45+ minutes in recent games) who may not play
    notes = []
    for team, team_players in players.groupby("team", sort=False):
        missing = team_players[
            (team_players["availability"] < 0.75) & (team_players["usual_minutes"] >= 45)
        ]
        if missing.empty:
            continue
        names = ", ".join(
            f"{player.player} ({player.news or 'unavailable'})" for player in missing.itertuples()
        )
        share = team_players["xg_share_missing"].iloc[0]
        notes.append(f"Missing for {team}: {names} -> {share:.0%} of their usual xG.")
    return notes


def print_goalscorers(players, per_team):
    if players.empty:
        print("\nGoalscorers: no player data (run fetch_fpl.py and lesson12_goalscorers.py)")
        return

    if per_team:
        shown = players.sort_values("chance", ascending=False).groupby("team", sort=False).head(per_team)
    else:
        shown = players.sort_values("chance", ascending=False).head(6)

    table = pd.DataFrame({
        "Player": shown["player"] + shown["penalties_order"].eq(1).map({True: " (pens)", False: ""}),
        "Team": shown["team"],
        "Pos": shown["position"],
        "Minutes": shown["expected_minutes"].round().astype(int),
        "xG/90": shown["xg_per_90"].map("{:.2f}".format),
        "Chance": shown["chance"].map("{:.0%}".format),
        "Fair odds": (1 / shown["chance"]).map("{:.2f}".format),
    })
    if shown["scored"].notna().any():
        table["Scored"] = shown["scored"].map({True: "yes", False: "", 1.0: "yes", 0.0: ""}).fillna("")

    print("\nLikely goalscorers (anytime):")
    print(table.to_string(index=False))


def predict_result(game, markets):
    home, away = game.home_team, game.away_team
    chances = {
        f"{home} win": market_chance(markets, f"{home} win"),
        "Draw": market_chance(markets, "Draw"),
        f"{away} win": market_chance(markets, f"{away} win"),
    }
    ranked = sorted(chances.items(), key=lambda item: item[1], reverse=True)
    (top_pick, top_chance), (_, second_chance) = ranked[0], ranked[1]

    if top_chance >= 0.5:
        return f"{top_pick} ({top_chance:.0%})"
    if top_chance - second_chance < 0.05:
        return f"Too close to call, slight lean: {top_pick} ({top_chance:.0%})"
    return f"Lean {top_pick} ({top_chance:.0%})"


def confidence_level(chance):
    for threshold, label in CONFIDENCE_LEVELS:
        if chance >= threshold:
            return label


def main_pick(markets):
    # The model's pick for the match: the most likely result when the model is
    # confident enough, otherwise the safest double chance (which covers the draw)
    result = markets[markets["market"] == "1X2"].sort_values("chance", ascending=False).iloc[0]
    level = confidence_level(result["chance"])
    if result["chance"] >= CONFIDENT_CHANCE:
        return result, level
    double = markets[markets["market"] == "Double chance"].sort_values("chance", ascending=False).iloc[0]
    return double, level


def summary_rows(game, markets, players):
    # One short row per match for the final summary tables
    home, away = game.home_team, game.away_team
    chances = {
        "H": market_chance(markets, f"{home} win"),
        "D": market_chance(markets, "Draw"),
        "A": market_chance(markets, f"{away} win"),
    }
    pick, level = main_pick(markets)

    score = score_matrix(game.exp_home_goals, game.exp_away_goals)
    best_home, best_away = np.unravel_index(score.argmax(), score.shape)
    over = market_chance(markets, "Over 2.5 goals")
    match = f"{home} vs {away}"

    result_row = {
        "Match": match,
        "Home/Draw/Away": f"{chances['H']:.0%} / {chances['D']:.0%} / {chances['A']:.0%}",
        "Confidence": level,
        "Pick": f"{pick['pick']} {pick['chance']:.0%}",
        "Likely score": f"{best_home}-{best_away}",
        "Exp. goals": f"{game.exp_home_goals + game.exp_away_goals:.1f}",
        "Over 2.5": f"{over:.0%}",
        "Both score": f"{market_chance(markets, 'Both teams score: Yes'):.0%}",
    }

    best = best_value_bet(markets)
    top_scorer = players.sort_values("chance", ascending=False).head(1)
    stats_row = {
        "Match": match,
        "Corners": f"{game.exp_home_corners + game.exp_away_corners:.1f}",
        "Cards": f"{game.exp_home_cards + game.exp_away_cards:.1f}",
        "Top scorer": (
            f"{top_scorer['player'].iloc[0]} {top_scorer['chance'].iloc[0]:.0%}"
            if not top_scorer.empty else "-"
        ),
        "Best bet": (
            f"{best['pick']} @ {best['bookmaker_odds']:.2f} ({best['edge']:+.0%})"
            if best is not None else "-"
        ),
    }

    checks = {"level": level}
    if pd.notna(game.home_goals):
        total_goals = game.home_goals + game.away_goals
        result_row["Actual"] = f"{game.home_goals:.0f}-{game.away_goals:.0f}"
        checks["pick"] = bool(pick["won"])
        checks["over_under"] = (over >= 0.5) == (total_goals > 2.5)
        result_row["Pick right?"] = "right" if checks["pick"] else "wrong"
        result_row["O/U 2.5"] = "right" if checks["over_under"] else "wrong"

        stats_row["Corners"] += f" ({game.home_corners + game.away_corners:.0f})"
        stats_row["Cards"] += f" ({game.home_cards + game.away_cards:.0f})"
        if not top_scorer.empty and pd.notna(top_scorer["scored"].iloc[0]):
            checks["scorer"] = bool(top_scorer["scored"].iloc[0])
            stats_row["Top scorer"] += " (scored)" if checks["scorer"] else " (no)"
        if best is not None:
            stats_row["Best bet"] += " won" if best["profit"] > 0 else " lost"

    # Same values as numbers (not text), for the website's tables
    raw = {
        "match": match,
        "home_team": home,
        "away_team": away,
        "home_chance": chances["H"],
        "draw_chance": chances["D"],
        "away_chance": chances["A"],
        "confidence": level,
        "pick": pick["pick"],
        "pick_market": pick["market"],
        "pick_chance": pick["chance"],
        "pick_odds": pick["bookmaker_odds"],
        "pick_edge": pick["edge"],
        "likely_score": f"{best_home}-{best_away}",
        "exp_goals": game.exp_home_goals + game.exp_away_goals,
        "over_2_5": over,
        "both_score": market_chance(markets, "Both teams score: Yes"),
        "exp_corners": game.exp_home_corners + game.exp_away_corners,
        "exp_cards": game.exp_home_cards + game.exp_away_cards,
        "top_scorer": top_scorer["player"].iloc[0] if not top_scorer.empty else None,
        "top_scorer_chance": top_scorer["chance"].iloc[0] if not top_scorer.empty else None,
        "best_bet": best["pick"] if best is not None else None,
        "best_bet_market": best["market"] if best is not None else None,
        "best_bet_odds": best["bookmaker_odds"] if best is not None else None,
        "best_bet_chance": best["chance"] if best is not None else None,
        "best_bet_edge": best["edge"] if best is not None else None,
    }
    if pd.notna(game.home_goals):
        raw.update({
            "actual": f"{game.home_goals:.0f}-{game.away_goals:.0f}",
            "actual_corners": game.home_corners + game.away_corners,
            "actual_cards": game.home_cards + game.away_cards,
            "pick_right": checks["pick"],
            "over_under_right": checks["over_under"],
            "btts_right": (raw["both_score"] >= 0.5) == (game.home_goals > 0 and game.away_goals > 0),
            "top_scorer_scored": checks.get("scorer"),
            "best_bet_profit": best["profit"] if best is not None else None,
        })

    return result_row, stats_row, checks, raw


def likely_scores(game, count=3):
    score = score_matrix(game.exp_home_goals, game.exp_away_goals)
    best = np.argsort(score, axis=None)[::-1][:count]
    return ", ".join(
        f"{i}-{j} ({score[i, j]:.0%})"
        for i, j in zip(*np.unravel_index(best, score.shape))
    )


# ---------------------------------------------------------------------------
# 1. BACKTEST: can we trust Model 2?
# ---------------------------------------------------------------------------

backtest_markets = pd.concat(
    [list_markets(game) for _, game in backtest.iterrows()],
    ignore_index=True,
)

print("=" * 78)
print("MODEL 2 - MATCH ANALYST")
print(f"Backtest: {len(backtest)} matches, {backtest['date'].min():%b %Y} to {backtest['date'].max():%b %Y}")
print("=" * 78)

# When the model is confident (60%+), how often is it right?
confident = backtest_markets[backtest_markets["chance"] >= 0.6].copy()
confident["group"] = confident["market"].str.replace(r" [0-9.]+$", "", regex=True)
reliability = confident.groupby("group").agg(
    Picks=("chance", "size"),
    Model_said=("chance", "mean"),
    Happened=("won", "mean"),
)
reliability["Model_said"] = reliability["Model_said"].map("{:.0%}".format)
reliability["Happened"] = reliability["Happened"].map("{:.0%}".format)
print("When the model gives a pick 60% or more, how often did it happen?")
print(reliability.rename(columns={"Model_said": "Model said", "Happened": "Happened"}).to_string())

value = backtest_markets[
    (backtest_markets["edge"] >= MIN_EDGE)
    & (backtest_markets["bookmaker_odds"] <= MAX_VALUE_ODDS)
].copy()
value["group"] = value["market"].str.replace(r" [0-9.]+$", "", regex=True)
roi = value.groupby("group").agg(
    Bets=("profit", "size"),
    Win=("won", "mean"),
    Profit=("profit", "sum"),
)
roi["ROI"] = (roi["Profit"] / roi["Bets"]).map("{:+.1%}".format)
roi["Win"] = roi["Win"].map("{:.0%}".format)
roi["Profit"] = roi["Profit"].round(1)
print(f"\nValue bets (edge >= {MIN_EDGE:.0%}, odds <= {MAX_VALUE_ODDS:.1f}), 1 unit each:")
print(roi.rename(columns={"Win": "Win %", "Profit": "Profit (units)"}).to_string())

# How often is the main pick right at each confidence level?
level_rows = []
for _, match_markets in backtest_markets.groupby(["date", "match"]):
    pick, level = main_pick(match_markets)
    top_result = match_markets[match_markets["market"] == "1X2"]["chance"].idxmax()
    level_rows.append({
        "level": level,
        "pick_type": pick["market"],
        "pick_right": pick["won"],
        "result_right": match_markets.loc[top_result, "won"],
    })
levels = pd.DataFrame(level_rows)
level_table = levels.groupby("level").agg(
    Matches=("level", "size"),
    Result_right=("result_right", "mean"),
    Pick=("pick_type", "first"),
    Pick_right=("pick_right", "mean"),
).reindex([label for _, label in CONFIDENCE_LEVELS])
level_accuracy = level_table["Pick_right"].to_dict()  # used in the summary

level_table["Share"] = (level_table["Matches"] / len(levels)).map("{:.0%}".format)
level_table["Result_right"] = level_table["Result_right"].map("{:.0%}".format)
level_table["Pick_right"] = level_table["Pick_right"].map("{:.0%}".format)
level_table.index = [
    f"{label} ({threshold:.0%}+)" if threshold else f"{label} (<{CONFIDENT_CHANCE:.0%})"
    for threshold, label in CONFIDENCE_LEVELS
]
level_table = level_table[["Matches", "Share", "Result_right", "Pick", "Pick_right"]].rename(columns={
    "Result_right": "Result right", "Pick": "Main pick", "Pick_right": "Pick right",
})
print("\nHow often is the main pick right? (confidence = chance of the most likely result)")
print(level_table.to_string())

# Every backtest match, predicted vs actual (the website's "Past rounds" page)
markets_by_match = dict(list(backtest_markets.groupby(["date", "match"])))
history = []
for _, game in backtest.iterrows():
    match_markets = markets_by_match[(game.date, f"{game.home_team} vs {game.away_team}")]
    *_, raw = summary_rows(game, match_markets, players_in_match(game))
    history.append({"date": game.date, "season": game.season, **raw})

# Tables for the website (app.py)
APP_DIR.mkdir(parents=True, exist_ok=True)
reliability.to_csv(APP_DIR / "reliability.csv", index_label="Market")
roi.to_csv(APP_DIR / "value_bets.csv", index_label="Market")
level_table.to_csv(APP_DIR / "confidence_levels.csv", index_label="Confidence")
pd.DataFrame(history).to_csv(APP_DIR / "history.csv", index=False)


# ---------------------------------------------------------------------------
# 2. MATCH CARDS
# ---------------------------------------------------------------------------

replay_mode = upcoming.empty
preview = backtest.tail(10) if replay_mode else upcoming

team_filter = sys.argv[1] if len(sys.argv) > 1 else None
if team_filter:
    preview = preview[
        preview["home_team"].str.contains(team_filter, case=False)
        | preview["away_team"].str.contains(team_filter, case=False)
    ]
    if preview.empty:
        print(f"\nNo game found for '{team_filter}'.")

print("\n" + "=" * 78)
print("REPLAY of the latest 10 games (no upcoming fixtures with odds)" if replay_mode
      else "UPCOMING MATCHES")
print("=" * 78)

all_predictions = []
summary_results, summary_stats, summary_checks = [], [], []
website_rows = []

for _, game in preview.iterrows():
    home, away = game.home_team, game.away_team
    markets = list_markets(game)
    all_predictions.append(markets)

    print(f"\n{game.date:%a %d %b %Y}   {home} vs {away}   (referee: {game.referee})")
    print("-" * 78)

    stat_table = pd.DataFrame({
        "": ["Goals", "Shots on target", "Corners", "Cards"],
        home: [game.exp_home_goals, game.exp_home_shots_on_target, game.exp_home_corners, game.exp_home_cards],
        away: [game.exp_away_goals, game.exp_away_shots_on_target, game.exp_away_corners, game.exp_away_cards],
    })
    stat_table["Total"] = stat_table[home] + stat_table[away]
    print("Expected stats (Model 1):")
    print(stat_table.to_string(index=False, float_format=lambda value: f"{value:.1f}"))

    match_players = players_in_match(game)

    result_row, stats_row, checks, raw = summary_rows(game, markets, match_players)
    summary_results.append(result_row)
    summary_stats.append(stats_row)
    summary_checks.append(checks)

    notes = assess(game, markets) + missing_player_notes(match_players)
    print("\nAssessment:")
    for note in notes:
        print(f"  - {note}")

    pick, level = main_pick(markets)

    # Same match, saved for the website
    website_rows.append({
        "mode": "replay" if replay_mode else "upcoming",
        "date": game.date,
        "referee": game.referee,
        **raw,
        "result_text": predict_result(game, markets),
        "backtest_right": level_accuracy[level],
        "likely_scores": likely_scores(game),
        "notes": "\n".join(notes),
    })

    print("\nPrediction:")
    print(f"  Result:        {predict_result(game, markets)}")
    print(f"  Confidence:    {level} (main pick right {level_accuracy[level]:.0%} of the time in backtest)")
    print(f"  Main pick:     {pick['pick']} ({pick['chance']:.0%})")
    print(f"  Likely scores: {likely_scores(game)}")

    best = best_value_bet(markets)
    if best is None:
        print("  Best bet:      none - bookmaker prices look fair")
    else:
        print(
            f"  Best bet:      {best['pick']} @ {best['bookmaker_odds']:.2f} "
            f"(model {best['chance']:.0%}, edge {best['edge']:+.0%})"
        )

    if replay_mode:
        print(f"  Actual result: {game.home_goals:.0f}-{game.away_goals:.0f}, "
              f"{game.home_corners + game.away_corners:.0f} corners, "
              f"{game.home_cards + game.away_cards:.0f} cards")

    print_goalscorers(match_players, per_team=8 if team_filter else None)

    shown = markets if team_filter else markets[markets["market"].isin(KEY_MARKETS)]
    table = shown[["market", "pick", "chance", "fair_odds", "bookmaker_odds", "edge"]].copy()
    table["chance"] = table["chance"].map("{:.0%}".format)
    table["fair_odds"] = table["fair_odds"].map("{:.2f}".format)
    table["bookmaker_odds"] = table["bookmaker_odds"].map(
        lambda odds: f"{odds:.2f}" if pd.notna(odds) else "-"
    )
    table["edge"] = table["edge"].map(lambda edge: f"{edge:+.0%}" if pd.notna(edge) else "-")
    table.columns = ["Market", "Pick", "Chance", "Fair odds", "Bookmaker", "Edge"]
    print("\nMarkets:" + ("" if team_filter else " (run with a team name to see all)"))
    print(table.to_string(index=False))

# A run for one team only would overwrite the full round with one match, so skip saving
if all_predictions and not team_filter:
    pd.concat(all_predictions, ignore_index=True).to_csv(OUTPUT_PATH, index=False)
    pd.DataFrame(website_rows).to_csv(APP_DIR / "summary.csv", index=False)
    print(f"\nAll markets saved to {OUTPUT_PATH}")


# ---------------------------------------------------------------------------
# 3. FINAL SUMMARY: every match on one line
# ---------------------------------------------------------------------------

if summary_results:
    print("\n" + "=" * 78)
    print("SUMMARY" + (" - REPLAY (actual result in brackets / last columns)" if replay_mode else ""))
    print("=" * 78)

    print("\nResult and goals:")
    print(pd.DataFrame(summary_results).to_string(index=False))

    print("\nStats and picks:")
    print(pd.DataFrame(summary_stats).to_string(index=False))

    # Confident picks: only matches where the model is 50%+ on one result
    confident = [
        row for row, check in zip(summary_results, summary_checks)
        if check["level"] != "Low"
    ]
    print(f"\nConfident picks (one result {CONFIDENT_CHANCE:.0%}+):")
    if confident:
        confident_table = pd.DataFrame([
            {
                "Match": row["Match"],
                "Pick": row["Pick"],
                "Confidence": row["Confidence"],
                "Right in backtest": f"{level_accuracy[row['Confidence']]:.0%}",
                **({"Actual": row["Actual"], "Pick right?": row["Pick right?"]} if "Actual" in row else {}),
            }
            for row in confident
        ])
        print(confident_table.to_string(index=False))
    else:
        print("None this round - every match is too close. The pick column shows the safest double chance.")

    checks = pd.DataFrame(summary_checks)
    if "pick" in checks:
        lines = []
        for _, label in CONFIDENCE_LEVELS:
            done = checks.loc[checks["level"] == label, "pick"]
            if len(done):
                lines.append(f"{label} {int(done.sum())}/{len(done)}")
        print(f"\nMain pick right: {int(checks['pick'].sum())}/{len(checks)} ({', '.join(lines)})")

        other = []
        for column, label in [("over_under", "over/under 2.5"), ("scorer", "top scorer scored")]:
            if column in checks:
                done = checks[column].dropna()
                other.append(f"{label} {int(done.sum())}/{len(done)}")
        print("Also right: " + ", ".join(other))

    print("\nHome/Draw/Away = model chances. Exp. goals, Corners, Cards = expected match totals.")
    print(f"Confidence: High = {CONFIDENCE_LEVELS[0][0]:.0%}+, Medium = {CONFIDENCE_LEVELS[1][0]:.0%}+ "
          f"on one result. Low = pick is the safest double chance (covers the draw).")
    print("Best bet = edge of 5%+ against the bookmaker (odds up to 5.0), '-' if none.")
