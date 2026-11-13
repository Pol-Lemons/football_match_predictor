import pandas as pd
import streamlit as st

from app_data import (
    ACCENT, BARS_MOTION, CONFIDENCE_BADGES, DATA_DIR, HEADER_MOTION, MATCH_CSS, animate, apply_style, chance_bar,
    colour_table, load, match_header, number_column, percent_column, player_rating, rating_colour, read, short,
    stat_bars, team_colours, text_column, tick,
)


st.set_page_config(page_title="Match details", page_icon=":material/sports_soccer:", layout="wide")
apply_style()
st.markdown(MATCH_CSS, unsafe_allow_html=True)

summary = load("app/summary.csv")
stats = load("match_stats.csv")
markets = load("match_predictions.csv")
players = load("player_predictions.csv")
results = load("premier_league_matches.csv")
player_matches = load("../external/fpl_player_matches.csv")
fpl_players = load("../external/fpl_players.csv")
# Seasons 2000/01-2020/21 for head-to-head (fetch_h2h_history.py); optional
old_path = DATA_DIR.parent / "external" / "h2h_history.csv"
old_results = read(old_path) if old_path.exists() else results.iloc[:0]

# Opened from a fixture link (?match=...) or a table row on the home page
options = summary["match"].tolist()
chosen = st.session_state.get("match") or st.query_params.get("match")
match_name = st.selectbox("Match", options, index=options.index(chosen) if chosen in options else 0)
st.session_state["match"] = match_name
st.query_params["match"] = match_name  # the address always points at the match on screen

row = summary[summary["match"] == match_name].iloc[0]
home, away = row["home_team"], row["away_team"]

played = pd.notna(row.get("actual"))  # only known when the round has been played (replay)
if played:
    score = row["actual"].replace("-", " - ")
    label = f"Full time · predicted {row['likely_score']}"
else:
    score = row["likely_score"].replace("-", " - ")
    label = "Predicted score"
animate(
    match_header(home, away, score, label, f"{row['date']:%a %d %b %Y} · Referee: {row['referee']}"),
    HEADER_MOTION,
)

prediction_tab, stats_tab, h2h_tab, odds_tab, scorers_tab = st.tabs(
    ["Prediction", "Statistics", "Head-to-head", "Odds & markets", "Goalscorers"]
)


# ---------------------------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------------------------

with prediction_tab, st.container(key="overview"):
    columns = st.columns([3, 1], vertical_alignment="center")
    with columns[0]:
        chance_bar(home, away, row["home_chance"], row["draw_chance"], row["away_chance"])
    columns[1].metric("Confidence", CONFIDENCE_BADGES[row["confidence"]],
                      help=f"Picks at this level were right {row['backtest_right']:.0%} in the backtest")

    columns = st.columns(4)
    columns[0].metric("Main pick", short(row["pick"]), f"{row['pick_chance']:.0%} chance", delta_color="off")
    columns[1].metric("Likely score", row["likely_score"])
    if played:
        columns[2].metric("Real result", row["actual"], f"pick {tick(row['pick_right'])}", delta_color="off")
    if pd.notna(row["best_bet"]):
        columns[3].metric("Best bet", short(row["best_bet"]),
                          f"@ {row['best_bet_odds']:.2f}, edge {row['best_bet_edge']:+.0%}", delta_color="off")
    columns = st.columns(4)
    btts_pick = "Yes" if row["both_score"] >= 0.5 else "No"
    columns[0].metric("Both teams to score", btts_pick,
                      f"{max(row['both_score'], 1 - row['both_score']):.0%} chance"
                      + (f" {tick(row['btts_right'])}" if played else ""), delta_color="off")
    over_pick = "Over 2.5" if row["over_2_5"] >= 0.5 else "Under 2.5"
    columns[1].metric("Goals", over_pick,
                      f"{max(row['over_2_5'], 1 - row['over_2_5']):.0%} chance"
                      + (f" {tick(row['over_under_right'])}" if played else ""), delta_color="off")
    columns[2].metric("Exp. corners", f"{row['exp_corners']:.1f}",
                      f"real {row['actual_corners']:.0f}" if played else None, delta_color="off")
    columns[3].metric("Exp. cards", f"{row['exp_cards']:.1f}",
                      f"real {row['actual_cards']:.0f}" if played else None, delta_color="off")
    st.caption(f"{row['result_text']} · Likely scores: {row['likely_scores']}")

    game = stats[
        (stats["date"] == row["date"]) & (stats["home_team"] == home) & (stats["away_team"] == away)
    ].iloc[0]
    stat_names = {"goals": "Goals", "shots_on_target": "Shots on target", "corners": "Corners", "cards": "Cards"}
    stat_table = pd.DataFrame({
        "Stat": list(stat_names.values()),
        short(home): [game[f"exp_home_{stat}"] for stat in stat_names],
        short(away): [game[f"exp_away_{stat}"] for stat in stat_names],
    })
    stat_table["Total"] = stat_table[short(home)] + stat_table[short(away)]

    left, right = st.columns(2)
    with left:
        st.subheader("Expected stats")
        teams = [short(home), short(away)]
        st.dataframe(
            stat_table.style.format("{:.1f}", subset=teams + ["Total"]).highlight_max(
                axis=1, subset=teams, props=f"color:{ACCENT}; font-weight:700"
            ),
            hide_index=True, column_config={
            "Stat": text_column("Stat"),
            **{
                column: st.column_config.NumberColumn(column, format="%.1f", alignment="center")
                for column in [short(home), short(away), "Total"]
            },
        })
    with right:
        st.subheader("Assessment")
        st.markdown("\n".join(f"- {note}" for note in row["notes"].split("\n")))


# ---------------------------------------------------------------------------
# MATCH STATS AND PLAYER RATINGS (used by the Statistics and Head-to-head tabs)
# ---------------------------------------------------------------------------

def show_match_stats(match):
    colours = team_colours(match["home_team"], match["away_team"])
    animate(stat_bars([
        (label, match[f"home_{column}"], match[f"away_{column}"], 0)
        for label, column in [
            ("Goals", "goals"), ("Total shots", "shots"), ("Shots on target", "shots_on_target"),
            ("Corners", "corners"), ("Fouls", "fouls"),
            ("Yellow cards", "yellow_cards"), ("Red cards", "red_cards"),
        ]
    ], *colours), BARS_MOTION)


def show_players(date, home_team, away_team):
    lineup = player_matches[
        (player_matches["date"] == date) & (player_matches["home_team"] == home_team)
        & (player_matches["away_team"] == away_team) & (player_matches["minutes"] > 0)
    ].merge(fpl_players[["player_id", "player", "position"]], on="player_id")
    if lineup.empty:
        st.caption("Player ratings and actions are only available for 2026/27 matches (FPL data).")
        return

    lineup["rating"] = player_rating(lineup["bps"], lineup["minutes"])
    best = lineup.loc[lineup["rating"].idxmax()]
    st.markdown(f"**Player of the match:** {best['player']} ({best['team']}) · {best['rating']:.1f}")

    for team in [home_team, away_team]:
        team_players = lineup[lineup["team"] == team].sort_values("rating", ascending=False)
        table = pd.DataFrame({
            "Player": team_players["player"],
            "Pos": team_players["position"],
            "Min": team_players["minutes"],
            "Rating": team_players["rating"],
            "Goals": team_players["goals"],
            "Assists": team_players["assists"],
            "xG": team_players["expected_goals"],
            "xA": team_players["expected_assists"],
            "Tackles": team_players["tackles"],
            "Blocks & int.": team_players["clearances_blocks_interceptions"],
            "Recoveries": team_players["recoveries"],
            "Saves": team_players["saves"],
            "Cards": team_players["yellow_cards"].map(lambda count: "🟨" * count)
                     + team_players["red_cards"].map(lambda count: "🟥" * count),
        })
        st.write(f"**{team}**")
        st.dataframe(
            colour_table(
                table, {"Rating": "{:.1f}", "xG": "{:.2f}", "xA": "{:.2f}"},
                heat=["Goals", "Assists", "xG", "xA", "Tackles"],
            ).map(rating_colour, subset=["Rating"]),
            hide_index=True,
            column_config={column: text_column(column) for column in table.columns if column != "Player"},
        )
    st.caption(
        "Rating 0-10 is our own score, made from FPL's Bonus Points System (BPS): an average starter "
        "gets about 6.6, the best 10% get 8.6+. Players under 10 minutes have no rating. "
        "Blocks & int. = clearances, blocks and interceptions."
    )


# ---------------------------------------------------------------------------
# STATISTICS: the model's expected stats before kick-off, the real ones after
# ---------------------------------------------------------------------------

with stats_tab:
    real = results[
        (results["date"] == row["date"]) & (results["home_team"] == home) & (results["away_team"] == away)
    ]
    if real.empty:
        st.subheader("Expected stats")
        animate(stat_bars([
            ("Goals", game["exp_home_goals"], game["exp_away_goals"], 1),
            ("Shots on target", game["exp_home_shots_on_target"], game["exp_away_shots_on_target"], 1),
            ("Corners", game["exp_home_corners"], game["exp_away_corners"], 1),
            ("Cards", game["exp_home_cards"], game["exp_away_cards"], 1),
        ], *team_colours(home, away)), BARS_MOTION)
        st.caption("The model's expectation. After the match, the real stats replace it.")
    else:
        st.subheader("Match stats")
        show_match_stats(real.iloc[0])
        st.caption("Bars use each club's colours. Possession is not in our data source (Football-Data.co.uk).")
        st.subheader("Player ratings")
        show_players(row["date"], home, away)


# ---------------------------------------------------------------------------
# HEAD-TO-HEAD: earlier league meetings since 2000/01 - click one for its details
# ---------------------------------------------------------------------------

with h2h_tab, st.container(key="h2h"):
    all_results = pd.concat([old_results, results], ignore_index=True)
    meetings = all_results[
        (all_results["date"] < row["date"])
        & (
            ((all_results["home_team"] == home) & (all_results["away_team"] == away))
            | ((all_results["home_team"] == away) & (all_results["away_team"] == home))
        )
    ].sort_values("date", ascending=False).reset_index(drop=True)
    since = f"{all_results['date'].min():%B %Y}"

    if meetings.empty:
        st.write(f"No Premier League meetings in our data (since {since}).")
    else:
        home_goals = meetings["home_goals"].where(meetings["home_team"] == home, meetings["away_goals"])
        away_goals = meetings["away_goals"].where(meetings["home_team"] == home, meetings["home_goals"])
        total_goals = meetings["home_goals"] + meetings["away_goals"]

        columns = st.columns(3)
        columns[0].metric(f"{short(home)} wins", int((home_goals > away_goals).sum()))
        columns[1].metric("Draws", int((home_goals == away_goals).sum()))
        columns[2].metric(f"{short(away)} wins", int((home_goals < away_goals).sum()))
        st.caption(
            f"{len(meetings)} meetings since {since} · {total_goals.mean():.1f} goals per game · "
            f"over 2.5 in {(total_goals > 2.5).mean():.0%} · both teams scored in "
            f"{((meetings['home_goals'] > 0) & (meetings['away_goals'] > 0)).mean():.0%}"
        )

        score = meetings["home_goals"].astype(int).astype(str) + " - " + meetings["away_goals"].astype(int).astype(str)
        half_time = (meetings["home_goals_half_time"].astype(int).astype(str) + " - "
                     + meetings["away_goals_half_time"].astype(int).astype(str))
        table = pd.DataFrame({
            "Date": meetings["date"].dt.strftime("%d %b %Y"),
            "Home": meetings["home_team"],
            "Score": score,
            "Away": meetings["away_team"],
            "Half time": half_time,
            f"{short(home)} result": [
                "W" if goals_for > goals_against else "D" if goals_for == goals_against else "L"
                for goals_for, goals_against in zip(home_goals, away_goals)
            ],
        })
        st.caption("Click a meeting to see its match stats (and player ratings for 2026/27 matches).")
        selected = st.dataframe(
            colour_table(table), hide_index=True, height=min(36 * len(table) + 38, 390),
            on_select="rerun", selection_mode="single-row", key=f"h2h_{match_name}",
            column_config={column: text_column(column) for column in table.columns},
        )

        # The latest meeting is shown until another one is clicked
        meeting = meetings.iloc[selected.selection.rows[0] if selected.selection.rows else 0]
        animate(match_header(
            meeting["home_team"], meeting["away_team"], score[meeting.name],
            f"Full time · half time {half_time[meeting.name]}",
            f"{meeting['date']:%a %d %b %Y} · Referee: {meeting['referee']}",
        ), HEADER_MOTION)
        show_match_stats(meeting)
        show_players(meeting["date"], meeting["home_team"], meeting["away_team"])


# ---------------------------------------------------------------------------
# ODDS: every market, select rows to add them to the bet slip
# ---------------------------------------------------------------------------

with odds_tab:
    match_markets = markets[(markets["date"] == row["date"]) & (markets["match"] == match_name)]
    groups = ["All", *match_markets["market"].str.replace(r" [0-9.]+$", "", regex=True).unique()]
    group = st.radio("Market", groups, horizontal=True)
    if group != "All":
        match_markets = match_markets[match_markets["market"].str.startswith(group)]

    odds_table = pd.DataFrame({
        "Market": match_markets["market"],
        "Pick": match_markets["pick"].map(short),
        "Chance": match_markets["chance"] * 100,
        "Fair odds": match_markets["fair_odds"],
        "Bookmaker": match_markets["bookmaker_odds"],
        "Edge": match_markets["edge"] * 100,
    })
    selected = st.dataframe(
        colour_table(odds_table, {"Chance": "{:.0f}%", "Fair odds": "{:.2f}", "Bookmaker": "{:.2f}", "Edge": "{:+.0f}%"}),
        hide_index=True, height=480,
        on_select="rerun", selection_mode="multi-row", key=f"odds_{match_name}_{group}",
        column_config={
            "Chance": percent_column("Model chance", bar=True),
            "Fair odds": number_column("Fair odds", decimals=2),
            "Bookmaker": number_column("Bookmaker", decimals=2),
            "Edge": percent_column("Edge"),
        },
    )
    st.caption(
        "Fair odds = odds where you break even in the long run. Edge = expected profit at the "
        "bookmaker's odds (5%+ is a value bet). Select rows to add them to the bet slip."
    )

    picks = [f"{match_name} | {pick}" for pick in match_markets["pick"].iloc[selected.selection.rows]]
    if st.button("Add to bet slip", disabled=not picks, type="primary"):
        slip = st.session_state.setdefault("bet_slip", [])
        slip.extend(pick for pick in picks if pick not in slip)
        st.success(f"Bet slip: {len(slip)} pick(s).")
    st.page_link("pages/4_Bet_calculator.py", label="Open the bet calculator", icon=":material/calculate:")


# ---------------------------------------------------------------------------
# GOALSCORERS
# ---------------------------------------------------------------------------

with scorers_tab:
    match_players = players[
        (players["date"] == row["date"]) & (players["home_team"] == home) & (players["away_team"] == away)
    ]
    if match_players.empty:
        st.write("No player data for this match.")
    else:
        missing = match_players[(match_players["availability"] < 0.75) & (match_players["usual_minutes"] >= 45)]
        if not missing.empty:
            st.warning("Missing: " + ", ".join(f"{p.player} ({p.team}) - {p.news}" for p in missing.itertuples()))

        scorer_columns = st.columns(2)
        for column, team in zip(scorer_columns, [home, away]):
            team_players = match_players[match_players["team"] == team].nlargest(10, "chance")
            table = pd.DataFrame({
                "Player": team_players["player"] + team_players["penalties_order"].eq(1).map({True: " (pens)", False: ""}),
                "Pos": team_players["position"],
                "Minutes": team_players["expected_minutes"],
                "Chance": team_players["chance"] * 100,
                "Fair odds": 1 / team_players["chance"],
            })
            if team_players["scored"].notna().any():
                table["Scored"] = team_players["scored"].map(tick)
            column.write(f"**{team}**")
            column.dataframe(colour_table(
                table, {"Minutes": "{:.0f}", "Chance": "{:.0f}%", "Fair odds": "{:.2f}"},
            ), hide_index=True, column_config={
                "Pos": text_column("Pos"),
                "Minutes": number_column("Minutes", decimals=0),
                "Chance": percent_column("Chance", bar=True),
                "Fair odds": number_column("Fair odds", decimals=2),
                "Scored": text_column("Scored"),
            })
        st.caption("Chance to score at least once. Minutes = expected minutes from the last 5 matches and injury news.")
