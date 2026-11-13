import pandas as pd
import streamlit as st

from app_data import (
    CONFIDENCE_BADGES, LIST_MOTION, MAX_VALUE_ODDS, MIN_EDGE, animate, apply_style, colour_table,
    fixture_list, load, number_column, page_header, percent_column, short, text_column, tick,
)


# Website home page: the round at a glance.
# Run with:  streamlit run app.py

st.set_page_config(page_title="Matchday · PL Predictor", page_icon=":material/sports_soccer:", layout="wide")
apply_style()

summary = load("app/summary.csv")
markets = load("match_predictions.csv")
replay = (summary["mode"] == "replay").all()

value_bets = markets[(markets["edge"] >= MIN_EDGE) & (markets["bookmaker_odds"] <= MAX_VALUE_ODDS)]
kpis = [
    ("Matches", len(summary), "", "Matches in this round"),
    ("Confident picks", int((summary["confidence"] != "Low").sum()), "", "One result has a 50%+ chance"),
    ("Value bets", len(value_bets), "", "Edge of 5%+ against the bookmaker"),
]
if replay:
    kpis.append(("Picks right", int(summary["pick_right"].sum()), f"/{len(summary)}", "Main picks that came true"))

last_day = summary["date"].max()
season_start = last_day.year if last_day.month >= 7 else last_day.year - 1  # seasons start in August
dates = f"{summary['date'].min():%a %d %b} – {last_day:%a %d %b %Y}"
page_header(
    f"Premier League · {season_start}/{(season_start + 1) % 100:02d}",
    "Last round" if replay else "This round",
    f"{dates} · predicted before kick-off, shown next to the final scores" if replay
    else f"{dates} · model predictions for every match",
    kpis,
)
st.page_link("pages/0_Recommended_bets.py", label="Recommended bets for this round",
             icon=":material/arrow_forward:")

animate(fixture_list(summary), LIST_MOTION)


# ---------------------------------------------------------------------------
# DETAIL TABLES
# ---------------------------------------------------------------------------

st.subheader("Prediction table")
st.caption("Select a row to open the match.")

predictions = pd.DataFrame({
    "Date": summary["date"].dt.strftime("%a %d %b"),
    "Match": summary["match"].map(short),
    "Home": summary["home_chance"] * 100,
    "Draw": summary["draw_chance"] * 100,
    "Away": summary["away_chance"] * 100,
    "Confidence": summary["confidence"].map(CONFIDENCE_BADGES),
    "Pick": summary["pick"].map(short),
    "Chance": summary["pick_chance"] * 100,
    "Score": summary["likely_score"],
    "Goals": summary["exp_goals"],
    "Over 2.5": summary["over_2_5"] * 100,
    "BTTS": summary["both_score"].map(lambda chance: "Yes" if chance >= 0.5 else "No"),
    "BTTS Yes": summary["both_score"] * 100,
})
if replay:
    predictions["Result"] = summary["actual"]
    predictions["Pick right"] = summary["pick_right"].map(tick)
    predictions["BTTS right"] = summary["btts_right"].map(tick)

selected = st.dataframe(
    colour_table(
        predictions,
        {**dict.fromkeys(["Home", "Draw", "Away", "Chance", "Over 2.5", "BTTS Yes"], "{:.0f}%"), "Goals": "{:.1f}"},
        heat=["Chance"],
    ),
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    column_config={
        "Date": text_column("Date"),
        "Match": text_column("Match"),
        "Home": percent_column("Home", bar=True, help="Model chance of a home win"),
        "Draw": percent_column("Draw", bar=True),
        "Away": percent_column("Away", bar=True),
        "Confidence": text_column("Confidence", help="High = one result 60%+, Medium = 50%+, Low = too close"),
        "Pick": st.column_config.TextColumn("Main pick", help="Most likely result, or the safest double chance when the game is close"),
        "Chance": percent_column("Chance"),
        "Score": text_column("Likely score"),
        "Goals": number_column("Exp. goals"),
        "Over 2.5": percent_column("Over 2.5"),
        "BTTS": text_column("BTTS", help="Both teams to score: the model's pick"),
        "BTTS Yes": percent_column("BTTS Yes", help="Model chance that both teams score"),
        "BTTS right": text_column("BTTS right"),
        "Result": text_column("Result"),
        "Pick right": text_column("Pick right"),
    },
)
if selected.selection.rows:
    st.session_state["match"] = summary["match"].iloc[selected.selection.rows[0]]
    st.switch_page("pages/1_Match_details.py")

st.subheader("Stats and picks")
stats = pd.DataFrame({
    "Match": summary["match"].map(short),
    "Corners": summary["exp_corners"],
    "Cards": summary["exp_cards"],
    "Top scorer": summary["top_scorer"],
    "Scorer chance": summary["top_scorer_chance"] * 100,
    "Best bet": summary["best_bet"].map(short).fillna("-"),
    "Odds": summary["best_bet_odds"],
    "Edge": summary["best_bet_edge"] * 100,
})
stats_config = {
    "Match": text_column("Match"),
    "Corners": number_column("Exp. corners"),
    "Cards": number_column("Exp. cards"),
    "Scorer chance": percent_column("Chance"),
    "Odds": number_column("Odds", decimals=2),
    "Edge": percent_column("Edge", help="Expected profit at the bookmaker's odds"),
}
if replay:
    stats.insert(2, "Real corners", summary["actual_corners"])
    stats.insert(4, "Real cards", summary["actual_cards"])
    stats["Scored"] = summary["top_scorer_scored"].map(tick)
    stats["Bet won"] = summary["best_bet_profit"].map(lambda profit: tick(profit > 0) if pd.notna(profit) else "")
    stats_config.update({
        "Real corners": number_column("Real corners", decimals=0),
        "Real cards": number_column("Real cards", decimals=0),
        "Scored": text_column("Scored"),
        "Bet won": text_column("Bet won"),
    })
stats_formats = {
    "Corners": "{:.1f}", "Cards": "{:.1f}", "Real corners": "{:.0f}", "Real cards": "{:.0f}",
    "Scorer chance": "{:.0f}%", "Odds": "{:.2f}", "Edge": "{:+.0f}%",
}
st.dataframe(colour_table(stats, stats_formats, heat=["Scorer chance"]), hide_index=True, column_config=stats_config)

st.caption(
    "Chances come from the model. Exp. = expected match total. Best bet = 5%+ edge against "
    "the bookmaker (odds up to 5.0). Team codes: ARS = Arsenal, MUN = Man United, NFO = Nott'm Forest, ..."
)
