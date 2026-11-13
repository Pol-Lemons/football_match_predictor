import pandas as pd
import streamlit as st

from app_data import apply_style, colour_table, CONFIDENCE_BADGES, load, short, text_column, tick


st.set_page_config(page_title="Past rounds", page_icon=":material/history:", layout="wide")
apply_style()
st.title("Past rounds")
st.caption("Every backtest match: what the model predicted before kick-off, and what happened.")

history = load("app/history.csv")
# shortcut: groups matches by calendar week, not by official gameweek (midweek rounds can merge with weekends)
history["round"] = history["date"].dt.to_period("W").dt.start_time

season = st.selectbox("Season", sorted(history["season"].unique(), reverse=True))
season_matches = history[history["season"] == season]

weekly = season_matches.groupby("round")["pick_right"].mean().mul(100).rename("Main pick right (%)")
st.subheader("Main pick right, week by week")
st.line_chart(weekly)
st.metric("Season so far", f"{season_matches['pick_right'].mean():.0%}", help=f"{len(season_matches)} matches")

rounds = sorted(season_matches["round"].unique(), reverse=True)
chosen = st.selectbox("Week", rounds, format_func=lambda week: f"Week of {week:%d %b %Y}")
week = season_matches[season_matches["round"] == chosen]

st.write(f"**{week['pick_right'].sum()}/{len(week)}** main picks right this week.")
st.dataframe(
    colour_table(pd.DataFrame({
        "Date": week["date"].dt.strftime("%a %d %b"),
        "Match": week["match"].map(short),
        "Confidence": week["confidence"].map(CONFIDENCE_BADGES),
        "Pick": week["pick"].map(short),
        "Likely score": week["likely_score"],
        "Result": week["actual"],
        "Pick right": week["pick_right"].map(tick),
        "O/U 2.5 right": week["over_under_right"].map(tick),
        "BTTS right": week["btts_right"].map(tick),
        "Top scorer": week["top_scorer"].fillna("-"),
        "Scored": week["top_scorer_scored"].map(tick),
    })),
    hide_index=True,
    column_config={
        column: text_column(column)
        for column in ["Date", "Match", "Confidence", "Likely score", "Result", "Pick right", "O/U 2.5 right", "BTTS right", "Scored"]
    },
)
