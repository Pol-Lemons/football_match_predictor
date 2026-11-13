import streamlit as st

from app_data import apply_style, colour_table, load


st.set_page_config(page_title="Model performance", page_icon=":material/insights:", layout="wide")
apply_style()
st.title("Model performance")

history = load("app/history.csv")
st.caption(
    f"Backtest on {len(history)} matches ({history['date'].min():%b %Y} to {history['date'].max():%b %Y}). "
    "Each season was predicted by models trained only on earlier seasons."
)

right = history["pick_right"].astype(bool)
confident = history["confidence"] != "Low"
columns = st.columns(4)
columns[0].metric("Main pick right", f"{right.mean():.0%}")
columns[1].metric("Confident picks right", f"{right[confident].mean():.0%}",
                  help=f"{confident.sum()} matches where one result was 50%+")
columns[2].metric("Over/Under 2.5 right", f"{history['over_under_right'].astype(bool).mean():.0%}")
columns[3].metric("Both teams to score right", f"{history['btts_right'].astype(bool).mean():.0%}",
                  help="Always picking 'Yes' would be right 55% of the time - the model barely beats it")

st.subheader("Main pick by confidence level")
st.dataframe(colour_table(load("app/confidence_levels.csv"), heat=["Result right", "Pick right"]),
             hide_index=True)
st.caption("Confidence = model chance of the most likely result. Low-confidence matches use the safest double chance.")

left, right_column = st.columns(2)
with left:
    st.subheader("When the model says 60%+")
    st.dataframe(colour_table(load("app/reliability.csv"), heat=["Happened"]), hide_index=True)
    st.caption("How often a pick happened when the model gave it 60% or more.")
with right_column:
    st.subheader("Value bets")
    st.dataframe(colour_table(load("app/value_bets.csv"), heat=["Win"]), hide_index=True)
    st.caption("1 unit on every bet with 5%+ edge (odds up to 5.0). ROI = profit per unit staked.")

left, right_column = st.columns(2)
with left:
    st.subheader("Match stats (Model 1)")
    st.dataframe(colour_table(load("app/stats_accuracy.csv")), hide_index=True)
    st.caption("How much better than guessing the league average.")
with right_column:
    st.subheader("Goalscorers")
    st.dataframe(colour_table(load("app/goalscorer_calibration.csv"), heat=["Scored"]), hide_index=True)
    st.caption("When the model gives a player this chance, how often he scored.")
