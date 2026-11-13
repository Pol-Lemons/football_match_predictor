import pandas as pd
import streamlit as st

from app_data import (
    apply_style, colour_table, CONFIDENCE_BADGES, MAX_VALUE_ODDS, MIN_EDGE,
    load, number_column, percent_column, short, text_column, tick,
)


st.set_page_config(page_title="Recommended bets", page_icon=":material/payments:", layout="wide")
apply_style()
st.title("Recommended bets")

summary = load("app/summary.csv")
markets = load("match_predictions.csv")
backtest_roi = load("app/value_bets.csv").set_index("Market")["ROI"]
replay = (summary["mode"] == "replay").all()
if replay:
    st.info("Replay of the latest round - the result columns show how each recommendation ended.")


# Number formats for every table on this page (a coloured table uses these)
FORMATS = {
    "Odds": "{:.2f}", "Fair odds": "{:.2f}", "Chance": "{:.0f}%", "Edge": "{:+.0f}%",
    "Profit (1 unit)": "{:+.2f}", "Backtest right": "{:.0f}%", "Exp. goals": "{:.1f}",
}


def add_to_slip(rows, label):
    picks = (rows["match"] + " | " + rows["pick"]).tolist()
    if st.button(label, disabled=not picks, key=label, type="primary"):
        slip = st.session_state.setdefault("bet_slip", [])
        slip.extend(pick for pick in picks if pick not in slip)
        st.success(f"Bet slip: {len(slip)} pick(s).")


# ---------------------------------------------------------------------------
# 1. VALUE BETS: the model's chance x bookmaker odds > 1.05
# ---------------------------------------------------------------------------

st.subheader("1. Value bets - best for profit")
st.caption(
    f"The model thinks the bookmaker's price is too high: edge {MIN_EDGE:.0%}+ and odds up to "
    f"{MAX_VALUE_ODDS:.1f}. They lose more often than safe picks, but pay more when they win."
)

value = markets[(markets["edge"] >= MIN_EDGE) & (markets["bookmaker_odds"] <= MAX_VALUE_ODDS)]
value = value.sort_values("edge", ascending=False)
if value.empty:
    st.write("No value bets this round - the bookmaker's prices look fair.")
else:
    group = value["market"].str.replace(r" [0-9.]+$", "", regex=True)
    table = pd.DataFrame({
        "Match": value["match"].map(short),
        "Market": value["market"],
        "Pick": value["pick"].map(short),
        "Odds": value["bookmaker_odds"],
        "Chance": value["chance"] * 100,
        "Edge": value["edge"] * 100,
        "Backtest ROI": group.map(backtest_roi).fillna("-"),
    })
    config = {
        "Match": text_column("Match"),
        "Odds": number_column("Odds", decimals=2),
        "Chance": percent_column("Model chance", bar=True),
        "Edge": percent_column("Edge", help="Expected profit per bet at these odds"),
        "Backtest ROI": text_column("Backtest ROI", help="Profit per bet for this market on past matches"),
    }
    if replay:
        table["Won"] = value["profit"].map(lambda profit: tick(profit > 0))
        table["Profit (1 unit)"] = value["profit"]
        config["Won"] = text_column("Won")
        config["Profit (1 unit)"] = number_column("Profit (1 unit)", decimals=2)
    st.dataframe(colour_table(table, FORMATS), hide_index=True, column_config=config)
    add_to_slip(value, "Add value bets to bet slip")


# ---------------------------------------------------------------------------
# 2. SAFE PICKS: matches where one result has 50%+
# ---------------------------------------------------------------------------

st.subheader("2. Safe picks - most likely to win")
st.caption("Matches where the model gives one result 50% or more. Lower odds, but right more often.")

safe = summary[summary["confidence"] != "Low"].sort_values("pick_chance", ascending=False)
if safe.empty:
    st.write("No confident picks this round - every match is too close.")
else:
    table = pd.DataFrame({
        "Match": safe["match"].map(short),
        "Confidence": safe["confidence"].map(CONFIDENCE_BADGES),
        "Pick": safe["pick"].map(short),
        "Chance": safe["pick_chance"] * 100,
        "Odds": safe["pick_odds"],
        "Edge": safe["pick_edge"] * 100,
        "Backtest right": safe["backtest_right"] * 100,
    })
    config = {
        "Match": text_column("Match"),
        "Confidence": text_column("Confidence"),
        "Chance": percent_column("Model chance", bar=True),
        "Odds": number_column("Odds", decimals=2),
        "Edge": percent_column("Edge", help="Negative = the price is lower than the model's fair odds"),
        "Backtest right": percent_column("Backtest right", help="How often picks at this confidence were right"),
    }
    if replay:
        table["Result"] = safe["actual"]
        table["Right"] = safe["pick_right"].map(tick)
        config["Result"] = text_column("Result")
        config["Right"] = text_column("Right")
    st.dataframe(colour_table(table, FORMATS, heat=["Backtest right"]), hide_index=True, column_config=config)
    add_to_slip(safe, "Add safe picks to bet slip")


# ---------------------------------------------------------------------------
# 3. CLOSE GAMES: safest double chance
# ---------------------------------------------------------------------------

close = summary[summary["confidence"] == "Low"].sort_values("pick_chance", ascending=False)
with st.expander(f"3. Close games - safest double chance ({len(close)})"):
    st.caption("No result reaches 50%, so the pick covers two of the three results. "
               "Bookmakers rarely list these odds in our data, so the fair odds are shown.")
    table = pd.DataFrame({
        "Match": close["match"].map(short),
        "Pick": close["pick"].map(short),
        "Chance": close["pick_chance"] * 100,
        "Fair odds": 1 / close["pick_chance"],
        "Backtest right": close["backtest_right"] * 100,
    })
    config = {
        "Match": text_column("Match"),
        "Chance": percent_column("Model chance", bar=True),
        "Fair odds": number_column("Fair odds", decimals=2),
        "Backtest right": percent_column("Backtest right"),
    }
    if replay:
        table["Result"] = close["actual"]
        table["Right"] = close["pick_right"].map(tick)
        config["Result"] = text_column("Result")
        config["Right"] = text_column("Right")
    st.dataframe(colour_table(table, FORMATS, heat=["Backtest right"]), hide_index=True, column_config=config)
    add_to_slip(close, "Add double chances to bet slip")

# ---------------------------------------------------------------------------
# 4. BOTH TEAMS TO SCORE
# ---------------------------------------------------------------------------

st.subheader("4. Both teams to score")
btts_backtest = load("app/history.csv")["btts_right"].mean()
st.caption(
    f"Yes if the model gives both teams scoring 50%+, otherwise No. Right {btts_backtest:.0%} of the time "
    "in the backtest - the model's chances stay close to 50/50, so the strongest picks are at the top."
)
btts = summary.assign(
    btts_pick=summary["both_score"].map(lambda chance: "Yes" if chance >= 0.5 else "No"),
    btts_chance=summary["both_score"].map(lambda chance: max(chance, 1 - chance)),
).sort_values("btts_chance", ascending=False)
table = pd.DataFrame({
    "Match": btts["match"].map(short),
    "Pick": btts["btts_pick"],
    "Chance": btts["btts_chance"] * 100,
    "Fair odds": 1 / btts["btts_chance"],
    "Exp. goals": btts["exp_goals"],
})
config = {
    "Match": text_column("Match"),
    "Pick": text_column("BTTS"),
    "Chance": percent_column("Model chance", bar=True),
    "Fair odds": number_column("Fair odds", decimals=2),
    "Exp. goals": number_column("Exp. goals"),
}
if replay:
    table["Result"] = btts["actual"]
    table["Right"] = btts["btts_right"].map(tick)
    config["Result"] = text_column("Result")
    config["Right"] = text_column("Right")
st.dataframe(colour_table(table, FORMATS, heat=["Exp. goals"]), hide_index=True, column_config=config)
add_to_slip(btts.assign(pick="Both teams score: " + btts["btts_pick"]), "Add BTTS picks to bet slip")

st.page_link("pages/4_Bet_calculator.py", label="Open the bet calculator", icon=":material/calculate:")
