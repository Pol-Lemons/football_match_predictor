import numpy as np
import pandas as pd
import streamlit as st

from app_data import apply_style, colour_table, load, short


st.set_page_config(page_title="Bet calculator", page_icon=":material/calculate:", layout="wide")
apply_style()
st.title("Bet calculator")
st.caption(
    "Choose picks and a stake to see cash in, cash out and what the model expects. "
    "**Singles** = each pick is its own bet. **Parlay** = one ticket with every pick; all must win."
)

markets = load("match_predictions.csv")
markets["key"] = markets["match"] + " | " + markets["pick"]

# The slip is shared with the "Odds & markets" tab of the match page
saved = [key for key in st.session_state.get("bet_slip", []) if key in set(markets["key"])]
slip = st.multiselect(
    "Bet slip", markets["key"], default=saved, format_func=short,
    placeholder="Choose picks here, or add them from a match's Odds & markets tab",
)
st.session_state["bet_slip"] = slip
if not slip:
    st.stop()

stake = st.number_input("Stake (per single / per parlay ticket)", min_value=0.0, value=100_000.0,
                        step=10_000.0, format="%.0f")

picked = markets.set_index("key").loc[slip].reset_index()
# Share of the stake refunded on average (Asian handicap pushes); 0 for other markets
picked["refund_chance"] = (1 - picked["fair_odds"] * picked["chance"]).clip(lower=0).fillna(0)

st.subheader("Odds")
edited = st.data_editor(
    pd.DataFrame({
        "Pick": picked["key"].map(short),
        "Odds": picked["bookmaker_odds"].fillna(picked["fair_odds"]).round(2),
        "Model chance": picked["chance"].map("{:.0%}".format),
    }),
    hide_index=True,
    disabled=["Pick", "Model chance"],
    column_config={"Odds": st.column_config.NumberColumn(min_value=1.01, step=0.01, format="%.2f")},
)
st.caption("Odds start at the bookmaker's average price, or the model's fair odds when there is none. "
           "Change them to your own bookmaker's price.")

odds = edited["Odds"].to_numpy(dtype=float)
chance = picked["chance"].to_numpy()
# Expected return per 1 staked: paid at the odds when it wins, refunded on a push
expected_return = odds * chance + picked["refund_chance"].to_numpy()

played = picked["won"].notna().all() if "won" in picked else False
if played:
    # Actual return per 1 staked: win share x odds + refunded share (half wins / pushes on handicaps)
    refund = (picked["profit"] + 1 - picked["bookmaker_odds"] * picked["won"]).fillna(0)
    actual_return = odds * picked["won"].to_numpy() + refund.to_numpy()


def money(value):
    return f"{value:,.0f}"


singles_tab, parlay_tab = st.tabs(["Singles", "Parlay"])

with singles_tab:
    singles = pd.DataFrame({
        "Pick": picked["key"].map(short),
        "Odds": odds,
        "Cash in": stake,
        "Cash out if win": stake * odds,
        "Profit if win": stake * (odds - 1),
        "Expected profit": stake * (expected_return - 1),
    })
    if played:
        singles["Actual cash out"] = stake * actual_return

    columns = st.columns(4)
    columns[0].metric("Total cash in", money(stake * len(picked)))
    columns[1].metric("Cash out if all win", money(singles["Cash out if win"].sum()))
    columns[2].metric("Expected profit", money(singles["Expected profit"].sum()),
                      help="Average result if these bets were placed many times, using the model's chances")
    if played:
        columns[3].metric("Actual profit", money(singles["Actual cash out"].sum() - stake * len(picked)))

    money_columns = [column for column in singles.columns if column not in ("Pick", "Odds")]
    st.dataframe(colour_table(singles, {"Odds": "{:.2f}", **{column: money for column in money_columns}}),
                 hide_index=True)

with parlay_tab:
    matches = picked["match"]
    if len(picked) < 2:
        st.info("A parlay needs at least 2 picks.")
    elif matches.duplicated().any():
        st.warning("A parlay can only have one pick per match: "
                   + ", ".join(matches[matches.duplicated()].unique()))
    else:
        combined_odds = odds.prod()
        win_chance = chance.prod()  # assumes the matches are independent
        columns = st.columns(3)
        columns[0].metric("Combined odds", f"{combined_odds:.2f}")
        columns[1].metric("Cash in", money(stake))
        columns[2].metric("Cash out if all win", money(stake * combined_odds))

        columns = st.columns(3)
        columns[0].metric("Profit if all win", money(stake * (combined_odds - 1)))
        columns[1].metric("Model chance all win", f"{win_chance:.1%}")
        columns[2].metric("Expected profit", money(stake * (expected_return.prod() - 1)),
                          help="Average result over many tickets, using the model's chances")

        if played:
            cash_out = stake * actual_return.prod()
            if cash_out > stake:
                st.success(f"Ticket won: cash out {money(cash_out)}, profit {money(cash_out - stake)}.")
            else:
                st.error(f"Ticket lost: cash out {money(cash_out)}, loss {money(stake - cash_out)}.")

        st.caption("Each extra pick multiplies the odds but also the chance of losing. "
                   "Handicap pushes and half wins are paid as at a bookmaker.")
