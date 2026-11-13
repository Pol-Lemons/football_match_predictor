from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest


# Smoke test: every page of the website loads without an error.
# Run the pipeline first, then:  python test_app.py

BASE_DIR = Path(__file__).resolve().parent
PAGES = sorted(f"pages/{page.name}" for page in (BASE_DIR / "pages").glob("*.py"))


def open_page(page=None, **state):
    # Start from app.py like a real visitor, so links between pages work
    app = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=60)
    for key, value in state.items():
        app.session_state[key] = value
    app.run()
    if page:
        app.switch_page(page).run()
    assert not app.exception, f"{page}: {app.exception}"
    assert not app.error, f"{page}: {[error.value for error in app.error]}"
    return app


open_page()
print("ok  app.py")
for page in PAGES:
    open_page(page)
    print(f"ok  {page}")

# The match page must work for every match, not just the first one
app = open_page("pages/1_Match_details.py")
for match in app.selectbox[0].options:
    app.selectbox[0].select(match).run()
    assert not app.exception, f"{match}: {app.exception}"
print(f"ok  all {len(app.selectbox[0].options)} matches on the match page")

# Bet calculator: a 2-pick parlay pays stake x odds1 x odds2
markets = pd.read_csv(BASE_DIR / "data" / "processed" / "match_predictions.csv")
legs = markets[markets["market"] == "1X2"].drop_duplicates("match").head(2)
slip = (legs["match"] + " | " + legs["pick"]).tolist()

app = open_page("pages/4_Bet_calculator.py", bet_slip=slip)
metrics = {metric.label: metric.value for metric in app.metric}
expected_odds = legs["bookmaker_odds"].round(2).prod()
assert metrics["Combined odds"] == f"{expected_odds:.2f}", metrics
assert metrics["Cash out if all win"] == f"{100_000 * expected_odds:,.0f}", metrics
print(f"ok  bet calculator: parlay odds {metrics['Combined odds']}, cash out {metrics['Cash out if all win']}")

# Two picks from the same match cannot go in one parlay
same_match = markets[markets["match"] == legs["match"].iloc[0]].head(2)
app = open_page("pages/4_Bet_calculator.py", bet_slip=(same_match["match"] + " | " + same_match["pick"]).tolist())
assert any("one pick per match" in warning.value for warning in app.warning)
print("ok  bet calculator blocks two picks from one match in a parlay")

# Team codes: longest names first, so "Man United" is not cut by a shorter name
from app_data import short

assert short("Man United vs Aston Villa") == "MUN vs AVL"
assert short("Nott'm Forest or Man City") == "NFO or MCI"
assert short("Over 2.5 goals") == "Over 2.5 goals"
print("ok  team codes")

# Match page look: bars split by share, no division by zero, team names escaped
from app_data import match_header, stat_bars

bars = stat_bars([("Corners", 6, 2, 0), ("Red cards", 0, 0, 0)])
assert "width:75%" in bars and "width:25%" in bars and "width:0%" in bars, bars
header = match_header("Nott'm Forest", "Unknown FC", "1 - 1", "Predicted score", "Sat")
assert "Nott&#x27;m Forest" in header and 'class="mh-logo">?<' in header, header
print("ok  match header and stat bars")

# Player rating: average starter ~6.6, capped 3-10, no rating under 10 minutes
from app_data import player_rating

ratings = player_rating(pd.Series([13, 33, 120, -40, 20]), pd.Series([90, 90, 90, 90, 5]))
assert ratings.iloc[0] == 6.6 and ratings.iloc[1] == 8.6, ratings
assert ratings.iloc[2] == 10.0 and ratings.iloc[3] == 3.0 and pd.isna(ratings.iloc[4]), ratings
print("ok  player rating")

# Head-to-head: clicking any meeting must work (old seasons have no player data)
app = open_page("pages/1_Match_details.py")
for match in app.selectbox[0].options:
    app.selectbox[0].select(match).run()
    assert not app.exception, f"{match}: {app.exception}"
print("ok  head-to-head and player tables for every match")

# Home page fixture row: escaped names, link to the match page, real score after the match
from app_data import fixture_row, match_link

card_match = pd.read_csv(BASE_DIR / "data" / "processed" / "app" / "summary.csv", parse_dates=["date"]).iloc[0]
row_html = fixture_row(card_match)
assert card_match["home_team"] in row_html and f"conf-{card_match['confidence']}" in row_html, row_html
assert f'href="{match_link(card_match["match"])}"' in row_html
assert (card_match["actual"].replace("-", "–") in row_html) if pd.notna(card_match.get("actual")) else ("VS" in row_html)
assert match_link("Nott'm Forest vs Man City") == "Match_details?match=Nott%27m%20Forest%20vs%20Man%20City"
print("ok  fixture row")

# A fixture link (?match=...) opens the match page on that match
third_match = pd.read_csv(BASE_DIR / "data" / "processed" / "app" / "summary.csv")["match"].iloc[2]
app = AppTest.from_file(str(BASE_DIR / "app.py"), default_timeout=60).run()
app.switch_page("pages/1_Match_details.py")
app.query_params["match"] = third_match
app.run()
assert not app.exception, app.exception
assert app.selectbox[0].value == third_match, "match page did not open on the linked match"
print("ok  match link")

# Coloured tables: meaning-only colours (✓ lime, ✗ red, signed numbers), one-colour highlight
from app_data import ACCENT, NEGATIVE, colour_table

styled = colour_table(
    pd.DataFrame({"Right": ["✓", "✗"], "Edge": [6.0, -3.0], "Chance": [80.0, 20.0], "Conf": ["High", "Low"]}),
    {"Edge": "{:+.0f}%", "Chance": "{:.0f}%"}, heat=["Chance"],
).to_html()
assert f"color: {ACCENT}" in styled and f"color: {NEGATIVE}" in styled  # ticks and signed edge
assert "rgba(215,249,75,0.25)" in styled and "rgba(215,249,75,0.03)" in styled  # highlight high / low
assert "+6%" in styled and "80%" in styled  # formats kept
print("ok  coloured tables")

# GSAP animation wrapper: same content -> same id (no replay on rerun), GSAP loaded, counters marked
from app_data import GSAP_URL, LIST_MOTION, animation_html

first = animation_html('<div class="mc">x</div>', LIST_MOTION, delay=0.14)
assert first == animation_html('<div class="mc">x</div>', LIST_MOTION, delay=0.14)
assert first != animation_html('<div class="mc">y</div>', LIST_MOTION, delay=0.14)
assert GSAP_URL in first and "const delay = 0.14;" in first and "__MOTION__" not in first
assert 'data-count="75.0"' in stat_bars([("Corners", 75.0, 25.0, 0)])
print("ok  gsap animations")

# Site effects: one Lenis smooth-scroll script on every page, no AI-style gimmicks left
import app_data
from app_data import LENIS_URL, SMOOTH_SCROLL_SCRIPT

assert LENIS_URL in SMOOTH_SCROLL_SCRIPT and "__URL__" not in SMOOTH_SCROLL_SCRIPT
assert not hasattr(app_data, "EFFECTS_SCRIPT"), "only one Lenis instance should be set up"
print("ok  site effects")
