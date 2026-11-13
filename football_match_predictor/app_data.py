import hashlib
import html
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st


# Shared by every page of the website: reads the files the pipeline saves,
# and formats teams, percentages and results the same way everywhere.

DATA_DIR = Path(__file__).resolve().parent / "data" / "processed"

MIN_EDGE = 0.05  # same value bet rule as lesson11_match_analyst.py
MAX_VALUE_ODDS = 5.0

TEAM_CODES = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Burnley": "BUR", "Chelsea": "CHE", "Coventry": "COV",
    "Crystal Palace": "CRY", "Everton": "EVE", "Fulham": "FUL", "Hull": "HUL",
    "Ipswich": "IPS", "Leeds": "LEE", "Leicester": "LEI", "Liverpool": "LIV",
    "Luton": "LUT", "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW",
    "Norwich": "NOR", "Nott'm Forest": "NFO", "Sheffield United": "SHU",
    "Southampton": "SOU", "Sunderland": "SUN", "Tottenham": "TOT", "Watford": "WAT",
    "West Ham": "WHU", "Wolves": "WOL",
}
CONFIDENCE_BADGES = {"High": "High", "Medium": "Medium", "Low": "Low"}  # coloured by colour_table

# Design tokens (same values as .streamlit/config.toml)
ACCENT = "#D7F94B"   # brand lime: positives, highlights, primary buttons
NEGATIVE = "#F2555A"
AMBER = "#F5B841"
MUTED = "#8B9098"
DRAW = "#4A4F57"

# Club colours (main, alternative) picked to read well on a dark background.
# The away side switches to its alternative colour when both clubs look alike.
TEAM_COLOURS = {
    "Arsenal": ("#EF0107", "#DB9F2C"), "Aston Villa": ("#A3C5E9", "#9B2D5A"),
    "Bournemouth": ("#DA291C", "#E8E8E8"), "Brentford": ("#E30613", "#FBB800"),
    "Brighton": ("#1A6BD6", "#FFCD00"), "Burnley": ("#A6264F", "#99D6EA"),
    "Chelsea": ("#2E62C9", "#DBA111"), "Coventry": ("#59CBE8", "#E8E8E8"),
    "Crystal Palace": ("#2F6BD6", "#C4122E"), "Everton": ("#2457D6", "#E8E8E8"),
    "Fulham": ("#E8E8E8", "#CC0000"), "Hull": ("#F5A12D", "#E8E8E8"),
    "Ipswich": ("#3A64A3", "#DE2C37"), "Leeds": ("#E8E8E8", "#FFCD00"),
    "Leicester": ("#2A5BD7", "#FDBE11"), "Liverpool": ("#C8102E", "#00B2A9"),
    "Luton": ("#F78F1E", "#E8E8E8"), "Man City": ("#6CABDD", "#E8E8E8"),
    "Man United": ("#DA291C", "#FBE122"), "Newcastle": ("#E8E8E8", "#41B6E6"),
    "Norwich": ("#FFF200", "#00A650"), "Nott'm Forest": ("#DD0000", "#E8E8E8"),
    "Sheffield United": ("#EE2737", "#E8E8E8"), "Southampton": ("#D71920", "#FFC20E"),
    "Sunderland": ("#EB172B", "#E8E8E8"), "Tottenham": ("#E8E8E8", "#3A5BC7"),
    "Watford": ("#FBEE23", "#ED2127"), "West Ham": ("#9B2F4A", "#1BB1E7"),
    "Wolves": ("#FDB913", "#E8E8E8"),
}


@st.cache_data(ttl=300)  # re-read every 5 minutes, so a new pipeline run shows up
def read(path):
    data = pd.read_csv(path)
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"])
    return data


def load(name):
    path = DATA_DIR / name
    if not path.exists():
        st.error(f"Missing {path.name} - run `python run_predictions.py` first.")
        st.stop()
    return read(path)


def short(text):
    # "Man United vs Aston Villa" -> "MUN vs AVL" (longest names first, so "Man United" wins over shorter ones)
    if not isinstance(text, str):
        return text
    for name in sorted(TEAM_CODES, key=len, reverse=True):
        text = text.replace(name, TEAM_CODES[name])
    return text


def tick(value):
    # True / False / missing -> ✓ / ✗ / blank (coloured by colour_table)
    return "" if pd.isna(value) else "✓" if value else "✗"


def team_colours(home, away):
    def distance(first, second):
        return sum((int(first[i:i + 2], 16) - int(second[i:i + 2], 16)) ** 2 for i in (1, 3, 5)) ** 0.5

    home_colour = TEAM_COLOURS.get(home, (ACCENT, ACCENT))[0]
    away_main, away_alternative = TEAM_COLOURS.get(away, ("#9AA4B2", "#9AA4B2"))
    away_colour = away_main if distance(home_colour, away_main) > 110 else away_alternative
    if distance(home_colour, away_colour) <= 110:
        away_colour = "#9AA4B2"
    return home_colour, away_colour


def match_link(match):
    # Each match has its own address, so rows are plain links and pages can be shared
    return f"Match_details?match={quote(match)}"


# Column formats: values are stored as 0-1, the tables multiply them by 100
def percent_column(label, bar=False, help=None):
    if bar:
        return st.column_config.ProgressColumn(label, format="%.0f%%", min_value=0, max_value=100, help=help)
    return st.column_config.NumberColumn(label, format="%.0f%%", help=help)


def number_column(label, decimals=1, help=None):
    return st.column_config.NumberColumn(label, format=f"%.{decimals}f", help=help)


def text_column(label, help=None):
    return st.column_config.TextColumn(label, help=help, alignment="center")


# ---------------------------------------------------------------------------
# Site look: one neutral dark surface, one lime accent, club colours for team data,
# condensed type for headlines and numbers. Call apply_style() at the top of every page.
# ---------------------------------------------------------------------------

APP_CSS = """
<style>
:root {--surface:#15171B; --surface-2:#1C1F24; --line:#24272D; --muted:#8B9098;
       --accent:#D7F94B; --neg:#F2555A; --amber:#F5B841; --display:'Barlow Condensed', sans-serif}

/* Headlines: condensed caps; section headings are quiet labels */
h1 {font-family:var(--display) !important; text-transform:uppercase; letter-spacing:.3px; line-height:1 !important}
h3 {font-family:var(--display) !important; text-transform:uppercase; letter-spacing:1px;
    font-size:1.05rem !important; color:var(--muted) !important; font-weight:600 !important; margin-top:.6rem}
[data-testid="stCaptionContainer"] {color:var(--muted)}

/* Number cards */
[data-testid="stMetric"] {background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:12px 16px}
[data-testid="stMetricLabel"] {text-transform:uppercase; letter-spacing:.7px; font-size:.72rem; color:var(--muted)}
[data-testid="stMetricValue"] {font-family:var(--display); font-variant-numeric:tabular-nums}

/* Tables, tabs, buttons, links */
[data-testid="stDataFrame"] {border:1px solid var(--line); border-radius:10px; overflow:hidden}
[data-baseweb="tab"] {text-transform:uppercase; letter-spacing:.7px; font-weight:600; font-size:.82rem}
[data-testid="stBaseButton-primary"] {color:#0C0D0F !important; font-weight:700; text-transform:uppercase;
    letter-spacing:.6px}
[data-testid="stPageLink"] a {font-weight:600}
[data-testid="stPageLink"] a span {color:var(--accent) !important}

/* Page header */
.ph {padding:4px 0 22px; border-bottom:1px solid var(--line); margin-bottom:8px}
.ph-eyebrow {font-size:.74rem; font-weight:600; letter-spacing:1.8px; text-transform:uppercase; color:var(--accent)}
.ph-title {font-family:var(--display); font-weight:700; font-size:3.4rem; line-height:.95;
    text-transform:uppercase; margin:8px 0 10px}
.ph-meta {color:var(--muted); font-size:.95rem}
.ph-kpis {display:flex; flex-wrap:wrap; margin-top:20px; row-gap:12px}
.ph-kpi {padding-right:28px; margin-right:28px; border-right:1px solid var(--line)}
.ph-kpi:last-child {border-right:none}
.ph-num {font-family:var(--display); font-size:2.3rem; font-weight:700; line-height:1; font-variant-numeric:tabular-nums}
.ph-lbl {font-size:.7rem; text-transform:uppercase; letter-spacing:.9px; color:var(--muted); margin-top:4px}

/* Fixture list (home page) */
.fx-day {font-size:.74rem; text-transform:uppercase; letter-spacing:1.4px; color:var(--muted);
    font-weight:600; margin:22px 0 8px}
.fx-list {border:1px solid var(--line); border-radius:10px; overflow:hidden; background:var(--surface)}
.fx {display:grid; grid-template-columns:minmax(0,1fr) 70px minmax(0,1fr) 230px 170px 14px; align-items:center;
    gap:16px; padding:13px 18px; border-top:1px solid var(--line); color:inherit !important;
    text-decoration:none !important; transition:background .15s}
.fx:first-child {border-top:none}
.fx:hover {background:var(--surface-2)}
.fx:focus-visible {outline:2px solid var(--accent); outline-offset:-2px}
.fx-team {display:flex; align-items:center; gap:10px; font-weight:600; min-width:0}
.fx-team.home {justify-content:flex-end; text-align:right}
.fx-team img, .fx-team .mh-logo {width:28px; height:28px; object-fit:contain; flex:none}
.fx-team .mh-logo {font-size:.6rem}
.fx-name {white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.fx-score {font-family:var(--display); font-size:1.55rem; font-weight:700; text-align:center;
    font-variant-numeric:tabular-nums; letter-spacing:1px}
.fx-score.pre {font-family:inherit; font-size:.8rem; color:var(--muted); font-weight:600; letter-spacing:.5px}
.fx-prob {display:flex; flex-direction:column; gap:6px}
.fx-bar {display:flex; height:6px; border-radius:3px; overflow:hidden; gap:2px}
.fx-bar i {display:block}
.fx-nums {display:flex; justify-content:space-between; font-size:.78rem; color:var(--muted); font-variant-numeric:tabular-nums}
.fx-nums b {color:#E7E9EC}
.fx-pick {display:flex; flex-direction:column; gap:2px; min-width:0}
.fx-lbl {font-size:.66rem; text-transform:uppercase; letter-spacing:.9px; color:var(--muted)}
.fx-val {font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis}
.fx-go {color:var(--muted)}
.conf {font-size:.66rem; font-weight:700; text-transform:uppercase; letter-spacing:.7px}
.conf-High {color:var(--accent)} .conf-Medium {color:var(--amber)} .conf-Low {color:var(--muted)}
.ok {color:var(--accent); font-weight:700} .no {color:var(--neg); font-weight:700}

/* Lenis: iframes must not swallow the wheel while it smooth-scrolls */
.lenis.lenis-smooth iframe {pointer-events:none}

@media (max-width:900px) {
  .ph-title {font-size:2.4rem}
  .ph-kpi {padding-right:18px; margin-right:18px}
  .fx {grid-template-columns:minmax(0,1fr) 56px minmax(0,1fr); row-gap:10px; padding:12px 14px}
  .fx-prob, .fx-pick {grid-column:1 / -1}
  .fx-pick {flex-direction:row; align-items:center; gap:10px}
  .fx-go {display:none}
}
</style>
"""

# Smooth scrolling with Lenis (https://github.com/darkroomengineering/lenis, MIT). Streamlit scrolls
# inside <main data-testid="stMain">, not the window, so Lenis is pointed at that element.
# One instance per page; nested scroll areas (tables, dropdowns) keep scrolling natively.
LENIS_URL = "https://cdn.jsdelivr.net/npm/lenis@1.3.27/dist/lenis.min.js"

SMOOTH_SCROLL_SCRIPT = """<script>
(() => {
  window.__lenisReady = window.__lenisReady || new Promise(resolve => {
    if (window.Lenis) return resolve();
    const tag = document.createElement('script');
    tag.src = '__URL__';
    tag.onload = resolve;
    document.head.appendChild(tag);
  });
  window.__lenisReady.then(() => {
    const wrapper = document.querySelector('[data-testid="stMain"]');
    if (!wrapper || window.__lenis?.options.wrapper === wrapper) return;
    window.__lenis?.destroy();
    window.__lenis = new Lenis({ wrapper, content: wrapper.firstElementChild, autoRaf: true, allowNestedScroll: true });
  });
})();
</script>""".replace("__URL__", LENIS_URL)


def apply_style():
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.html(SMOOTH_SCROLL_SCRIPT, unsafe_allow_javascript=True)


# ---------------------------------------------------------------------------
# Match page pieces: header with badges, win chance bar, comparison bars
# ---------------------------------------------------------------------------

BADGE_URL = "https://resources.premierleague.com/premierleague/badges/70/t{code}.png"

MATCH_CSS = """
<style>
.mh {display:grid; grid-template-columns:minmax(0,1fr) auto minmax(0,1fr); align-items:center; gap:16px;
     padding:26px 16px 22px; border:1px solid var(--line); border-radius:12px; background:var(--surface);
     margin:4px 0 16px; position:relative; overflow:hidden}
.mh::before, .mh::after {content:""; position:absolute; top:0; height:3px; width:50%}
.mh::before {left:0; background:var(--home)} .mh::after {right:0; background:var(--away)}
.mh-team {display:flex; flex-direction:column; align-items:center; gap:10px; text-align:center; min-width:0}
.mh-team img, .mh-logo {width:68px; height:68px; object-fit:contain}
.mh-logo {border-radius:50%; display:flex; align-items:center; justify-content:center;
          font-weight:700; background:var(--surface-2); border:1px solid var(--line)}
.mh-name {font-family:var(--display); font-weight:700; font-size:1.45rem; text-transform:uppercase; letter-spacing:.4px}
.mh-center {text-align:center}
.mh-score {font-family:var(--display); font-size:3.6rem; font-weight:700; line-height:1; letter-spacing:2px;
           font-variant-numeric:tabular-nums; margin:6px 0}
.mh-label {font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:1.1px}
.cb {margin:4px 0 20px}
.cb-labels {display:flex; justify-content:space-between; margin-bottom:8px}
.cb-labels > div {display:flex; flex-direction:column; gap:2px}
.cb-labels .cb-d {align-items:center} .cb-labels .cb-a {align-items:flex-end}
.cb-name {font-size:.7rem; color:var(--muted); text-transform:uppercase; letter-spacing:.9px}
.cb-pct {font-family:var(--display); font-size:2rem; font-weight:700; line-height:1; font-variant-numeric:tabular-nums}
.cb-bar {display:flex; height:8px; border-radius:4px; overflow:hidden; gap:2px}
.sb {--home:#D7F94B; --away:#9AA4B2}
.sb-row {margin:14px 0}
.sb-top {display:flex; justify-content:space-between; font-variant-numeric:tabular-nums; font-size:.92rem}
.sb-label {color:var(--muted); font-size:.74rem; text-transform:uppercase; letter-spacing:.9px}
.sb-lead {font-weight:700}
.sb-bars {display:flex; gap:4px; margin-top:6px}
.sb-half {flex:1; height:5px; border-radius:3px; background:var(--surface-2); display:flex; overflow:hidden}
.sb-half.home {justify-content:flex-end}
.sb-half.home div {background:var(--home)}
.sb-half.away div {background:var(--away)}
/* Centre the number cards, captions and headings in the overview and head-to-head tabs */
:is(.st-key-overview, .st-key-h2h) [data-testid="stMetric"] {text-align:center}
:is(.st-key-overview, .st-key-h2h) :is([data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"]) {display:flex; justify-content:center; width:100%}
:is(.st-key-overview, .st-key-h2h) :is([data-testid="stCaptionContainer"], h3) {text-align:center}
@media (max-width:640px) {
  .mh-team img, .mh-logo {width:46px; height:46px}
  .mh-score {font-size:2.4rem}
  .mh-name {font-size:1.05rem}
}
</style>
"""


@st.cache_data(ttl=3600)
def badge_codes():
    path = DATA_DIR.parent / "external" / "fpl_teams.csv"
    return pd.read_csv(path).set_index("team")["badge_code"].to_dict() if path.exists() else {}


def team_logo(team):
    # Club badge from the Premier League site; teams without a code get their 3-letter code instead
    code = badge_codes().get(team)
    if code is None:
        return f'<div class="mh-logo">{TEAM_CODES.get(team, "?")}</div>'
    return f'<img src="{BADGE_URL.format(code=code)}" alt="{html.escape(team)} badge">'


def match_header(home, away, score, label, sub):
    home_colour, away_colour = team_colours(home, away)
    teams = [
        f'<div class="mh-team">{team_logo(team)}<div class="mh-name">{html.escape(team)}</div></div>'
        for team in (home, away)
    ]
    return (
        f'<div class="mh" style="--home:{home_colour}; --away:{away_colour}">{teams[0]}'
        f'<div class="mh-center"><div class="mh-label">{html.escape(label)}</div>'
        f'<div class="mh-score">{html.escape(score)}</div>'
        f'<div class="mh-label">{html.escape(sub)}</div></div>'
        f'{teams[1]}</div>'
    )


# Win / draw / win bar that grows in with GSAP (https://gsap.com). The HTML already has the
# final widths, so if GSAP can't load the bar still shows, just without the animation.
GSAP_URL = "https://cdn.jsdelivr.net/npm/gsap@3.13.0/dist/gsap.min.js"


def chance_bar(home, away, home_chance, draw_chance, away_chance):
    home_colour, away_colour = team_colours(home, away)
    parts = [
        ("h", f"{short(home)} win", home_chance, home_colour),
        ("d", "Draw", draw_chance, DRAW),
        ("a", f"{short(away)} win", away_chance, away_colour),
    ]
    labels = ""
    for kind, name, chance, colour in parts:
        style = "" if kind == "d" else f' style="color:{colour}"'  # the draw keeps the normal text colour
        labels += (
            f'<div class="cb-{kind}"><span class="cb-name">{html.escape(name)}</span>'
            f'<span class="cb-pct" data-p="{chance * 100:.0f}"{style}>{chance:.0%}</span></div>'
        )
    bars = "".join(
        f'<div class="{kind}" style="flex:{chance:.4f}; background:{colour}"></div>'
        for kind, _, chance, colour in parts
    )
    st.html(
        f'<div class="cb">'
        f'<div class="cb-labels">{labels}</div><div class="cb-bar">{bars}</div></div>'
        f"""<script>
(() => {{
  const root = [...document.querySelectorAll('.cb')].pop();
  if (!root || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const run = () => {{
    const tl = gsap.timeline({{ defaults: {{ ease: 'power3.out' }} }});
    tl.from(root.querySelectorAll('.cb-bar div'), {{ flexGrow: 0, duration: 0.9, stagger: 0.08 }});
    root.querySelectorAll('.cb-pct').forEach(el => {{
      const n = {{ v: 0 }};
      el.textContent = '0%';
      tl.to(n, {{ v: +el.dataset.p, duration: 0.9, onUpdate: () => el.textContent = Math.round(n.v) + '%' }}, 0);
    }});
  }};
  if (window.gsap) return run();
  const s = document.createElement('script');
  s.src = '{GSAP_URL}';
  s.onload = run;
  document.head.appendChild(s);
}})();
</script>""",
        unsafe_allow_javascript=True,
    )


def stat_bars(rows, home_colour=ACCENT, away_colour="#9AA4B2"):
    # rows: (label, home value, away value, decimals). Each bar = that side's share of the total.
    parts = []
    for label, home, away, decimals in rows:
        total = home + away
        home_share = home / total * 100 if total else 0
        away_share = away / total * 100 if total else 0
        home_class = "sb-lead" if home > away else ""
        away_class = "sb-lead" if away > home else ""
        parts.append(
            f'<div class="sb-row"><div class="sb-top">'
            f'<span class="{home_class}" data-count="{home}" data-decimals="{decimals}">{home:.{decimals}f}</span>'
            f'<span class="sb-label">{html.escape(label)}</span>'
            f'<span class="{away_class}" data-count="{away}" data-decimals="{decimals}">{away:.{decimals}f}</span></div>'
            f'<div class="sb-bars"><div class="sb-half home"><div style="width:{home_share:.0f}%"></div></div>'
            f'<div class="sb-half away"><div style="width:{away_share:.0f}%"></div></div></div></div>'
        )
    return f'<div class="sb" style="--home:{home_colour}; --away:{away_colour}">{"".join(parts)}</div>'


# ---------------------------------------------------------------------------
# Player rating 0-10 from FPL's Bonus Points System (BPS)
# ---------------------------------------------------------------------------

def player_rating(bps, minutes):
    # shortcut: our own linear scale on BPS (median starter 13 BPS -> 6.6, top 10% 33+ -> 8.6+),
    # not SofaScore's ratings; replace with a real rating source if one becomes available
    rating = (6.6 + 1.2 * (bps - 13) / 12).clip(3.0, 10.0).round(1)
    return rating.where(minutes >= 10)


def rating_colour(value):
    # Usual rating scale of football apps: blue 9+, green 8+, light green 7+, yellow 6.5+, orange 6+, red below
    if pd.isna(value):
        return ""
    colour = ("#3B82F6" if value >= 9 else "#16A34A" if value >= 8 else "#4CAF50" if value >= 7
              else "#C9A227" if value >= 6.5 else "#E07B24" if value >= 6 else "#D64545")
    return f"background-color:{colour}; color:white; font-weight:700"


# ---------------------------------------------------------------------------
# Home page: page header with key numbers, fixture list grouped by day
# ---------------------------------------------------------------------------

def page_header(eyebrow, title, meta, kpis=()):
    # kpis: (label, number, suffix, tooltip); numbers count up from 0
    numbers = "".join(
        f'<div class="ph-kpi" title="{html.escape(tooltip)}">'
        f'<div class="ph-num" data-count="{number}" data-suffix="{html.escape(suffix)}">{number}{html.escape(suffix)}</div>'
        f'<div class="ph-lbl">{html.escape(label)}</div></div>'
        for label, number, suffix, tooltip in kpis
    )
    animate(
        f'<div class="ph"><div class="ph-eyebrow">{html.escape(eyebrow)}</div>'
        f'<div class="ph-title">{html.escape(title)}</div>'
        f'<div class="ph-meta">{html.escape(meta)}</div>'
        + (f'<div class="ph-kpis">{numbers}</div>' if numbers else "") + "</div>",
        HEADER_IN_MOTION,
    )


def fixture_row(match):
    # One clickable row: teams and score, win / draw / win bar in club colours, the model's pick
    home, away = match["home_team"], match["away_team"]
    home_colour, away_colour = team_colours(home, away)
    played = pd.notna(match.get("actual"))
    score = (
        f'<div class="fx-score">{html.escape(match["actual"]).replace("-", "–")}</div>' if played
        else '<div class="fx-score pre">VS</div>'
    )
    chances = [match["home_chance"], match["draw_chance"], match["away_chance"]]
    leader = chances.index(max(chances))
    bar = "".join(
        f'<i style="flex:{chance:.4f}; background:{colour}"></i>'
        for chance, colour in zip(chances, [home_colour, DRAW, away_colour])
    )
    numbers = "".join(
        f'<span>{"<b>" if position == leader else ""}{chance:.0%}{"</b>" if position == leader else ""}</span>'
        for position, chance in enumerate(chances)
    )
    result = ""
    if played:
        result = ' <span class="ok">✓</span>' if match["pick_right"] else ' <span class="no">✗</span>'
    return (
        f'<a class="fx" href="{match_link(match["match"])}" target="_self">'
        f'<div class="fx-team home"><span class="fx-name">{html.escape(home)}</span>{team_logo(home)}</div>'
        f'{score}'
        f'<div class="fx-team">{team_logo(away)}<span class="fx-name">{html.escape(away)}</span></div>'
        f'<div class="fx-prob"><div class="fx-bar">{bar}</div><div class="fx-nums">{numbers}</div></div>'
        f'<div class="fx-pick"><span class="fx-lbl">Model pick</span>'
        f'<span class="fx-val">{html.escape(short(match["pick"]))} · {match["pick_chance"]:.0%}{result}</span>'
        f'<span class="conf conf-{match["confidence"]}">{match["confidence"]} confidence</span></div>'
        f'<span class="fx-go">›</span></a>'
    )


def fixture_list(matches):
    # Matches grouped by day, one bordered list per day
    blocks = []
    for day, day_matches in matches.groupby(matches["date"].dt.date, sort=True):
        rows = "".join(fixture_row(match) for _, match in day_matches.iterrows())
        blocks.append(f'<div class="fx-day">{day:%A %d %B}</div><div class="fx-list">{rows}</div>')
    return "".join(blocks)


# ---------------------------------------------------------------------------
# Coloured tables: colour only where it carries meaning.
#   ✓ / W / Yes / High in lime, ✗ / L in red, Medium in amber; Edge / ROI / Profit columns
#   lime when positive and red when negative; "heat" columns get a light lime tint that grows with the value.
# ---------------------------------------------------------------------------

TEXT_COLOURS = {
    "✓": ACCENT, "✗": NEGATIVE, "W": ACCENT, "D": MUTED, "L": NEGATIVE,
    "Yes": ACCENT, "No": MUTED, "High": ACCENT, "Medium": AMBER, "Low": MUTED,
}
SIGNED_WORDS = ("Edge", "ROI", "Profit", "profit", "Improvement", "Home team", "Away team", "Match total")


def to_number(values):
    # 0.5, "51%", "+9.5%" -> numbers, so text tables from the CSVs can be coloured too
    return pd.to_numeric(values.astype(str).str.replace(r"[%+,]", "", regex=True), errors="coerce")


def cell_colour(value):
    colour = TEXT_COLOURS.get(value) if isinstance(value, str) else None
    return f"color:{colour}; font-weight:700" if colour else ""


def signed_colour(values):
    numbers = to_number(values)
    return [f"color:{ACCENT}; font-weight:700" if number > 0 else f"color:{NEGATIVE}; font-weight:700"
            if number < 0 else "" for number in numbers]


def heat_colour(values):
    numbers = to_number(values)
    low, high = numbers.min(), numbers.max()
    share = (numbers - low) / (high - low) if high > low else numbers * 0 + 0.5
    return [f"background-color:rgba(215,249,75,{0.03 + 0.22 * value:.2f})" if pd.notna(value) else ""
            for value in share]


def colour_table(table, formats=None, heat=()):
    # formats: {column: "{:.0f}%"} - a coloured table shows these instead of the column_config format
    formats = {column: form for column, form in (formats or {}).items() if column in table.columns}
    styler = table.style.format(formats, na_rep="").map(cell_colour)
    signed = [column for column in table.columns if any(word in column for word in SIGNED_WORDS)]
    if signed:
        styler = styler.apply(signed_colour, subset=signed)
    heat = [column for column in heat if column in table.columns]
    if heat:
        styler = styler.apply(heat_colour, subset=heat)
    return styler


# ---------------------------------------------------------------------------
# GSAP animations (https://gsap.com), same way as chance_bar: st.html runs the script on the page.
# The HTML already has its final look and GSAP only animates *from* a start state,
# so if GSAP can't load (or the viewer prefers reduced motion) everything still shows.
# Numbers with data-count="62" (+ data-suffix / data-decimals) count up from 0.
# Motion is short and small on purpose: it guides the eye, it is not decoration.
# ---------------------------------------------------------------------------

ANIMATION_SCRIPT = """<script>
(() => {
  const root = document.getElementById('__ID__');
  if (!root || matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  window.__gsapReady = window.__gsapReady || new Promise(resolve => {
    if (window.gsap) return resolve();
    const tag = document.createElement('script');
    tag.src = '__URL__';
    tag.onload = resolve;
    document.head.appendChild(tag);
  });
  window.__gsapReady.then(() => {
    const delay = __DELAY__;
    root.querySelectorAll('[data-count]').forEach(el => {
      const end = +el.dataset.count, decimals = +(el.dataset.decimals || 0), suffix = el.dataset.suffix || '';
      const n = { v: 0 };
      el.textContent = (0).toFixed(decimals) + suffix;
      gsap.to(n, { v: end, duration: 0.9, delay, ease: 'power2.out',
                   onUpdate: () => el.textContent = n.v.toFixed(decimals) + suffix });
    });
    __MOTION__
  });
})();
</script>"""

HEADER_IN_MOTION = """
gsap.from(root.querySelectorAll('.ph > *'), { y: 10, opacity: 0, duration: 0.5, stagger: 0.06, ease: 'power2.out' });
"""

LIST_MOTION = """
gsap.from(root.querySelectorAll('.fx-day, .fx'), { y: 8, opacity: 0, duration: 0.35, stagger: 0.035, delay, ease: 'power2.out' });
gsap.from(root.querySelectorAll('.fx-bar i'),
          { scaleX: 0, transformOrigin: 'left center', duration: 0.6, stagger: 0.01, delay: delay + 0.15, ease: 'power2.out' });
"""

HEADER_MOTION = """
const tl = gsap.timeline({ delay, defaults: { ease: 'power2.out' } });
tl.from(root.querySelectorAll('.mh-team'), { y: 10, opacity: 0, duration: 0.45, stagger: 0.06 })
  .from(root.querySelector('.mh-score'), { opacity: 0, scale: 0.92, duration: 0.45 }, '-=0.3')
  .from(root.querySelectorAll('.mh-label'), { opacity: 0, duration: 0.4 }, '-=0.3');
"""

BARS_MOTION = """
gsap.from(root.querySelectorAll('.sb-row'), { opacity: 0, duration: 0.35, stagger: 0.04, delay });
gsap.from(root.querySelectorAll('.sb-half.home div'),
          { scaleX: 0, transformOrigin: 'right center', duration: 0.7, stagger: 0.04, delay, ease: 'power2.out' });
gsap.from(root.querySelectorAll('.sb-half.away div'),
          { scaleX: 0, transformOrigin: 'left center', duration: 0.7, stagger: 0.04, delay, ease: 'power2.out' });
"""


def animation_html(markup, motion, delay=0.0):
    # The id comes from the content, so a rerun with the same content does not replay the animation
    element_id = "anim-" + hashlib.md5(f"{markup}{motion}{delay}".encode()).hexdigest()[:10]
    script = (ANIMATION_SCRIPT.replace("__ID__", element_id).replace("__URL__", GSAP_URL)
              .replace("__DELAY__", f"{delay:.2f}").replace("__MOTION__", motion))
    return f'<div id="{element_id}">{markup}</div>{script}'


def animate(markup, motion, delay=0.0):
    st.html(animation_html(markup, motion, delay), unsafe_allow_javascript=True)
