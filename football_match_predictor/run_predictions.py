import subprocess
import sys
from pathlib import Path


# Runs the whole pipeline with one command:
#   (optional) lesson01: download the latest results
#   (optional) fetch_weather: download stadium weather from Open-Meteo
#   (optional) fetch_fpl: download players, xG and injuries from Fantasy Premier League
#   Model 1 (lesson10):  build features and predict match stats
#   Model 1b (lesson12): split each team's expected goals between its players
#   Model 2 (lesson11):  assessment, prediction and markets
#
# Model 2 reads Model 1's output file, so they run one after the other.
#
# Usage:
#   python run_predictions.py                       -> all upcoming games
#   python run_predictions.py "Man United"          -> one team, every market
#   python run_predictions.py --update "Man United" -> download new results first

BASE_DIR = Path(__file__).resolve().parent

arguments = sys.argv[1:]
update = "--update" in arguments
team_arguments = [argument for argument in arguments if argument != "--update"]

# (name, command, show output?)
steps = []
if update:
    # lesson01 prints every match, so hide its output unless it fails
    steps.append(("Downloading latest results", ["lesson01_load_data.py"], False))
    steps.append(("Downloading weather (Open-Meteo)", ["fetch_weather.py"], False))
    steps.append(("Downloading players and injuries (FPL)", ["fetch_fpl.py"], False))
    steps.append(("Head-to-head history 2000-2021 (once)", ["fetch_h2h_history.py"], False))
steps += [
    ("Model 1: stats engineer", ["lesson10_stats_engineer.py"], True),
    ("Model 1b: goalscorers", ["lesson12_goalscorers.py"], True),
    ("Model 2: match analyst", ["lesson11_match_analyst.py", *team_arguments], True),
]

for name, command, show_output in steps:
    print(f"\n>>> {name}", flush=True)
    finished = subprocess.run(
        [sys.executable, *command],
        cwd=BASE_DIR,
        capture_output=not show_output,
        text=True,
    )

    # Stop if a step fails, so Model 2 never reads old or broken stats
    if finished.returncode != 0:
        if not show_output:
            print(finished.stdout[-2000:], finished.stderr[-2000:])
        print(f"\n{name} failed - stopping.")
        sys.exit(finished.returncode)

    if not show_output:
        print("Done.", flush=True)
                                