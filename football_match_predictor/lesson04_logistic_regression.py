from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    log_loss
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent

data_path = (
    BASE_DIR / "data" / "processed"/ "premier_league_form_features.csv"
)

matches = pd.read_csv(
    data_path,
    parse_dates=["date"]

)

matches = matches.sort_values(
    "date"
).reset_index(drop=True)

form_features = [
    "home_form_points_5",
    "home_form_goals_for_5",
    "home_form_goals_against_5",
    "away_form_points_5",
    "away_form_goals_for_5",
    "away_form_goals_against_5",
]

# Keo nha cai: 1X2, chau A (handicap), tai xiu 2.5
odds_features = [
    "odds_home_probability",
    "odds_draw_probability",
    "odds_away_probability",
    "asian_handicap_line",
    "odds_over_2_5_probability",
]

features = form_features + odds_features

target = "result"

test_season = "2025-2026"

# Only train on seasons BEFORE the test season (no peeking at the future)
train_data = matches.loc[
    matches["season"] < test_season
].copy()

test_data = matches.loc[
    matches["season"] == test_season
].copy()

if test_data.empty:
    raise ValueError(
        f"No matches found for season {test_season}"
        f"Available seasons: "
        f"{sorted(matches['season'].unique())}"
    )

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

print("Training seasons: ")
print(sorted(train_data["season"].unique()))

print("\nTesting season: ")
print(test_season)

print("\nTraining matches: ", len(train_data))
print("Testing matches: ", len(test_data))

#BASELINE DE LAM MOC SO SANH

baseline_model = DummyClassifier(
    strategy="most_frequent"
)
baseline_model.fit(X_train,y_train)

baseline_pred = baseline_model.predict(X_test)


baseline_accuracy = accuracy_score(
    y_test,
    baseline_pred
)

baseline_balanced_accuracy = (
    balanced_accuracy_score(
        y_test,
        baseline_pred
    )
)

#LOGISTIC REGRESSION

def train_logistic_model(feature_columns):
    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=42
            )
        )
    ])
    model.fit(X_train[feature_columns], y_train)
    return model


def score_model(name, predictions, probabilities=None):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Balanced Accuracy": balanced_accuracy_score(y_test, predictions),
        "Log Loss": (
            log_loss(y_test, probabilities, labels=["A", "D", "H"])
            if probabilities is not None else None
        ),
    }


# Mo hinh 1: chi dung phong do
form_model = train_logistic_model(form_features)
form_pred = form_model.predict(X_test[form_features])
form_probability = form_model.predict_proba(X_test[form_features])

# Mo hinh 2: phong do + keo nha cai
logistic_model = train_logistic_model(features)
y_pred = logistic_model.predict(X_test[features])
y_probability = logistic_model.predict_proba(X_test[features])

# Nha cai: chon ket qua co xac suat cao nhat theo keo 1X2
odds_probability = test_data[[
    "odds_away_probability",
    "odds_draw_probability",
    "odds_home_probability",
]].to_numpy()  # A, D, H order, same as the model classes

bookmaker_pred = (
    test_data[[
        "odds_home_probability",
        "odds_draw_probability",
        "odds_away_probability",
    ]]
    .idxmax(axis=1)
    .map({
        "odds_home_probability": "H",
        "odds_draw_probability": "D",
        "odds_away_probability": "A",
    })
)

comparison = pd.DataFrame([
    score_model("Baseline (always home win)", baseline_pred),
    score_model("Logistic: form only", form_pred, form_probability),
    score_model("Logistic: form + odds", y_pred, y_probability),
    score_model("Bookmaker favourite", bookmaker_pred, odds_probability),
])

print("\nModel comparison: ")
print(
    comparison.to_string(
        index=False,
        float_format="{:.3f}".format,
        na_rep="-"
    )
)
print("(Higher accuracy is better. Lower log loss is better.)")

labels  = ["H", "D", "A"]

print("\nClassification report: ")

print(
    classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=[
            "Home Win",
            "Draw",
            "Away Win"
        ],
        zero_division=0
    )
)

matrix =confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

matrix_table = pd.DataFrame(
    matrix,
    index=[
        "Actual Home Win",
        "Actual Draw",
        "Actual Away Win"
    ],
    columns=[
        "Predicted Home Win",
        "Predicted Draw",
        "Predicted Away Win"
    ]
)

print("\nConfusion matrix: ")
print(matrix_table)

#TAO BANG KET QUA CUNG XAC SUAT DU DOAN

result_names = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win"
}

results = test_data.reset_index(drop=True)

# The model stores probabilities in A, D, H order (alphabetical),
# so name each column by its class and pick them in H, D, A order.
probability = pd.DataFrame(
    y_probability * 100,
    columns=logistic_model.classes_
)

match_names = results["home_team"] + " vs " + results["away_team"]

prediction_table = pd.DataFrame({
    "Date": results["date"].dt.strftime("%d %b %Y"),
    "Match": match_names,
    "Score": (
        results["home_goals"].astype(str)
        + "-"
        + results["away_goals"].astype(str)
    ),
    "Form H-A": (
        results["home_form_points_5"].map("{:.1f}".format)
        + " - "
        + results["away_form_points_5"].map("{:.1f}".format)
    ),
    "AH line": results["asian_handicap_line"].map("{:+.2f}".format),
    "Over 2.5 %": results["odds_over_2_5_probability"] * 100,
    "Goals": (
        results["home_goals"] + results["away_goals"]
    ).map(lambda goals: f"{goals} ({'Over' if goals > 2 else 'Under'})"),
    "Actual": results["result"].map(result_names),
    "Predicted": pd.Series(y_pred).map(result_names),
    "Home %": probability["H"],
    "Draw %": probability["D"],
    "Away %": probability["A"],
    "Confidence %": probability.max(axis=1),
    "Correct": (results["result"] == y_pred).map({
        True: "yes",
        False: "no"
    })
})

# Pad text columns so they line up on the left
for column in prediction_table.select_dtypes(exclude="number"):
    text = prediction_table[column]
    prediction_table[column] = text.str.ljust(text.str.len().max())

rows_to_show = 15
shown = prediction_table.head(rows_to_show)
correct_count = (shown["Correct"] == "yes").sum()

print(f"\nFirst {rows_to_show} test predictions:")
print("(Form H-A = average points per game over the last 5 matches)")
print("(AH line = Asian handicap for the home team, -1.50 = home gives 1.5 goals)")
print("(Over 2.5 % = bookmaker chance of 3+ goals; Goals = what happened)\n")

print(
    shown.to_string(
        index=False,
        justify="left",
        float_format="{:.1f}".format
    )
)

print(
    f"\nCorrect in this sample: {correct_count} / {len(shown)} "
    f"({correct_count / len(shown):.0%})"
)
