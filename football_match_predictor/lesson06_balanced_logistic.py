from pathlib import Path

import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent


data_path = BASE_DIR / "data" / "processed" / "premier_league_form_features.csv"
matches = pd.read_csv(
    data_path,
    parse_dates=["date"]
)

matches  = matches.sort_values(
    "date"
).reset_index(drop=True)

features = [
     "home_form_points_5",
    "home_form_goals_for_5",
    "home_form_goals_against_5",
    "away_form_points_5",
    "away_form_goals_for_5",
    "away_form_goals_against_5"
]

test_season = "2025-2026"

train_data = matches.loc[
    matches["season"] != test_season
].copy()

test_data = matches.loc[
    matches["season"] == test_season
].copy()


X_train = train_data[features]
y_train = train_data["result"]

X_test = test_data[features]
y_test = test_data["result"]

baseline_model = DummyClassifier(
    strategy="most_frequent"
)

normal_model = Pipeline([
    (
        "scaler", StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])

balanced_model = Pipeline([
    (
        "scaler",
        StandardScaler()
    ),
    (
        "model",
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42
        )
    )
])

models = {
    "Baseline": baseline_model,
    "Normal Logistic": normal_model,
    "Balanced Logistic": balanced_model
}

comparison_rows = []
predictions = {}
probability = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)

    predictions[model_name] = y_pred
    probability[model_name] = y_probability

    report = classification_report(
        y_test,
        y_pred,
        labels=["H", "D", "A"],
        output_dict=True,
        zero_division=0
    )

    comparison_rows.append({
        "Model": model_name,
        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),
        "Balanced Accuracy": balanced_accuracy_score(
                 y_test,
                 y_pred             
        ),
        "Macro F1": f1_score(
            y_test,
            y_pred,
            labels=["H", "D", "A"],
            average="macro",
            zero_division=0 
        ),
        "Draw Recall": report["D"]["recall"],
        "Log Loss": log_loss(
            y_test,
            y_probability,
            labels=model.classes_
        )
    })

labels = ["H", "D", "A"]
label_names = {"H": "Home Win", "D": "Draw", "A": "Away Win"}


def percent(value):
    return f"{value * 100:.1f}%"


# Bảng so sánh model, các chỉ số hiển thị dạng %
comparison = pd.DataFrame(comparison_rows)

comparison_table = comparison.copy()
for column in ["Accuracy", "Balanced Accuracy", "Macro F1", "Draw Recall"]:
    comparison_table[column] = comparison_table[column].map(percent)
comparison_table["Log Loss"] = comparison_table["Log Loss"].round(3)

print(f"\nModel Comparison (test season {test_season}, {len(test_data)} matches):")
print(comparison_table.to_string(index=False))

# Tỉ lệ kết quả: thực tế vs mỗi model dự đoán
share_rows = [{
    "Model": "Actual",
    **{label_names[label]: percent((y_test == label).mean()) for label in labels}
}]
for model_name, y_pred in predictions.items():
    predicted = pd.Series(y_pred)
    share_rows.append({
        "Model": model_name,
        **{label_names[label]: percent((predicted == label).mean()) for label in labels}
    })

print("\nResult share (actual vs predicted):")
print(pd.DataFrame(share_rows).to_string(index=False))

balanced_pred = predictions[
    "Balanced Logistic"
]

print("\nBalanced Logistic classification report: ")

print(
    classification_report(
        y_test,
        balanced_pred,
        labels=["H", "D", "A"],
    
    target_names=[
        "Home Win",
        "Draw",
        "Away Win"
    ],
    zero_division=0
    )
)
matrix = confusion_matrix(
    y_test,
    balanced_pred,
    labels=["H", "D", "A"]
)

matrix_table =pd.DataFrame(
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

print("\nBalanced Logistic confusion matrix: ")
print(matrix_table)

# Mỗi hàng cộng lại = 100%: ví dụ trong các trận hoà thật, bao nhiêu % bị đoán là H / D / A
matrix_percent = matrix_table.div(matrix_table.sum(axis=1), axis=0).map(percent)

print("\nBalanced Logistic confusion matrix (% of each actual result):")
print(matrix_percent)

balanced_probability = probability[
    "Balanced Logistic"
]

results = test_data[
    [
        "date",
        "home_team",
        "away_team",
        "result"
    ]
].reset_index(drop=True)

results = results.rename(columns={
    "result": "actual_result"
})

results["predicted_result"] = balanced_pred

class_names = {
    "H": "home_probability",
    "D": "draw_probability",
    "A": "away_probability"
}

for index, result_class in enumerate(
    balanced_model.classes_
):
    column_name = class_names[
        result_class
    ]

    results[column_name] = (
        balanced_probability[:, index] * 100
    ).round(1)

predicted_draws = results.loc[
    results["predicted_result"] == "D"
].copy()

print(
    "\nNumber of matches predicted as draw: ",
    len(predicted_draws) 
)

print("\nFirst 15 predicted draws: ")

if predicted_draws.empty:
    print("The model still predicts no draws.")
else: 
    print(
        predicted_draws.head(15).to_string(index=False)
    )