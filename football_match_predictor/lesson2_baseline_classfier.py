from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix
)


BASE_DIR = Path(__file__).resolve().parent

data_path = (
    BASE_DIR
    / "data"
    / "processed"
    / "premier_league_matches.csv"
)

matches = pd.read_csv(
    data_path,
    parse_dates=["date"]
)

matches = matches.sort_values(
    "date"
).reset_index(drop=True)


# Dùng các mùa cũ để train
train_data = matches.loc[
    matches["season"] != "2025-2026"
].copy()

# Dùng mùa mới nhất để test
test_data = matches.loc[
    matches["season"] == "2025-2026"
].copy()


print("Training seasons:")
print(sorted(train_data["season"].unique()))

print("\nTesting season:")
print(sorted(test_data["season"].unique()))

print("\nTraining matches:", len(train_data))
print("Testing matches:", len(test_data))


# Baseline không dùng thông tin thật của trận đấu.
# Cột dummy chỉ giúp scikit-learn nhận đúng định dạng X.
X_train = pd.DataFrame({
    "dummy": [0] * len(train_data)
})

X_test = pd.DataFrame({
    "dummy": [0] * len(test_data)
})

y_train = train_data["result"]
y_test = test_data["result"]


print("\nTraining result distribution:")
print(
    y_train.value_counts()
    .rename(index={
        "H": "Home Win",
        "D": "Draw",
        "A": "Away Win"
    })
)


baseline_model = DummyClassifier(
    strategy="most_frequent"
)

baseline_model.fit(
    X_train,
    y_train
)

y_pred = baseline_model.predict(X_test)


accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)


print("\nBaseline always predicts:")
print(y_pred[0])

print(f"\nAccuracy: {accuracy:.3f}")
print(
    f"Balanced accuracy: "
    f"{balanced_accuracy:.3f}"
)


labels = ["H", "D", "A"]

report = classification_report(
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

print("\nClassification report:")
print(report)


matrix = confusion_matrix(
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

print("\nConfusion matrix:")
print(matrix_table)