import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)

# =========================
# PATHS
# =========================
BASE_OUTPUT_DIR = Path("../output_v2")
MODEL_OUTPUT_DIR = BASE_OUTPUT_DIR / "logistic_regression"
MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# LOAD DATA
# =========================
X_train = np.load(BASE_OUTPUT_DIR / "X_train_pca.npy")
X_test = np.load(BASE_OUTPUT_DIR / "X_test_pca.npy")
y_train = np.load(BASE_OUTPUT_DIR / "y_train.npy")
y_test = np.load(BASE_OUTPUT_DIR / "y_test.npy")

label_encoder = joblib.load(BASE_OUTPUT_DIR / "label_encoder.pkl")
class_names = list(label_encoder.classes_)

# =========================
# MODEL
# =========================
model = LogisticRegression(
    max_iter=5000,
    C=3.0,
    solver="lbfgs",
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# =========================
# METRICS
# =========================
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

report_text = classification_report(
    y_test, y_pred, target_names=class_names, zero_division=0
)

report_dict = classification_report(
    y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True
)

# =========================
# SAVE METRICS
# =========================
metrics_df = pd.DataFrame([{
    "Model": "logistic_regression_v2",
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1_Score": f1
}])
metrics_df.to_csv(MODEL_OUTPUT_DIR / "metrics.csv", index=False)

with open(MODEL_OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
    f.write(report_text)

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(ax=ax, xticks_rotation=45, cmap="Blues", colorbar=False)
plt.title("Confusion Matrix - Logistic Regression")
plt.tight_layout()
plt.savefig(MODEL_OUTPUT_DIR / "confusion_matrix.png")
plt.close()

# =========================
# OVERALL METRICS CHART
# =========================
metric_names = ["Accuracy", "Precision", "Recall", "F1_Score"]
values = [accuracy, precision, recall, f1]

plt.figure(figsize=(8, 5))
bars = plt.bar(metric_names, values)
plt.ylim(0, 1)
plt.title("Overall Metrics - Logistic Regression")
plt.ylabel("Score")

for bar, value in zip(bars, values):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value:.4f}",
        ha="center",
        va="bottom"
    )

plt.tight_layout()
plt.savefig(MODEL_OUTPUT_DIR / "overall_metrics.png")
plt.close()

# =========================
# PER-CLASS F1 CHART
# =========================
rows = []
for key, value in report_dict.items():
    if isinstance(value, dict) and key not in ["macro avg", "weighted avg"]:
        rows.append({
            "class": key,
            "f1-score": value.get("f1-score", 0)
        })

report_df = pd.DataFrame(rows)

plt.figure(figsize=(10, 6))
bars = plt.bar(report_df["class"], report_df["f1-score"])
plt.ylim(0, 1)
plt.title("Per-Class F1 Score - Logistic Regression")
plt.ylabel("F1 Score")
plt.xticks(rotation=45)

for bar, value in zip(bars, report_df["f1-score"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 0.01,
        f"{value:.4f}",
        ha="center",
        va="bottom",
        fontsize=8
    )

plt.tight_layout()
plt.savefig(MODEL_OUTPUT_DIR / "per_class_f1.png")
plt.close()

# =========================
# SAVE MODEL
# =========================
joblib.dump(model, MODEL_OUTPUT_DIR / "model.pkl")

print("\n===== Logistic Regression Results =====")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")
print("\nClassification Report:\n")
print(report_text)
print(f"\nSaved all outputs to: {MODEL_OUTPUT_DIR}")