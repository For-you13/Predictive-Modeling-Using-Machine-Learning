"""
PART 2 — PREDICTIVE MODELING USING MACHINE LEARNING

Goal:
Predict whether a retail order will be profitable.

Target:
    Profitable = 1 if Profit > 0 else 0
    Profitable = 0 otherwise

Models:
    1. Logistic Regression
    2. Decision Tree
    3. Random Forest

Metrics:
    Accuracy, Precision, Recall, F1, ROC-AUC

Visualizations:
    Confusion Matrix
    ROC Curves
    Model Comparison
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

BASE = Path(__file__).resolve().parent
DATA_FILE = BASE / "data" / "retail_sales_raw.csv"
MODEL_DIR = BASE / "models"
RESULT_DIR = BASE / "results"
VIZ_DIR = BASE / "visualizations"

MODEL_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)

# -------------------------------------------------
# 1. Load and clean data
# -------------------------------------------------
df = pd.read_csv(DATA_FILE)

df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

numeric_columns = ["Quantity", "Sales", "Discount", "Profit", "Shipping_Days"]
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.drop_duplicates().copy()

for col in df.select_dtypes(include="object").columns:
    if df[col].isna().any():
        mode = df[col].mode()
        df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")

for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].median())

# IQR outlier clipping
for col in numeric_columns:
    q1 = df[col].quantile(.25)
    q3 = df[col].quantile(.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    df[col] = df[col].clip(low, high)

# Feature engineering
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Sales_per_Unit"] = df["Sales"] / df["Quantity"].replace(0, np.nan)
df["Sales_per_Unit"] = df["Sales_per_Unit"].fillna(0)

# Target
df["Profitable"] = (df["Profit"] > 0).astype(int)

print("=" * 65)
print("PART 2 — PREDICTIVE MODELING USING MACHINE LEARNING")
print("=" * 65)
print("\nTarget distribution:")
print(df["Profitable"].value_counts())

# -------------------------------------------------
# 2. Prepare features and target
# -------------------------------------------------
# Profit is deliberately excluded because it directly determines the target.
drop_columns = [
    "Profitable", "Profit", "Order_ID", "Order_Date"
]

X = df.drop(columns=drop_columns)
y = df["Profitable"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

numeric_features = X_train.select_dtypes(include="number").columns.tolist()
categorical_features = X_train.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)

# -------------------------------------------------
# 3. Define models
# -------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=6,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        random_state=42
    )
}

results = []
fitted_models = {}

# -------------------------------------------------
# 4. Train and evaluate
# -------------------------------------------------
for name, algorithm in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", algorithm)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(
            y_test, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_test, predictions, zero_division=0
        ),
        "F1_Score": f1_score(
            y_test, predictions, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(
            y_test, probabilities
        )
    })

    fitted_models[name] = pipeline

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    "ROC_AUC", ascending=False
).reset_index(drop=True)

results_df.to_csv(
    RESULT_DIR / "model_comparison.csv",
    index=False
)

# -------------------------------------------------
# 5. Save best model
# -------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]

import joblib
joblib.dump(
    best_model,
    MODEL_DIR / "best_model.pkl"
)

print("\n--- MODEL COMPARISON ---")
print(results_df.to_string(index=False))
print(f"\nBest model: {best_model_name}")

# -------------------------------------------------
# 6. Confusion Matrix for best model
# -------------------------------------------------
best_predictions = best_model.predict(X_test)

cm = confusion_matrix(y_test, best_predictions)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Profitable", "Profitable"]
).plot(ax=ax)

ax.set_title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()
plt.savefig(
    VIZ_DIR / "confusion_matrix.png",
    dpi=150
)
plt.close()

# -------------------------------------------------
# 7. ROC Curve for all models
# -------------------------------------------------
plt.figure(figsize=(9, 6))

for name, model in fitted_models.items():

    probabilities = model.predict_proba(X_test)[:, 1]

    fpr, tpr, _ = roc_curve(
        y_test,
        probabilities
    )

    roc_score = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{name} (AUC = {roc_score:.2f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()

plt.savefig(
    VIZ_DIR / "roc_curve_comparison.png",
    dpi=150
)
plt.close()

# -------------------------------------------------
# 8. Model performance chart
# -------------------------------------------------
plot_df = results_df.set_index("Model")[
    ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
]

ax = plot_df.plot(
    kind="bar",
    figsize=(11, 6)
)

ax.set_title("Machine Learning Model Performance")
ax.set_ylabel("Score")
ax.set_ylim(0, 1)
plt.xticks(rotation=15)
plt.legend(loc="lower right")
plt.tight_layout()

plt.savefig(
    VIZ_DIR / "model_performance.png",
    dpi=150
)
plt.close()

# -------------------------------------------------
# 9. Save detailed report
# -------------------------------------------------
report = f"""PART 2 — PREDICTIVE MODELING REPORT
==========================================

Problem:
Predict whether a retail order will be profitable.

Target:
Profitable = 1 when Profit > 0
Profitable = 0 otherwise

Dataset:
Retail sales dataset

Train/Test split:
80% training
20% testing

Models tested:
1. Logistic Regression
2. Decision Tree
3. Random Forest

Best model:
{best_model_name}

MODEL RESULTS
------------------------------------------
{results_df.to_string(index=False)}

Important modeling decision:
The Profit column was removed from the model features because
Profit directly determines the Profitable target. Keeping it would
cause target leakage and produce an unrealistic model.

Generated files:
- models/best_model.pkl
- results/model_comparison.csv
- visualizations/confusion_matrix.png
- visualizations/roc_curve_comparison.png
- visualizations/model_performance.png
"""

(RESULT_DIR / "model_report.txt").write_text(
    report,
    encoding="utf-8"
)

print("\nPart 2 complete!")
