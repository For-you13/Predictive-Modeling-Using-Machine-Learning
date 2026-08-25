# Part 2 — Predictive Modeling Using Machine Learning

## Retail Data Science Project

This is **Part 2 of 4**.

The goal is to build machine-learning models that predict whether a retail order will be profitable.

## Models used

1. Logistic Regression
2. Decision Tree
3. Random Forest

## Evaluation metrics

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Confusion Matrix
- ROC Curve

## Machine-learning problem

The target variable is:

```text
Profitable
    1 → Profit > 0
    0 → Profit <= 0
```

The `Profit` column is **not used as an input feature** because it directly determines the target. This prevents target leakage and makes the experiment more realistic.

## Folder structure

```text
Part-2-Predictive-Modeling/
│
├── data/
│   └── retail_sales_raw.csv
│
├── models/
│   └── best_model.pkl
│
├── results/
│   ├── model_comparison.csv
│   └── model_report.txt
│
├── visualizations/
│   ├── confusion_matrix.png
│   ├── roc_curve_comparison.png
│   └── model_performance.png
│
├── train_model.py
├── requirements.txt
└── README.md
```

## How to run

Install the libraries:

```bash
pip install -r requirements.txt
```

Run the model:

```bash
python train_model.py
```

The program will:

```text
Load Data
   ↓
Clean Data
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Train 3 ML Models
   ↓
Evaluate Models
   ↓
Compare Results
   ↓
Select Best Model
   ↓
Save Model + Visualizations
```

## Output

The program creates:

- `best_model.pkl`
- `model_comparison.csv`
- `model_report.txt`
- Confusion matrix
- ROC curve comparison
- Model performance chart

## Learning outcomes

This part demonstrates:

- Supervised machine learning
- Classification
- Train/test splitting
- Feature engineering
- Categorical encoding
- Feature scaling
- Model comparison
- Performance evaluation
- Model persistence

**Next:** Part 3 — Exploratory Data Analysis (EDA).
