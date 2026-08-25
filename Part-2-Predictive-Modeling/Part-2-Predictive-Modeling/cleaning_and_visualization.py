"""
PART 1 — DATA CLEANING & VISUALIZATION
Retail Sales Dataset

Run:
    python cleaning_and_visualization.py

The script:
1. Loads raw data
2. Checks missing values and duplicates
3. Removes duplicates
4. Fills missing categorical/numeric values
5. Detects and clips numeric outliers using the IQR method
6. Creates useful columns
7. Saves the cleaned dataset
8. Creates visual reports
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

BASE = Path(__file__).resolve().parent
RAW_FILE = BASE / "data" / "retail_sales_raw.csv"
OUTPUT_DIR = BASE / "output"
VIZ_DIR = BASE / "visualizations"

OUTPUT_DIR.mkdir(exist_ok=True)
VIZ_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")

# -----------------------------
# 1. Load dataset
# -----------------------------
df = pd.read_csv(RAW_FILE)

print("=" * 60)
print("PART 1 — DATA CLEANING & VISUALIZATION")
print("=" * 60)
print(f"Rows loaded: {len(df)}")
print(f"Columns: {len(df.columns)}")

# -----------------------------
# 2. Initial data-quality report
# -----------------------------
print("\n--- Missing Values Before Cleaning ---")
print(df.isnull().sum())

print("\n--- Duplicate Rows Before Cleaning ---")
print(df.duplicated().sum())

# -----------------------------
# 3. Remove duplicates
# -----------------------------
duplicates_removed = int(df.duplicated().sum())
df = df.drop_duplicates().copy()

# -----------------------------
# 4. Correct data types
# -----------------------------
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")

numeric_columns = [
    "Quantity", "Sales", "Discount", "Profit", "Shipping_Days"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# -----------------------------
# 5. Handle missing values
# -----------------------------
# Categorical columns → mode
for column in df.select_dtypes(include="object").columns:
    if df[column].isnull().any():
        mode = df[column].mode()
        fill_value = mode.iloc[0] if not mode.empty else "Unknown"
        df[column] = df[column].fillna(fill_value)

# Numeric columns → median
for column in df.select_dtypes(include=np.number).columns:
    df[column] = df[column].fillna(df[column].median())

# -----------------------------
# 6. Detect and handle outliers
#    IQR method
# -----------------------------
outlier_report = {}

for column in ["Sales", "Quantity", "Discount", "Profit", "Shipping_Days"]:
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1

    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr

    mask = (df[column] < lower_limit) | (df[column] > upper_limit)
    outlier_count = int(mask.sum())

    outlier_report[column] = {
        "outliers": outlier_count,
        "lower_limit": lower_limit,
        "upper_limit": upper_limit
    }

    # Winsorize/clip rather than deleting useful records.
    df[column] = df[column].clip(lower_limit, upper_limit)

# -----------------------------
# 7. Feature processing
# -----------------------------
df["Year"] = df["Order_Date"].dt.year
df["Month"] = df["Order_Date"].dt.month
df["Month_Name"] = df["Order_Date"].dt.strftime("%b")

df["Sales_per_Unit"] = (
    df["Sales"] / df["Quantity"].replace(0, np.nan)
).fillna(0)

# -----------------------------
# 8. Save cleaned dataset
# -----------------------------
clean_file = OUTPUT_DIR / "cleaned_retail_data.csv"
df.to_csv(clean_file, index=False)

# -----------------------------
# 9. Save cleaning report
# -----------------------------
report_lines = [
    "DATA CLEANING REPORT",
    "=" * 50,
    f"Rows before cleaning: {len(df) + duplicates_removed}",
    f"Duplicate rows removed: {duplicates_removed}",
    f"Rows after cleaning: {len(df)}",
    f"Missing values after cleaning: {int(df.isnull().sum().sum())}",
    "",
    "OUTLIERS DETECTED USING IQR",
    "-" * 50
]

for column, info in outlier_report.items():
    report_lines.append(
        f"{column}: {info['outliers']} outliers detected "
        f"(clipped to [{info['lower_limit']:.2f}, {info['upper_limit']:.2f}])"
    )

(OUTPUT_DIR / "cleaning_report.txt").write_text(
    "\n".join(report_lines),
    encoding="utf-8"
)

# -----------------------------
# 10. Visualization 1
# Sales by Category
# -----------------------------
plt.figure(figsize=(9, 5))
category_sales = (
    df.groupby("Category")["Sales"]
    .sum()
    .sort_values(ascending=False)
)
category_sales.plot(kind="bar")
plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(VIZ_DIR / "01_sales_by_category.png", dpi=150)
plt.close()

# -----------------------------
# 11. Visualization 2
# Monthly Sales Trend
# -----------------------------
monthly = (
    df.assign(Month_Period=df["Order_Date"].dt.to_period("M"))
    .groupby("Month_Period")["Sales"]
    .sum()
)

plt.figure(figsize=(12, 5))
monthly.plot(marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(VIZ_DIR / "02_monthly_sales_trend.png", dpi=150)
plt.close()

# -----------------------------
# 12. Visualization 3
# Profit by Region
# -----------------------------
plt.figure(figsize=(8, 5))
region_profit = (
    df.groupby("Region")["Profit"]
    .sum()
    .sort_values(ascending=False)
)
region_profit.plot(kind="bar")
plt.title("Total Profit by Region")
plt.xlabel("Region")
plt.ylabel("Profit")
plt.tight_layout()
plt.savefig(VIZ_DIR / "03_profit_by_region.png", dpi=150)
plt.close()

# -----------------------------
# 13. Visualization 4
# Sales distribution
# -----------------------------
plt.figure(figsize=(9, 5))
sns.histplot(df["Sales"], bins=30, kde=True)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.tight_layout()
plt.savefig(VIZ_DIR / "04_sales_distribution.png", dpi=150)
plt.close()

# -----------------------------
# 14. Visualization 5
# Box plot for Sales
# -----------------------------
plt.figure(figsize=(8, 4.5))
sns.boxplot(x=df["Sales"])
plt.title("Sales Distribution and Outlier Check")
plt.tight_layout()
plt.savefig(VIZ_DIR / "05_sales_boxplot.png", dpi=150)
plt.close()

# -----------------------------
# 15. Visualization 6
# Correlation heatmap
# -----------------------------
numeric_for_corr = [
    "Quantity", "Sales", "Discount", "Profit",
    "Shipping_Days", "Sales_per_Unit"
]

plt.figure(figsize=(9, 6))
sns.heatmap(
    df[numeric_for_corr].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(VIZ_DIR / "06_correlation_heatmap.png", dpi=150)
plt.close()

# -----------------------------
# 16. Final report
# -----------------------------
print("\n--- Cleaning Complete ---")
print(f"Cleaned dataset: {clean_file}")
print(f"Visualizations: {VIZ_DIR}")

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())

print("\n--- Dataset Preview ---")
print(df.head())

print("\n--- Basic Statistics ---")
print(df[["Quantity", "Sales", "Discount", "Profit", "Shipping_Days"]].describe())

print("\nPART 1 COMPLETE!")
