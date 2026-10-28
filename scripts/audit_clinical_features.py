import pandas as pd
import numpy as np
import yaml
from pathlib import Path

CONFIG = Path("config/clinical_features.yaml")

print("=" * 78)
print("NUTRICHAT — CLINICAL FEATURE AUDIT")
print("=" * 78)

# ---------------------------------------------------------
# Load configuration
# ---------------------------------------------------------

with open(CONFIG, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

profile_path = Path(config["data"]["profile_file"])
mapping_path = Path(config["data"]["mapping_file"])

numeric_features = config["numeric_features"]
categorical_features = config["categorical_features"]

clinical = pd.read_csv(profile_path)
mapping = pd.read_csv(mapping_path)

# ---------------------------------------------------------
# Merge split information
# ---------------------------------------------------------

df = mapping.merge(
    clinical,
    on="clinical_id",
    how="left",
    validate="one_to_one",
)

print("\n1. DATASET")
print("-" * 78)
print("Rows              :", len(df))
print("Clinical IDs      :", df["clinical_id"].nunique())
print("Missing profiles  :", df["age"].isna().sum())

print("\nSplit:")
print(df["split"].value_counts().to_string())

# ---------------------------------------------------------
# Numeric summary
# ---------------------------------------------------------

print("\n2. NUMERIC FEATURE SUMMARY")
print("-" * 78)

summary = (
    df[numeric_features]
    .describe()
    .T[
        [
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ]
    ]
    .round(3)
)

print(summary.to_string())

# ---------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------

print("\n3. PEARSON CORRELATION MATRIX")
print("-" * 78)

corr = (
    df[numeric_features]
    .corr(method="pearson")
    .round(3)
)

print(corr.to_string())

# ---------------------------------------------------------
# Strong correlations
# ---------------------------------------------------------

print("\n4. STRONG NUMERIC CORRELATIONS |r| >= 0.70")
print("-" * 78)

strong_pairs = []

for i, col1 in enumerate(numeric_features):
    for col2 in numeric_features[i + 1:]:

        r = corr.loc[col1, col2]

        if abs(r) >= 0.70:
            strong_pairs.append(
                (col1, col2, float(r))
            )

if strong_pairs:
    for col1, col2, r in strong_pairs:
        print(
            f"{col1:20s} <-> "
            f"{col2:20s} r={r: .3f}"
        )
else:
    print("None")

# ---------------------------------------------------------
# BMI consistency
# ---------------------------------------------------------

print("\n5. BMI REDUNDANCY CHECK")
print("-" * 78)

calculated_bmi = (
    df["weight_kg"]
    /
    ((df["height_cm"] / 100.0) ** 2)
)

bmi_difference = (
    calculated_bmi - df["bmi"]
).abs()

print(
    "Mean absolute BMI difference :",
    round(bmi_difference.mean(), 4)
)

print(
    "Max absolute BMI difference  :",
    round(bmi_difference.max(), 4)
)

print(
    "BMI approximately derivable  :",
    bool(bmi_difference.mean() < 0.1)
)

# ---------------------------------------------------------
# Condition relation diagnostic
# descriptive only
# ---------------------------------------------------------

print("\n6. CONDITION / SOURCE FEATURE RELATION")
print("-" * 78)

for condition in [
    "hypertension",
    "dyslipidemia",
]:

    print(f"\n{condition.upper()}")

    grouped = (
        df
        .groupby(condition)[
            [
                "sbp",
                "dbp",
                "ldl",
                "hdl",
                "triglycerides",
            ]
        ]
        .mean()
        .round(2)
    )

    print(grouped.to_string())

# ---------------------------------------------------------
# Categorical distributions
# ---------------------------------------------------------

print("\n7. CATEGORICAL DISTRIBUTIONS")
print("-" * 78)

for col in categorical_features:

    print(f"\n{col}:")

    counts = df[col].value_counts(
        dropna=False
    )

    percentages = (
        df[col]
        .value_counts(
            normalize=True,
            dropna=False,
        )
        * 100
    ).round(2)

    table = pd.DataFrame({
        "count": counts,
        "percent": percentages,
    })

    print(table.to_string())

# ---------------------------------------------------------
# Split numeric means
# ---------------------------------------------------------

print("\n8. NUMERIC MEANS BY FROZEN SPLIT")
print("-" * 78)

print(
    df
    .groupby("split")[numeric_features]
    .mean()
    .round(2)
    .to_string()
)

# ---------------------------------------------------------
# Save correlation matrix
# ---------------------------------------------------------

out = Path(
    "results/tables/"
    "exp03_clinical_feature_correlation.csv"
)

out.parent.mkdir(
    parents=True,
    exist_ok=True,
)

corr.to_csv(out)

print("\nSaved:")
print(out.resolve())

print("\nCLINICAL FEATURE AUDIT: PASS")