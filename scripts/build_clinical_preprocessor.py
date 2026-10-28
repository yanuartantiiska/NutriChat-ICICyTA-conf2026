import json
from pathlib import Path

import pandas as pd
import yaml


CONFIG_PATH = Path("config/clinical_features.yaml")
OUT_PATH = Path("config/clinical_preprocessor.json")


# =========================================================
# Load configuration
# =========================================================

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

profile_path = Path(config["data"]["profile_file"])
mapping_path = Path(config["data"]["mapping_file"])

numeric_features = config["numeric_features"]
categorical_features = config["categorical_features"]


# =========================================================
# Load data
# =========================================================

clinical = pd.read_csv(profile_path)
mapping = pd.read_csv(mapping_path)

df = mapping.merge(
    clinical,
    on="clinical_id",
    how="left",
    validate="one_to_one",
)

train = df[df["split"] == "train"].copy()
val = df[df["split"] == "val"].copy()
test = df[df["split"] == "test"].copy()


print("=" * 78)
print("NUTRICHAT — CLINICAL PREPROCESSOR")
print("=" * 78)

print("\nDataset:")
print("Train :", len(train))
print("Val   :", len(val))
print("Test  :", len(test))


# =========================================================
# Validation
# =========================================================

required = numeric_features + categorical_features

missing_columns = [
    col for col in required
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns: {missing_columns}"
    )

assert len(train) == 2443
assert len(val) == 523
assert len(test) == 524

assert train[required].isna().sum().sum() == 0
assert val[required].isna().sum().sum() == 0
assert test[required].isna().sum().sum() == 0


# =========================================================
# Numeric statistics — TRAIN ONLY
# =========================================================

numeric_stats = {}

print("\nNUMERIC STATISTICS — TRAIN ONLY")
print("-" * 78)

for col in numeric_features:

    mean = float(train[col].mean())

    # pandas sample standard deviation (ddof=1)
    std = float(train[col].std())

    if std <= 0:
        raise ValueError(
            f"Invalid standard deviation for {col}: {std}"
        )

    numeric_stats[col] = {
        "mean": mean,
        "std": std,
    }

    print(
        f"{col:20s} "
        f"mean={mean:10.4f} "
        f"std={std:10.4f}"
    )


# =========================================================
# Categorical vocabulary — TRAIN ONLY
#
# 0 is reserved for unknown category.
# Known categories start from 1.
# =========================================================

categorical_vocab = {}

print("\nCATEGORICAL VOCABULARY — TRAIN ONLY")
print("-" * 78)

for col in categorical_features:

    values = sorted(
        train[col]
        .astype(str)
        .unique()
        .tolist()
    )

    vocab = {
        value: idx + 1
        for idx, value in enumerate(values)
    }

    categorical_vocab[col] = vocab

    print(f"{col:20s} {vocab}")


# =========================================================
# Check unseen categories in validation/test
# =========================================================

print("\nUNSEEN CATEGORY CHECK")
print("-" * 78)

unseen_summary = {}

for col in categorical_features:

    known = set(
        categorical_vocab[col].keys()
    )

    val_values = set(
        val[col].astype(str).unique()
    )

    test_values = set(
        test[col].astype(str).unique()
    )

    unseen_val = sorted(
        val_values - known
    )

    unseen_test = sorted(
        test_values - known
    )

    unseen_summary[col] = {
        "val": unseen_val,
        "test": unseen_test,
    }

    print(
        f"{col:20s} "
        f"val_unseen={unseen_val} "
        f"test_unseen={unseen_test}"
    )


# =========================================================
# Save preprocessing specification
# =========================================================

output = {
    "fit_split": "train",
    "train_rows": len(train),

    "numeric_features": numeric_features,
    "categorical_features": categorical_features,

    "numeric_stats": numeric_stats,

    "categorical_vocab": categorical_vocab,

    "unknown_category_index": 0,

    "unseen_categories": unseen_summary,

    "data_type": "synthetic_clinical_profiles",
}


OUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

with open(
    OUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        output,
        f,
        indent=2,
        ensure_ascii=False,
    )


print("\nSaved:")
print(OUT_PATH.resolve())

print("\nCLINICAL PREPROCESSOR: PASS")