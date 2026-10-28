import random
import re
from pathlib import Path

import pandas as pd
import yaml

from src.dataset.multimodal_text_templates import (
    TEMPLATES,
    FOOD_REFERENCES,
    CONDITION_EXPRESSIONS,
    QUERY_PREFIXES,
    validate_templates,
)

SEED = 42

GROUNDING_FILE = Path(
    "data/processed/multimodal_grounding.csv"
)

DESIGN_FILE = Path(
    "config/multimodal_text_design.yaml"
)

OUT_FILE = Path(
    "data/processed/multimodal_dataset_v2.csv"
)


# =========================================================
# Helpers
# =========================================================

def normalize_text(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([?.!,])", r"\1", text)
    return text.strip()


# =========================================================
# Load
# =========================================================

random.seed(SEED)

validate_templates()

grounding = pd.read_csv(GROUNDING_FILE)

with open(
    DESIGN_FILE,
    "r",
    encoding="utf-8",
) as f:
    design = yaml.safe_load(f)


explicit_probability = float(
    design["condition_mention"][
        "explicit_probability"
    ]
)


print("=" * 78)
print(
    "NUTRICHAT — GENERATE EXP04 MULTIMODAL QUERIES"
)
print("=" * 78)

print("\nSeed:", SEED)
print("Grounding rows:", len(grounding))


# =========================================================
# Balanced assignment pools
# =========================================================

query_types = list(TEMPLATES.keys())

family_lookup = {}

for query_type, families in TEMPLATES.items():
    family_lookup[query_type] = list(
        families.keys()
    )


# Shuffle record processing order only for assignment.
# Final output will be restored to original grounding order.

indices = list(range(len(grounding)))
random.shuffle(indices)


query_counter = {
    q: 0
    for q in query_types
}

family_counter = {
    family:
        0
    for q in query_types
    for family in family_lookup[q]
}


generated = {}

used_normalized_texts = set()


# =========================================================
# Generate one query per record
# =========================================================

for position, idx in enumerate(indices):

    row = grounding.iloc[idx]

    # -----------------------------------------------------
    # Query type:
    # round-robin after randomized record ordering
    # -----------------------------------------------------

    query_type = query_types[
        position % len(query_types)
    ]

    query_counter[
        query_type
    ] += 1


    # -----------------------------------------------------
    # Template family:
    # choose least-used family inside query type
    # -----------------------------------------------------

    candidate_families = (
        family_lookup[query_type]
    )

    min_count = min(
        family_counter[f]
        for f in candidate_families
    )

    least_used = [
        f
        for f in candidate_families
        if family_counter[f] == min_count
    ]

    family_id = random.choice(
        least_used
    )

    family_counter[
        family_id
    ] += 1


    # -----------------------------------------------------
    # Clinical group
    # -----------------------------------------------------

    clinical_group = str(
        row["clinical_group"]
    ).strip()

    if clinical_group not in (
        CONDITION_EXPRESSIONS
    ):
        raise ValueError(
            "Unknown clinical group: "
            f"{clinical_group}"
        )


    # -----------------------------------------------------
    # Explicit / implicit condition mention
    # -----------------------------------------------------

    condition_explicit = (
        random.random()
        < explicit_probability
    )

    if condition_explicit:

        condition_ref = random.choice(
            CONDITION_EXPRESSIONS[
                clinical_group
            ]
        )

    else:

        condition_ref = ""


    # -----------------------------------------------------
    # Generate unique surface form
    # -----------------------------------------------------

    templates = TEMPLATES[
        query_type
    ][family_id]

    success = False

    # Multiple attempts because the combination space
    # can occasionally produce an existing utterance.

    for _ in range(200):

        template = random.choice(
            templates
        )

        food_ref = random.choice(
            FOOD_REFERENCES
        )

        if condition_explicit:

            condition_ref = random.choice(
                CONDITION_EXPRESSIONS[
                    clinical_group
                ]
            )

        else:
            condition_ref = ""

        prefix = random.choice(
            QUERY_PREFIXES
        )

        text = template.format(
            food_ref=food_ref,
            condition_ref=condition_ref,
        )

        text = prefix + text

        text = clean_text(text)

        normalized = normalize_text(
            text
        )

        if normalized not in (
            used_normalized_texts
        ):
            used_normalized_texts.add(
                normalized
            )

            success = True
            break


    if not success:
        raise RuntimeError(
            "Unable to generate unique query "
            f"for record {row['record_id']}."
        )


    generated[idx] = {
        "user_utterance": text,
        "query_type": query_type,
        "template_family": family_id,
        "condition_explicit":
            condition_explicit,
    }


# =========================================================
# Attach generated text in ORIGINAL grounding order
# =========================================================

out = grounding.copy()

out["user_utterance"] = [
    generated[i]["user_utterance"]
    for i in range(len(out))
]

out["query_type"] = [
    generated[i]["query_type"]
    for i in range(len(out))
]

out["template_family"] = [
    generated[i]["template_family"]
    for i in range(len(out))
]

out["condition_explicit"] = [
    generated[i]["condition_explicit"]
    for i in range(len(out))
]


# =========================================================
# Audit
# =========================================================

normalized = (
    out["user_utterance"]
    .map(normalize_text)
)


print("\nROWS")
print("-" * 78)

print("Total:", len(out))

print(
    "Unique record_id:",
    out["record_id"].nunique(),
)

print(
    "Unique clinical_id:",
    out["clinical_id"].nunique(),
)

print(
    "Unique utterances:",
    normalized.nunique(),
)

print(
    "Exact normalized duplicates:",
    int(normalized.duplicated().sum()),
)


print("\nQUERY TYPE DISTRIBUTION")
print("-" * 78)

print(
    out["query_type"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nTEMPLATE FAMILY DISTRIBUTION")
print("-" * 78)

print(
    out["template_family"]
    .value_counts()
    .sort_index()
    .to_string()
)


print("\nCONDITION MENTION")
print("-" * 78)

explicit_count = int(
    out["condition_explicit"].sum()
)

print(
    "Explicit:",
    explicit_count,
)

print(
    "Implicit:",
    len(out) - explicit_count,
)

print(
    "Explicit %:",
    round(
        explicit_count
        / len(out)
        * 100,
        2,
    ),
)


print("\nCLINICAL GROUP")
print("-" * 78)

print(
    out["clinical_group"]
    .value_counts()
    .to_string()
)


print("\nFROZEN SPLIT")
print("-" * 78)

print(
    out["split"]
    .value_counts()
    .to_string()
)


print("\nSAMPLE QUERIES")
print("-" * 78)

print(
    out[
        [
            "record_id",
            "split",
            "clinical_group",
            "query_type",
            "template_family",
            "condition_explicit",
            "user_utterance",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# =========================================================
# Leakage / prohibited literal audit
# =========================================================

# These column VALUES must never have been directly
# inserted into the generated text.
#
# We do not compare common words such as numeric values
# because accidental string overlap would be meaningless.
# Ingredient names are checked record-wise.

ingredient_leak_count = 0

for _, row in out.iterrows():

    utterance = normalize_text(
        row["user_utterance"]
    )

    ingredients = str(
        row["ingredient_names"]
    ).split("|")

    for ingredient in ingredients:

        ingredient = normalize_text(
            ingredient
        )

        if (
            ingredient
            and len(ingredient) >= 4
            and ingredient in utterance
        ):
            ingredient_leak_count += 1
            break


print("\nLEAKAGE AUDIT")
print("-" * 78)

print(
    "Ingredient-name leakage:",
    ingredient_leak_count,
)


# =========================================================
# Assertions
# =========================================================

assert len(out) == 3490

assert (
    out["record_id"].nunique()
    == 3490
)

assert (
    out["clinical_id"].nunique()
    == 3490
)

assert normalized.nunique() == 3490

assert normalized.duplicated().sum() == 0

assert ingredient_leak_count == 0

assert (
    set(out["query_type"].unique())
    == set(query_types)
)

assert (
    out["template_family"].nunique()
    == 20
)

assert (
    out["split"]
    .value_counts()
    .to_dict()
    ==
    {
        "train": 2443,
        "test": 524,
        "val": 523,
    }
)


# =========================================================
# Save
# =========================================================

OUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

out.to_csv(
    OUT_FILE,
    index=False,
    encoding="utf-8-sig",
)


print("\nSaved:")
print(OUT_FILE.resolve())

print(
    "\nEXP04 MULTIMODAL QUERY GENERATION: PASS"
)