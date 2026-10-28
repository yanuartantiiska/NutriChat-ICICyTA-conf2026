import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def make_profile(rng, index):
    """
    Generate a plausible synthetic clinical profile.

    IMPORTANT:
    These profiles are synthetic and intended only for
    computational/prototype experiments. They are NOT real
    patient records and must not be interpreted clinically.
    """

    clinical_id = f"clinical_{index:05d}"

    # ---------------------------------------------------------
    # Demographic
    # ---------------------------------------------------------

    age = int(
        np.clip(
            rng.normal(50, 15),
            18,
            80,
        )
    )

    sex = rng.choice(
        ["male", "female"]
    )

    height_cm = float(
        np.clip(
            rng.normal(
                168 if sex == "male" else 158,
                7,
            ),
            140,
            190,
        )
    )

    bmi = float(
        np.clip(
            rng.normal(25.5, 4.5),
            17,
            40,
        )
    )

    height_m = height_cm / 100.0

    weight_kg = float(
        bmi * (height_m ** 2)
    )

    # ---------------------------------------------------------
    # Blood pressure
    # ---------------------------------------------------------

    sbp = float(
        np.clip(
            rng.normal(
                118 + max(age - 40, 0) * 0.35,
                15,
            ),
            90,
            190,
        )
    )

    dbp = float(
        np.clip(
            rng.normal(
                76 + max(age - 40, 0) * 0.12,
                10,
            ),
            55,
            120,
        )
    )

    hypertension = int(
        sbp >= 140 or dbp >= 90
    )

    # ---------------------------------------------------------
    # Lipid profile
    # ---------------------------------------------------------

    ldl = float(
        np.clip(
            rng.normal(125, 35),
            50,
            250,
        )
    )

    hdl = float(
        np.clip(
            rng.normal(
                50 if sex == "male" else 55,
                12,
            ),
            20,
            100,
        )
    )

    triglycerides = float(
        np.clip(
            rng.normal(145, 55),
            40,
            400,
        )
    )

    dyslipidemia = int(
        (ldl >= 130)
        or (hdl < 40)
        or (triglycerides >= 150)
    )

    # ---------------------------------------------------------
    # Diabetes flag
    # Prototype variable only
    # ---------------------------------------------------------

    diabetes_probability = (
        0.08
        + 0.005 * max(age - 40, 0)
        + 0.015 * max(bmi - 25, 0)
    )

    diabetes_probability = float(
        np.clip(
            diabetes_probability,
            0.05,
            0.45,
        )
    )

    diabetes = int(
        rng.random()
        < diabetes_probability
    )

    # ---------------------------------------------------------
    # Activity
    # ---------------------------------------------------------

    activity_level = rng.choice(
        [
            "low",
            "moderate",
            "high",
        ],
        p=[
            0.35,
            0.50,
            0.15,
        ],
    )

    # ---------------------------------------------------------
    # Dietary constraints
    # ---------------------------------------------------------

    if hypertension:
        sodium_limit_mg = 1500.0
    else:
        sodium_limit_mg = 2300.0

    # Prototype energy target
    if sex == "male":
        base_energy = 2200.0
    else:
        base_energy = 1800.0

    if activity_level == "low":
        activity_adjustment = -200.0

    elif activity_level == "high":
        activity_adjustment = 300.0

    else:
        activity_adjustment = 0.0

    energy_limit_kcal = (
        base_energy
        + activity_adjustment
    )

    return {
        "clinical_id": clinical_id,

        "age": age,
        "sex": sex,

        "height_cm": round(
            height_cm,
            1,
        ),

        "weight_kg": round(
            weight_kg,
            1,
        ),

        "bmi": round(
            bmi,
            1,
        ),

        "sbp": round(
            sbp,
            0,
        ),

        "dbp": round(
            dbp,
            0,
        ),

        "ldl": round(
            ldl,
            1,
        ),

        "hdl": round(
            hdl,
            1,
        ),

        "triglycerides": round(
            triglycerides,
            1,
        ),

        "hypertension": hypertension,
        "diabetes": diabetes,
        "dyslipidemia": dyslipidemia,

        "activity_level": activity_level,

        "sodium_limit_mg": sodium_limit_mg,

        "energy_limit_kcal": energy_limit_kcal,

        "profile_type": "synthetic",
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Generate reproducible synthetic clinical "
            "profiles for NutriChat prototype experiments."
        )
    )

    parser.add_argument(
        "--splits",
        required=True,
        help=(
            "Input visual_splits.csv containing "
            "record_id and split."
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output clinical_profiles.csv",
    )

    parser.add_argument(
        "--merged_out",
        required=True,
        help=(
            "Output split file merged with "
            "clinical_id."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    # =========================================================
    # Load locked visual split
    # =========================================================

    df = pd.read_csv(
        args.splits
    )

    required = [
        "record_id",
        "split",
    ]

    missing = [
        col
        for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print()
    print("=" * 68)
    print("       NUTRICHAT SYNTHETIC CLINICAL PROFILE GENERATOR")
    print("=" * 68)

    print(
        f"\nInput records : {len(df):,}"
    )

    print("\nOriginal locked split:")

    print(
        df["split"]
        .value_counts()
        .to_string()
    )

    # =========================================================
    # Generate profiles
    # =========================================================

    rng = np.random.default_rng(
        args.seed
    )

    profiles = []

    clinical_ids = []

    for index in range(
        1,
        len(df) + 1,
    ):

        profile = make_profile(
            rng,
            index,
        )

        profiles.append(
            profile
        )

        clinical_ids.append(
            profile["clinical_id"]
        )

    clinical_df = pd.DataFrame(
        profiles
    )

    # =========================================================
    # Attach profile ID WITHOUT changing split
    # =========================================================

    merged = df.copy()

    merged["clinical_id"] = (
        clinical_ids
    )

    # =========================================================
    # Verification
    # =========================================================

    assert len(merged) == len(df)

    assert (
        merged["split"].values
        == df["split"].values
    ).all(), (
        "ERROR: original split changed."
    )

    assert (
        merged["record_id"].values
        == df["record_id"].values
    ).all(), (
        "ERROR: record ordering changed."
    )

    assert (
        merged["clinical_id"]
        .nunique()
        == len(merged)
    )

    # =========================================================
    # Save
    # =========================================================

    out_path = Path(
        args.out
    )

    merged_path = Path(
        args.merged_out
    )

    out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    merged_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    clinical_df.to_csv(
        out_path,
        index=False,
    )

    merged.to_csv(
        merged_path,
        index=False,
    )

    # =========================================================
    # Summary
    # =========================================================

    print(
        f"\nClinical profiles generated : "
        f"{len(clinical_df):,}"
    )

    print("\nClinical condition counts:")

    print(
        f"Hypertension    : "
        f"{clinical_df['hypertension'].sum():,}"
    )

    print(
        f"Dyslipidemia    : "
        f"{clinical_df['dyslipidemia'].sum():,}"
    )

    print(
        f"Diabetes        : "
        f"{clinical_df['diabetes'].sum():,}"
    )

    print("\nSplit after clinical merge:")

    print(
        merged["split"]
        .value_counts()
        .to_string()
    )

    print("\nVerification:")

    print(
        "Record count unchanged : PASS"
    )

    print(
        "Split unchanged        : PASS"
    )

    print(
        "Record order unchanged : PASS"
    )

    print(
        "Unique clinical IDs    : PASS"
    )

    print("\nFiles saved:")

    print(
        out_path.resolve()
    )

    print(
        merged_path.resolve()
    )

    print()
    print(
        "WARNING: All clinical profiles are SYNTHETIC."
    )

    print(
        "They are intended only for computational "
        "prototype experiments, not clinical validation."
    )

    print("=" * 68)


if __name__ == "__main__":
    main()