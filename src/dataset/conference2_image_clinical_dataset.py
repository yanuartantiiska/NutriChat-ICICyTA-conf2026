from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageClinicalNutritionDataset(Dataset):
    """
    Conference 2 - E3 Image + Clinical dataset.

    Authoritative food/image/target source:
        frozen split CSV.

    Clinical linkage:
        record_id -> clinical_id through multimodal_dataset_v2.csv.

    Clinical features:
        synthetic contextual scenarios only.
        They are NOT naturally observed patient-meal pairs.

    Continuous preprocessing:
        TRAIN-only statistics from the frozen preprocessing JSON.
    """

    def __init__(
        self,
        split_csv,
        multimodal_csv,
        clinical_csv,
        image_processor,
        targets,
        target_scaler,
        clinical_preprocessing_json,
    ):
        super().__init__()

        self.split_csv = Path(split_csv)
        self.multimodal_csv = Path(multimodal_csv)
        self.clinical_csv = Path(clinical_csv)

        self.image_processor = image_processor
        self.targets = list(targets)
        self.target_scaler = target_scaler

        # --------------------------------------------------
        # Load frozen preprocessing specification
        # --------------------------------------------------
        prep_path = Path(clinical_preprocessing_json)

        with open(
            prep_path,
            "r",
            encoding="utf-8",
        ) as f:
            self.prep = json.load(f)

        if (
            self.prep["feature_set"]
            != "E3_CLINICAL_FEATURES_V1"
        ):
            raise RuntimeError(
                "Unexpected clinical feature set."
            )

        if self.prep["fit_scope"] != "TRAIN_ONLY":
            raise RuntimeError(
                "Clinical preprocessing must be TRAIN_ONLY."
            )

        if self.prep["encoded_dimension"] != 16:
            raise RuntimeError(
                "Expected clinical encoded dimension = 16."
            )

        self.continuous = list(
            self.prep["continuous_features"]
        )

        self.binary = list(
            self.prep["binary_features"]
        )

        self.means = self.prep["continuous_mean"]
        self.stds = self.prep["continuous_std"]

        # --------------------------------------------------
        # Load sources
        # --------------------------------------------------
        split = pd.read_csv(self.split_csv)
        mm = pd.read_csv(self.multimodal_csv)
        clinical = pd.read_csv(self.clinical_csv)

        # --------------------------------------------------
        # Required columns
        # --------------------------------------------------
        split_required = {
            "record_id",
            "image_path",
            *self.targets,
        }

        missing = split_required - set(split.columns)

        if missing:
            raise RuntimeError(
                f"Missing split columns: {sorted(missing)}"
            )

        mm_required = {
            "record_id",
            "clinical_id",
        }

        missing = mm_required - set(mm.columns)

        if missing:
            raise RuntimeError(
                f"Missing multimodal columns: {sorted(missing)}"
            )

        clinical_required = {
            "clinical_id",
            *self.continuous,
            *self.binary,
            "sex",
            "activity_level",
        }

        missing = clinical_required - set(clinical.columns)

        if missing:
            raise RuntimeError(
                f"Missing clinical columns: {sorted(missing)}"
            )

        # --------------------------------------------------
        # Duplicate guards
        # --------------------------------------------------
        if split["record_id"].duplicated().any():
            raise RuntimeError(
                "Duplicate record_id in split CSV."
            )

        mapping = mm[
            ["record_id", "clinical_id"]
        ].copy()

        if mapping["record_id"].duplicated().any():
            raise RuntimeError(
                "Duplicate record_id in multimodal mapping."
            )

        if clinical["clinical_id"].duplicated().any():
            raise RuntimeError(
                "Duplicate clinical_id in clinical profiles."
            )

        # --------------------------------------------------
        # Explicit one-to-one clinical linkage
        #
        # Frozen split CSVs contain a placeholder clinical_id
        # column whose values are null. They define split
        # membership, image paths, and nutrition targets, but
        # they are NOT the authoritative source of clinical
        # linkage.
        #
        # Authoritative linkage:
        # record_id -> clinical_id
        # from multimodal_dataset_v2.csv.
        #
        # Never rely on row order.
        # --------------------------------------------------

        if "clinical_id" in split.columns:
            split_clinical_non_null = int(
                split["clinical_id"].notna().sum()
            )

            if split_clinical_non_null != 0:
                raise RuntimeError(
                    "Frozen split clinical_id is expected "
                    "to be a null placeholder, but "
                    f"{split_clinical_non_null} non-null "
                    "values were found."
                )

            split = split.drop(
                columns=["clinical_id"]
            )

        df = split.merge(
            mapping,
            on="record_id",
            how="left",
            validate="one_to_one",
        )

        if len(df) != len(split):
            raise RuntimeError(
                "Row count changed after multimodal "
                "clinical mapping."
            )

        if df["clinical_id"].isna().any():
            raise RuntimeError(
                "Missing clinical_id after authoritative "
                "record_id -> clinical_id mapping."
            )

        df = df.merge(
            clinical,
            on="clinical_id",
            how="left",
            validate="one_to_one",
            indicator="_clinical_merge",
        )

        if len(df) != len(split):
            raise RuntimeError(
                "Row count changed after clinical profile join."
            )

        missing_profile = (
            df["_clinical_merge"] != "both"
        )

        if missing_profile.any():
            raise RuntimeError(
                "Clinical profile missing for "
                f"{int(missing_profile.sum())} records."
            )

        df = df.drop(
            columns=["_clinical_merge"]
        )

        # --------------------------------------------------
        # Target guards
        # --------------------------------------------------
        if df[self.targets].isna().any().any():
            raise RuntimeError(
                "Missing nutrition targets detected."
            )

        # --------------------------------------------------
        # Clinical feature guards
        # --------------------------------------------------
        feature_cols = (
            self.continuous
            + self.binary
            + ["sex", "activity_level"]
        )

        if df[feature_cols].isna().any().any():
            raise RuntimeError(
                "Missing E3 clinical feature detected."
            )

        valid_sex = {"female", "male"}

        if not set(df["sex"].unique()).issubset(
            valid_sex
        ):
            raise RuntimeError(
                "Unexpected sex category."
            )

        valid_activity = {
            "low",
            "moderate",
            "high",
        }

        if not set(
            df["activity_level"].unique()
        ).issubset(valid_activity):
            raise RuntimeError(
                "Unexpected activity_level category."
            )

        # --------------------------------------------------
        # Explicitly guard excluded features from model input
        # --------------------------------------------------
        expected_excluded = {
            "clinical_id",
            "sodium_limit_mg",
            "energy_limit_kcal",
            "profile_type",
        }

        configured_excluded = set(
            self.prep["excluded_features"]
        )

        if configured_excluded != expected_excluded:
            raise RuntimeError(
                "Unexpected excluded-feature specification."
            )

        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def _encode_clinical(self, row):
        values = []

        # ----------------------------------------------
        # 9 continuous -> TRAIN-only z-score
        # ----------------------------------------------
        for col in self.continuous:
            x = float(row[col])
            mean = float(self.means[col])
            std = float(self.stds[col])

            if std <= 0:
                raise RuntimeError(
                    f"Invalid std for {col}: {std}"
                )

            values.append(
                (x - mean) / std
            )

        # ----------------------------------------------
        # 3 binary -> raw 0/1
        # ----------------------------------------------
        for col in self.binary:
            x = float(row[col])

            if x not in (0.0, 1.0):
                raise RuntimeError(
                    f"Non-binary value for {col}: {x}"
                )

            values.append(x)

        # ----------------------------------------------
        # sex -> 1 dimension
        # female=0, male=1
        # ----------------------------------------------
        sex_map = self.prep[
            "categorical_encoding"
        ]["sex"]

        sex = str(row["sex"])

        if sex not in sex_map:
            raise RuntimeError(
                f"Unknown sex category: {sex}"
            )

        values.append(
            float(sex_map[sex])
        )

        # ----------------------------------------------
        # activity -> full 3-way one-hot
        # ----------------------------------------------
        activity_map = self.prep[
            "categorical_encoding"
        ]["activity_level"]

        activity = str(row["activity_level"])

        if activity not in activity_map:
            raise RuntimeError(
                f"Unknown activity category: {activity}"
            )

        values.extend(
            [
                float(x)
                for x in activity_map[activity]
            ]
        )

        vector = np.asarray(
            values,
            dtype=np.float32,
        )

        if vector.shape != (16,):
            raise RuntimeError(
                f"Expected clinical vector (16,), "
                f"got {vector.shape}"
            )

        if not np.isfinite(vector).all():
            raise RuntimeError(
                "Non-finite clinical feature detected."
            )

        return vector

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = Path(
            str(row["image_path"])
        )

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("RGB")

        processed = self.image_processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = processed[
            "pixel_values"
        ].squeeze(0)

        target = np.asarray(
            [
                float(row[col])
                for col in self.targets
            ],
            dtype=np.float32,
        )

        target_tensor = torch.tensor(
            target,
            dtype=torch.float32,
        )

        target_normalized = (
            self.target_scaler.normalize(
                target_tensor
            )
        )

        clinical_vector = (
            self._encode_clinical(row)
        )

        return {
            "record_id": str(row["record_id"]),
            "clinical_id": str(row["clinical_id"]),
            "pixel_values": pixel_values,
            "clinical_features": torch.tensor(
                clinical_vector,
                dtype=torch.float32,
            ),
            "target": torch.tensor(
                target,
                dtype=torch.float32,
            ),
            "target_normalized": target_normalized,
        }
