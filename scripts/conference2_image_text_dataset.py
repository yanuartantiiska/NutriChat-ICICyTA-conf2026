from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageTextNutritionDataset(Dataset):
    """
    Conference-2 Image+Text dataset.

    Alignment policy:
    - split membership and nutrition targets come from the locked split CSV;
    - user_utterance comes from multimodal_dataset_v2.csv;
    - modalities are joined strictly by record_id;
    - row-order alignment is never used.
    """

    def __init__(
        self,
        split_csv,
        multimodal_csv,
        image_processor,
        tokenizer,
        target_names,
        target_scaler,
        text_column="user_utterance",
        max_length=48,
    ):
        self.split_csv = Path(split_csv)
        self.multimodal_csv = Path(multimodal_csv)

        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.target_names = list(target_names)
        self.target_scaler = target_scaler
        self.text_column = text_column
        self.max_length = int(max_length)

        # ---------------------------------------------------------
        # Load locked split and multimodal context
        # ---------------------------------------------------------
        split_df = pd.read_csv(self.split_csv)
        multimodal_df = pd.read_csv(self.multimodal_csv)

        required_split = {
            "record_id",
            "image_path",
            *self.target_names,
        }

        required_multimodal = {
            "record_id",
            self.text_column,
        }

        missing_split = required_split - set(split_df.columns)
        missing_multimodal = (
            required_multimodal - set(multimodal_df.columns)
        )

        if missing_split:
            raise ValueError(
                "Missing split columns: "
                f"{sorted(missing_split)}"
            )

        if missing_multimodal:
            raise ValueError(
                "Missing multimodal columns: "
                f"{sorted(missing_multimodal)}"
            )

        # ---------------------------------------------------------
        # Strong uniqueness guards
        # ---------------------------------------------------------
        if split_df["record_id"].duplicated().any():
            raise ValueError(
                "Duplicate record_id detected in split CSV."
            )

        if multimodal_df["record_id"].duplicated().any():
            raise ValueError(
                "Duplicate record_id detected in multimodal CSV."
            )

        # Keep only the fields required from multimodal dataset.
        text_df = multimodal_df[
            ["record_id", self.text_column]
        ].copy()

        # ---------------------------------------------------------
        # One-to-one join by record_id
        # ---------------------------------------------------------
        merged = split_df.merge(
            text_df,
            on="record_id",
            how="left",
            validate="one_to_one",
            indicator=True,
        )

        unmatched = (
            merged["_merge"] != "both"
        ).sum()

        if unmatched != 0:
            raise ValueError(
                f"{unmatched} split records have no matching "
                "multimodal text."
            )

        merged = merged.drop(
            columns=["_merge"]
        )

        if len(merged) != len(split_df):
            raise ValueError(
                "Row count changed after image-text join."
            )

        if merged[self.text_column].isna().any():
            raise ValueError(
                "Missing user_utterance after join."
            )

        if (
            merged[self.text_column]
            .astype(str)
            .str.strip()
            .eq("")
            .any()
        ):
            raise ValueError(
                "Empty user_utterance detected."
            )

        # ---------------------------------------------------------
        # Target guards
        # ---------------------------------------------------------
        for target in self.target_names:
            if merged[target].isna().any():
                raise ValueError(
                    f"Missing target values: {target}"
                )

        self.df = merged.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        record_id = str(row["record_id"])
        image_path = Path(str(row["image_path"]))
        text = str(row[self.text_column])

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # ---------------------------------------------------------
        # Image
        # ---------------------------------------------------------
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            image_encoded = self.image_processor(
                images=image,
                return_tensors="pt",
            )

        pixel_values = (
            image_encoded["pixel_values"]
            .squeeze(0)
        )

        # ---------------------------------------------------------
        # Text
        # ---------------------------------------------------------
        text_encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        input_ids = (
            text_encoded["input_ids"]
            .squeeze(0)
        )

        attention_mask = (
            text_encoded["attention_mask"]
            .squeeze(0)
        )

        # ---------------------------------------------------------
        # Targets
        # ---------------------------------------------------------
        target = torch.tensor(
            [
                float(row[name])
                for name in self.target_names
            ],
            dtype=torch.float32,
        )

        target_normalized = (
            self.target_scaler.normalize(
                target.unsqueeze(0)
            )
            .squeeze(0)
        )

        return {
            "record_id": record_id,
            "text": text,
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target": target,
            "target_normalized": target_normalized,
        }