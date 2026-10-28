import torch
from torch.utils.data import Dataset

from src.dataset.conference2_image_text_dataset import (
    ImageTextNutritionDataset,
)
from src.dataset.conference2_image_clinical_dataset import (
    ImageClinicalNutritionDataset,
)


class ImageTextClinicalNutritionDataset(Dataset):
    """
    Conference 2 E4 dataset.

    Combines image, generated text, and synthetic clinical
    context for five-target food-nutrition regression.

    Reuses the audited E2 and E3 preprocessing pipelines.
    """

    def __init__(
        self,
        split_csv,
        multimodal_csv,
        clinical_csv,
        image_processor,
        tokenizer,
        target_names,
        target_scaler,
        text_column,
        max_length,
        clinical_preprocessing_json,
    ):
        self.image_text = ImageTextNutritionDataset(
            split_csv=split_csv,
            multimodal_csv=multimodal_csv,
            image_processor=image_processor,
            tokenizer=tokenizer,
            target_names=target_names,
            target_scaler=target_scaler,
            text_column=text_column,
            max_length=max_length,
        )

        self.image_clinical = ImageClinicalNutritionDataset(
            split_csv=split_csv,
            multimodal_csv=multimodal_csv,
            clinical_csv=clinical_csv,
            image_processor=image_processor,
            targets=target_names,
            target_scaler=target_scaler,
            clinical_preprocessing_json=clinical_preprocessing_json,
        )

        assert len(self.image_text) == len(self.image_clinical), (
            "E2/E3 dataset lengths differ."
        )

        self._verify_alignment()

    def _verify_alignment(self):
        """
        Verify one-to-one record alignment before training.

        Dataset order must match because both datasets
        originate from the same authoritative split CSV.
        """
        text_df = self.image_text.df
        clinical_df = self.image_clinical.df

        assert "record_id" in text_df.columns
        assert "record_id" in clinical_df.columns

        text_ids = text_df["record_id"].astype(str).tolist()
        clinical_ids = clinical_df["record_id"].astype(str).tolist()

        assert len(text_ids) == len(set(text_ids)), (
            "Duplicate record_id in image-text dataset."
        )

        assert len(clinical_ids) == len(set(clinical_ids)), (
            "Duplicate record_id in image-clinical dataset."
        )

        assert text_ids == clinical_ids, (
            "E2/E3 record ordering or linkage mismatch."
        )

        self.record_ids = text_ids

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, idx):
        text_item = self.image_text[idx]
        clinical_row = self.image_clinical.df.iloc[idx]

        assert text_item["record_id"] == str(
            clinical_row["record_id"]
        ), "Record alignment mismatch."

        clinical_vector = self.image_clinical._encode_clinical(
            clinical_row
        )

        clinical_features = torch.as_tensor(
            clinical_vector,
            dtype=torch.float32,
        )

        assert clinical_features.shape == (16,), (
            "Expected 16 clinical features."
        )

        assert torch.isfinite(clinical_features).all(), (
            "Non-finite clinical features."
        )

        return {
            "record_id": text_item["record_id"],
            "clinical_id": str(clinical_row["clinical_id"]),
            "text": text_item["text"],
            "pixel_values": text_item["pixel_values"],
            "input_ids": text_item["input_ids"],
            "attention_mask": text_item["attention_mask"],
            "clinical_features": clinical_features,
            "target": text_item["target"],
            "target_normalized": text_item["target_normalized"],
        }
