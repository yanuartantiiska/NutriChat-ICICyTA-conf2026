from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class ViTNutritionDataset(Dataset):

    def __init__(
        self,
        csv_path,
        processor,
        target_names,
    ):
        self.csv_path = Path(csv_path)
        self.processor = processor
        self.target_names = target_names

        if not self.csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {self.csv_path}"
            )

        self.df = pd.read_csv(
            self.csv_path
        ).reset_index(drop=True)

        required = [
            "record_id",
            "image_path",
            *self.target_names,
        ]

        missing = [
            col
            for col in required
            if col not in self.df.columns
        ]

        if missing:
            raise ValueError(
                f"Missing columns: {missing}"
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(
        self,
        idx,
    ):
        row = self.df.iloc[idx]

        image_path = Path(
            row["image_path"]
        )

        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        processed = self.processor(
            images=image,
            return_tensors="pt",
        )

        pixel_values = processed[
            "pixel_values"
        ].squeeze(0)

        target = torch.tensor(
            [
                float(row[name])
                for name in self.target_names
            ],
            dtype=torch.float32,
        )

        return {
            "pixel_values": pixel_values,
            "target": target,
            "record_id": row["record_id"],
        }