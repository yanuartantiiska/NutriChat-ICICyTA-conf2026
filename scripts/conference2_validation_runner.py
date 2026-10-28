"""
Conference 2 unified validation runner.

Supported split: VALIDATION ONLY.
TEST evaluation is intentionally not implemented.
"""

from pathlib import Path
import gc
import json

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from src.evaluation.target_scaler import TargetScaler
from src.vision.vit_regressor import NutriViTRegressor
from src.vision.vit_dataset import ViTNutritionDataset

from src.multimodal.conference2_image_text_regressor import (
    ImageTextNutritionRegressor,
)
from src.multimodal.conference2_image_clinical_regressor import (
    ImageClinicalNutritionRegressor,
)
from src.multimodal.conference2_image_text_clinical_regressor import (
    ImageTextClinicalNutritionRegressor,
)
from src.multimodal.conference2_attention_fusion_regressor import (
    ImageTextClinicalAttentionRegressor,
)

from src.dataset.conference2_image_text_dataset import (
    ImageTextNutritionDataset,
)
from src.dataset.conference2_image_clinical_dataset import (
    ImageClinicalNutritionDataset,
)
from src.dataset.conference2_image_text_clinical_dataset import (
    ImageTextClinicalNutritionDataset,
)


TARGETS = [
    "mass_g",
    "energy_kcal",
    "protein_g",
    "fat_g",
    "carbohydrate_g",
]

CONFIGS = {
    "E1": "conference2_e1_image.yaml",
    "E2": "conference2_e2_image_text.yaml",
    "E3": "conference2_e3_image_clinical.yaml",
    "E4": "conference2_e4_image_text_clinical.yaml",
    "E5": "conference2_e5_attention_fusion.yaml",
}


def checkpoint_path(root, cfg):
    cp = cfg["checkpoint"]
    if "path" in cp:
        return root / cp["path"]
    return root / cp["directory"] / cp["filename"]


def make_model(exp, cfg):
    if exp == "E1":
        return NutriViTRegressor(
            model_name=cfg["model"]["name"],
            num_targets=5,
            dropout=cfg["model"]["dropout"],
        )

    if exp == "E2":
        return ImageTextNutritionRegressor(
            image_model_name=cfg["image_encoder"]["name"],
            text_model_name=cfg["text_encoder"]["name"],
            projection_dim=cfg["fusion"]["image_projection"],
            regression_hidden=cfg["fusion"]["regression_hidden"],
            num_targets=5,
            dropout=cfg["fusion"]["dropout"],
            freeze_text=not cfg["text_encoder"]["trainable"],
        )

    if exp == "E3":
        return ImageClinicalNutritionRegressor(
            image_model_name=cfg["image_encoder"]["name"],
            clinical_input_dim=cfg["clinical_encoder"]["input_dim"],
            projection_dim=cfg["clinical_encoder"]["output_dim"],
            regression_hidden_dim=cfg["fusion"]["regression_hidden"],
            num_targets=5,
            dropout=cfg["fusion"]["dropout"],
        )

    common = dict(
        image_model_name=cfg["image_encoder"]["name"],
        text_model_name=cfg["text_encoder"]["name"],
        image_hidden_dim=cfg["image_encoder"]["hidden_size"],
        text_hidden_dim=cfg["text_encoder"]["hidden_size"],
        clinical_input_dim=cfg["clinical_encoder"]["input_dim"],
        projection_dim=cfg["fusion"]["image_projection"],
        clinical_hidden_dim=cfg["clinical_encoder"]["hidden_dim"],
        regression_hidden_dim=cfg["fusion"]["regression_hidden"],
        dropout=cfg["fusion"]["dropout"],
        num_targets=5,
        freeze_text=not cfg["text_encoder"]["trainable"],
    )

    if exp == "E4":
        return ImageTextClinicalNutritionRegressor(**common)

    if exp == "E5":
        return ImageTextClinicalAttentionRegressor(
            **common,
            num_heads=cfg["fusion"]["num_heads"],
            num_layers=cfg["fusion"]["num_layers"],
            attention_dropout=cfg["fusion"]["attention_dropout"],
        )

    raise ValueError(f"Unknown experiment: {exp}")


def make_dataset(exp, cfg, root, scaler):
    image_name = (
        cfg["model"]["name"]
        if exp == "E1"
        else cfg["image_encoder"]["name"]
    )

    processor = AutoImageProcessor.from_pretrained(
        image_name,
        use_fast=False,
    )

    split_csv = str(root / cfg["data"]["val_csv"])

    if exp == "E1":
        return ViTNutritionDataset(
            csv_path=split_csv,
            processor=processor,
            target_names=TARGETS,
        )

    multimodal_csv = str(
        root / cfg["data"]["multimodal_csv"]
    )

    if exp in ("E2", "E4", "E5"):
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["text_encoder"]["name"]
        )

    if exp == "E2":
        return ImageTextNutritionDataset(
            split_csv=split_csv,
            multimodal_csv=multimodal_csv,
            image_processor=processor,
            tokenizer=tokenizer,
            target_names=TARGETS,
            target_scaler=scaler,
            text_column=cfg["data"]["text_column"],
            max_length=cfg["text_encoder"]["max_length"],
        )

    clinical_csv = str(
        root / cfg["data"]["clinical_csv"]
    )
    clinical_preprocessing = str(
        root / cfg["clinical_encoder"]["preprocessing"]
    )

    if exp == "E3":
        return ImageClinicalNutritionDataset(
            split_csv=split_csv,
            multimodal_csv=multimodal_csv,
            clinical_csv=clinical_csv,
            image_processor=processor,
            targets=TARGETS,
            target_scaler=scaler,
            clinical_preprocessing_json=clinical_preprocessing,
        )

    return ImageTextClinicalNutritionDataset(
        split_csv=split_csv,
        multimodal_csv=multimodal_csv,
        clinical_csv=clinical_csv,
        image_processor=processor,
        tokenizer=tokenizer,
        target_names=TARGETS,
        target_scaler=scaler,
        text_column=cfg["data"]["text_column"],
        max_length=cfg["text_encoder"]["max_length"],
        clinical_preprocessing_json=clinical_preprocessing,
    )


def compute_metrics(y_true, y_pred, scaler):
    std = scaler.std.numpy().reshape(1, -1)

    normalized_mae = float(
        np.mean(np.abs(y_pred - y_true) / std)
    )

    per_target = {}

    for i, name in enumerate(TARGETS):
        actual = y_true[:, i].astype(np.float64)
        predicted = y_pred[:, i].astype(np.float64)

        mae = float(np.mean(np.abs(predicted - actual)))
        rmse = float(
            np.sqrt(np.mean((predicted - actual) ** 2))
        )

        ss_res = float(np.sum((actual - predicted) ** 2))
        ss_tot = float(np.sum((actual - actual.mean()) ** 2))

        r2 = (
            1.0 - ss_res / ss_tot
            if ss_tot > 0
            else float("nan")
        )

        per_target[name] = {
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }

    return normalized_mae, per_target


def run_validation(root=None, batch_size=8):
    root = Path.cwd() if root is None else Path(root)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    scaler = TargetScaler(
        root / "config/conference2_target_stats.json"
    )

    results = {}

    print("=" * 76)
    print("CONFERENCE 2 - UNIFIED VALIDATION E1-E5")
    print("=" * 76)
    print("Device:", device)
    print("Split: VALIDATION ONLY")

    for exp, filename in CONFIGS.items():
        print(f"\n[{exp}]")

        with (root / "config" / filename).open(
            encoding="utf-8"
        ) as f:
            cfg = yaml.safe_load(f)

        dataset = make_dataset(exp, cfg, root, scaler)

        if len(dataset) != 523:
            raise RuntimeError(
                f"{exp}: unexpected validation size "
                f"{len(dataset)}"
            )

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
        )

        model = make_model(exp, cfg)

        checkpoint = torch.load(
            checkpoint_path(root, cfg),
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        reference = float(
            checkpoint["best_val_mae_normalized"]
        )

        model.to(device)
        model.eval()

        predictions = []
        ground_truth = []

        with torch.inference_mode():
            for batch in loader:
                inputs = {
                    "pixel_values":
                        batch["pixel_values"].to(device)
                }

                if exp in ("E2", "E4", "E5"):
                    inputs["input_ids"] = (
                        batch["input_ids"].to(device)
                    )
                    inputs["attention_mask"] = (
                        batch["attention_mask"].to(device)
                    )

                if exp in ("E3", "E4", "E5"):
                    inputs["clinical_features"] = (
                        batch["clinical_features"].to(device)
                    )

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=(device.type == "cuda"),
                ):
                    pred_normalized = model(**inputs)

                pred_original = scaler.denormalize(
                    pred_normalized.float().cpu()
                )

                predictions.append(pred_original.numpy())
                ground_truth.append(
                    batch["target"].numpy()
                )

        y_pred = np.concatenate(predictions, axis=0)
        y_true = np.concatenate(ground_truth, axis=0)

        if y_pred.shape != (523, 5):
            raise RuntimeError(
                f"{exp}: invalid prediction shape"
            )

        if not (
            np.isfinite(y_pred).all()
            and np.isfinite(y_true).all()
        ):
            raise RuntimeError(
                f"{exp}: non-finite predictions or targets"
            )

        normalized_mae, per_target = compute_metrics(
            y_true, y_pred, scaler
        )

        difference = abs(normalized_mae - reference)

        if difference >= 1e-5:
            raise RuntimeError(
                f"{exp}: checkpoint reproduction mismatch "
                f"{difference}"
            )

        results[exp] = {
            "samples": 523,
            "normalized_mae": normalized_mae,
            "checkpoint_normalized_mae": reference,
            "absolute_difference": difference,
            "per_target": per_target,
        }

        print("Normalized MAE:", normalized_mae)
        print("Checkpoint MAE:", reference)
        print("Difference:", difference)
        print("Reproduction: PASS")

        del model, checkpoint, dataset, loader
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n[SAFETY]")
    print("TEST dataset accessed : NO")
    print("Training performed    : NO")
    print("Checkpoints modified  : NO")
    print("=" * 76)
    print("UNIFIED VALIDATION: PASS")

    return results
