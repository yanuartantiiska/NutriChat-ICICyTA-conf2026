import copy
import csv
import json
import math
import random
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoImageProcessor


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.target_scaler import TargetScaler
from src.vision.vit_dataset import ViTNutritionDataset
from src.vision.vit_regressor import NutriViTRegressor


CONFIG_PATH = (
    PROJECT_ROOT
    / "config"
    / "conference2_e1_image.yaml"
)

EXPECTED_TARGETS = [
    "mass_g",
    "energy_kcal",
    "protein_g",
    "fat_g",
    "carbohydrate_g",
]

EXPECTED_SPLIT_COUNTS = {
    "train": 2443,
    "val": 523,
    "test": 524,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# METRICS
# ============================================================

def compute_metrics(y_true, y_pred, target_names):

    results = {}

    for i, name in enumerate(target_names):

        true_i = y_true[:, i]
        pred_i = y_pred[:, i]

        mae = mean_absolute_error(
            true_i,
            pred_i,
        )

        rmse = math.sqrt(
            mean_squared_error(
                true_i,
                pred_i,
            )
        )

        r2 = r2_score(
            true_i,
            pred_i,
        )

        results[name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
        }

    return results


# ============================================================
# TRAIN
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    target_scaler,
    device,
    grad_scaler,
    amp_enabled,
):

    model.train()

    running_loss = 0.0
    total_samples = 0

    progress = tqdm(
        loader,
        desc="TRAIN",
        leave=False,
    )

    for batch in progress:

        pixel_values = batch[
            "pixel_values"
        ].to(
            device,
            non_blocking=True,
        )

        targets = batch[
            "target"
        ].to(
            device,
            non_blocking=True,
        )

        targets_norm = target_scaler.normalize(
            targets
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):

            predictions_norm = model(
                pixel_values
            )

            loss = criterion(
                predictions_norm,
                targets_norm,
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                "Non-finite training loss."
            )

        grad_scaler.scale(
            loss
        ).backward()

        grad_scaler.step(
            optimizer
        )

        grad_scaler.update()

        batch_size = pixel_values.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        progress.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    return running_loss / total_samples


# ============================================================
# VALIDATION
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    criterion,
    target_scaler,
    target_names,
    device,
    amp_enabled,
):

    model.eval()

    running_loss = 0.0
    total_samples = 0

    true_original = []
    pred_original = []

    for batch in tqdm(
        loader,
        desc="VAL",
        leave=False,
    ):

        pixel_values = batch[
            "pixel_values"
        ].to(
            device,
            non_blocking=True,
        )

        targets = batch[
            "target"
        ].to(
            device,
            non_blocking=True,
        )

        targets_norm = target_scaler.normalize(
            targets
        )

        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):

            predictions_norm = model(
                pixel_values
            )

            loss = criterion(
                predictions_norm,
                targets_norm,
            )

        predictions = (
            target_scaler
            .denormalize(
                predictions_norm.float()
            )
        )

        batch_size = pixel_values.size(0)

        running_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        true_original.append(
            targets.detach().cpu()
        )

        pred_original.append(
            predictions.detach().cpu()
        )

    y_true = torch.cat(
        true_original,
        dim=0,
    ).numpy()

    y_pred = torch.cat(
        pred_original,
        dim=0,
    ).numpy()

    metrics = compute_metrics(
        y_true,
        y_pred,
        target_names,
    )

    std = (
        target_scaler.std
        .numpy()
        .reshape(1, -1)
    )

    normalized_mae = float(
        np.mean(
            np.abs(
                (y_pred - y_true)
                / std
            )
        )
    )

    return (
        running_loss / total_samples,
        normalized_mae,
        metrics,
    )


# ============================================================
# MAIN
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Temporary epoch override for smoke testing. "
            "If omitted, use frozen YAML configuration."
        ),
    )

    return parser.parse_args()

def main():

    print("=" * 78)
    args = parse_args()
    print("CONFERENCE 2 — E1 IMAGE-ONLY BASELINE")
    print("=" * 78)

    # --------------------------------------------------------
    # CONFIG
    # --------------------------------------------------------

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as f:
        config = yaml.safe_load(f)

    assert (
        config["experiment"]["protocol"]
        == "conference2_frozen_v1"
    )

    assert (
        config["targets"]["names"]
        == EXPECTED_TARGETS
    )

    assert (
        config["model"]["num_targets"]
        == len(EXPECTED_TARGETS)
    )

    assert (
        config["training"]["loss"]
        == "SmoothL1"
    )

    assert (
        config["evaluation"]
        ["model_selection_uses_test"]
        is False
    )

    seed = config["experiment"]["seed"]

    set_seed(seed)

    # --------------------------------------------------------
    # DEVICE
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    amp_enabled = (
        device.type == "cuda"
        and config["training"]["mixed_precision"]
    )

    print("Protocol    :", config["experiment"]["protocol"])
    print("Seed        :", seed)
    print("Device      :", device)
    print("AMP         :", amp_enabled)
    print("Targets     :", EXPECTED_TARGETS)

    # --------------------------------------------------------
    # SPLIT GUARDS
    # --------------------------------------------------------

    train_path = (
        PROJECT_ROOT
        / config["data"]["train_csv"]
    )

    val_path = (
        PROJECT_ROOT
        / config["data"]["val_csv"]
    )

    test_path = (
        PROJECT_ROOT
        / config["data"]["test_csv"]
    )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # Intentionally read only test header/count here.
    # No test targets are used for training/model selection.
    test_df_guard = pd.read_csv(
        test_path,
        usecols=["record_id"],
    )

    assert len(train_df) == EXPECTED_SPLIT_COUNTS["train"]
    assert len(val_df) == EXPECTED_SPLIT_COUNTS["val"]
    assert len(test_df_guard) == EXPECTED_SPLIT_COUNTS["test"]

    assert train_df["record_id"].is_unique
    assert val_df["record_id"].is_unique
    assert test_df_guard["record_id"].is_unique

    train_ids = set(train_df["record_id"])
    val_ids = set(val_df["record_id"])
    test_ids = set(test_df_guard["record_id"])

    assert not (train_ids & val_ids)
    assert not (train_ids & test_ids)
    assert not (val_ids & test_ids)

    print(
        "Split guard : PASS "
        "(2443 / 523 / 524)"
    )

    # --------------------------------------------------------
    # TARGET SCALER
    # --------------------------------------------------------

    scaler_path = (
        PROJECT_ROOT
        / config["targets"]["scaler"]
    )

    target_scaler = TargetScaler(
        scaler_path
    )

    assert (
        target_scaler.targets
        == EXPECTED_TARGETS
    )

    print("Scaler      : PASS")

    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_dir = (
        PROJECT_ROOT
        / config["output"]["directory"]
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_dir = (
        PROJECT_ROOT
        / config["checkpoint"]["directory"]
    )

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = (
        checkpoint_dir
        / config["checkpoint"]["filename"]
    )

    history_path = (
        output_dir
        / config["output"]["history"]
    )

    frozen_config_path = (
        output_dir
        / "config_frozen.json"
    )

    scaler_copy_path = (
        output_dir
        / "target_scaler_frozen.json"
    )

    # --------------------------------------------------------
    # SAVE FROZEN CONFIG + SCALER
    # --------------------------------------------------------

    with open(
        frozen_config_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            config,
            f,
            indent=2,
        )

    with open(
        scaler_path,
        "r",
        encoding="utf-8",
    ) as f:
        scaler_payload = json.load(f)

    with open(
        scaler_copy_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            scaler_payload,
            f,
            indent=2,
        )

    # --------------------------------------------------------
    # PROCESSOR
    # --------------------------------------------------------

    processor = AutoImageProcessor.from_pretrained(
        config["model"]["name"]
    )

    # --------------------------------------------------------
    # DATASETS
    # --------------------------------------------------------

    train_dataset = ViTNutritionDataset(
        train_path,
        processor,
        EXPECTED_TARGETS,
    )

    val_dataset = ViTNutritionDataset(
        val_path,
        processor,
        EXPECTED_TARGETS,
    )

    # IMPORTANT:
    # no test Dataset / DataLoader is created here.

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    print("Train       :", len(train_dataset))
    print("Validation  :", len(val_dataset))
    print("Test loader : NOT CREATED")

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = NutriViTRegressor(
        model_name=config["model"]["name"],
        num_targets=5,
        dropout=config["model"]["dropout"],
    ).to(device)

    # --------------------------------------------------------
    # LOSS
    # --------------------------------------------------------

    criterion = nn.SmoothL1Loss()

    # --------------------------------------------------------
    # OPTIMIZER
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    grad_scaler = torch.amp.GradScaler(
        "cuda",
        enabled=amp_enabled,
    )

    # --------------------------------------------------------
    # HISTORY
    # --------------------------------------------------------

    history_columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "val_mae_normalized",
    ]

    for target in EXPECTED_TARGETS:
        history_columns.extend([
            f"{target}_mae",
            f"{target}_rmse",
            f"{target}_r2",
        ])

    with open(
        history_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)
        writer.writerow(history_columns)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    epochs = (
        args.epochs
        if args.epochs is not None
        else config["training"]["epochs"]
    )

    run_mode = (
        "SMOKE_TEST"
        if args.epochs is not None
        else "FINAL_FROZEN_RUN"
    )

    print("Run mode    :", run_mode)
    print("Run epochs  :", epochs)

    patience = (
        config["training"]
        ["early_stopping"]
        ["patience"]
    )

    best_val_mae = float("inf")
    best_epoch = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):

        print()
        print(
            f"Epoch {epoch}/{epochs}"
        )

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            target_scaler,
            device,
            grad_scaler,
            amp_enabled,
        )

        (
            val_loss,
            val_mae_norm,
            metrics,
        ) = validate(
            model,
            val_loader,
            criterion,
            target_scaler,
            EXPECTED_TARGETS,
            device,
            amp_enabled,
        )

        print(
            f"Train loss   : {train_loss:.6f}"
        )

        print(
            f"Val loss     : {val_loss:.6f}"
        )

        print(
            f"Val MAE norm : {val_mae_norm:.6f}"
        )

        row = [
            epoch,
            train_loss,
            val_loss,
            val_mae_norm,
        ]

        for target in EXPECTED_TARGETS:

            m = metrics[target]

            print(
                f"{target:18s} "
                f"MAE={m['mae']:.4f} "
                f"RMSE={m['rmse']:.4f} "
                f"R2={m['r2']:.4f}"
            )

            row.extend([
                m["mae"],
                m["rmse"],
                m["r2"],
            ])

        with open(
            history_path,
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            csv.writer(f).writerow(row)

        # ----------------------------------------------------
        # CHECKPOINT SELECTION — VALIDATION ONLY
        # ----------------------------------------------------

        if val_mae_norm < best_val_mae:

            best_val_mae = val_mae_norm
            best_epoch = epoch
            patience_counter = 0

            checkpoint = {
                "experiment":
                    config["experiment"]["name"],

                "protocol":
                    config["experiment"]["protocol"],

                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "best_val_mae_normalized":
                    best_val_mae,

                "target_names":
                    EXPECTED_TARGETS,

                "target_scaler":
                    scaler_payload,

                "config":
                    copy.deepcopy(config),

                "test_used_for_model_selection":
                    False,
            }

            torch.save(
                checkpoint,
                checkpoint_path,
            )

            print(
                "BEST CHECKPOINT SAVED:",
                checkpoint_path,
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/{patience})"
            )

        if (
            config["training"]
            ["early_stopping"]
            ["enabled"]
            and patience_counter >= patience
        ):

            print(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # RUN SUMMARY
    # --------------------------------------------------------

    summary = {
        "experiment":
            config["experiment"]["name"],

        "protocol":
            config["experiment"]["protocol"],

        "seed":
            seed,

        "device":
            str(device),

        "targets":
            EXPECTED_TARGETS,

        "loss":
            "SmoothL1Loss",

        "best_epoch":
            best_epoch,

        "best_val_mae_normalized":
            best_val_mae,

        "train_samples":
            len(train_dataset),

        "val_samples":
            len(val_dataset),

        "test_samples":
            EXPECTED_SPLIT_COUNTS["test"],

        "test_loader_created":
            False,

        "test_used_for_model_selection":
            False,

        "checkpoint":
            str(checkpoint_path),
    }

    summary_path = (
        output_dir
        / config["output"]["summary"]
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print()
    print("=" * 78)
    print("E1 TRAINING COMPLETE")
    print("=" * 78)

    print("Best epoch        :", best_epoch)
    print(
        "Best val MAE norm :",
        f"{best_val_mae:.6f}",
    )
    print("Checkpoint        :", checkpoint_path)
    print("History           :", history_path)
    print("Test evaluation   : NOT PERFORMED")
    print("=" * 78)


if __name__ == "__main__":
    main()