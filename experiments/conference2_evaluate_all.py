"""
Conference 2 unified evaluator.

Preflight V1:
- No TEST data access
- No model instantiation
- No checkpoint loading
- No inference
"""

import argparse
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]

EXPECTED_TARGETS = [
    "mass_g",
    "energy_kcal",
    "protein_g",
    "fat_g",
    "carbohydrate_g",
]

CONFIG_FILES = {
    "E1": "conference2_e1_image.yaml",
    "E2": "conference2_e2_image_text.yaml",
    "E3": "conference2_e3_image_clinical.yaml",
    "E4": "conference2_e4_image_text_clinical.yaml",
    "E5": "conference2_e5_attention_fusion.yaml",
}

EXPECTED_EXPERIMENTS = {
    "E1": "conference2_e1_image_only",
    "E2": "conference2_e2_image_text",
    "E3": "conference2_e3_image_clinical",
    "E4": "conference2_e4_image_text_clinical",
    "E5": "conference2_e5_attention_fusion",
}

EXPECTED_CHECKPOINTS = {
    "E1": "e1_image_only_best.pt",
    "E2": "e2_image_text_best.pt",
    "E3": "e3_image_clinical_best.pt",
    "E4": "e4_image_text_clinical_best.pt",
    "E5": "e5_attention_fusion_best.pt",
}


def resolve_checkpoint(config):
    checkpoint = config["checkpoint"]

    if "path" in checkpoint:
        return ROOT / checkpoint["path"]

    return (
        ROOT
        / checkpoint["directory"]
        / checkpoint["filename"]
    )


def run_preflight():
    print("=" * 76)
    print("CONFERENCE 2 - UNIFIED EVALUATOR PREFLIGHT V1")
    print("=" * 76)

    scaler_path = (
        ROOT / "config/conference2_target_stats.json"
    )

    if not scaler_path.is_file():
        raise FileNotFoundError(scaler_path)

    with scaler_path.open(encoding="utf-8") as f:
        scaler = json.load(f)

    assert scaler["targets"] == EXPECTED_TARGETS, (
        "Scaler target order mismatch."
    )

    for name in EXPECTED_TARGETS:
        assert name in scaler["mean"], (
            f"Missing scaler mean: {name}"
        )
        assert name in scaler["std"], (
            f"Missing scaler std: {name}"
        )
        assert float(scaler["std"][name]) > 0, (
            f"Invalid scaler std: {name}"
        )

    print("Scaler: PASS")
    print("Target order: PASS")

    for experiment, filename in CONFIG_FILES.items():
        path = ROOT / "config" / filename

        if not path.is_file():
            raise FileNotFoundError(path)

        with path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)

        assert (
            config["experiment"]["protocol"]
            == "conference2_frozen_v1"
        ), f"{experiment}: protocol mismatch"

        assert (
            config["experiment"]["seed"] == 42
        ), f"{experiment}: seed mismatch"

        assert (
            config["experiment"]["name"]
            == EXPECTED_EXPERIMENTS[experiment]
        ), f"{experiment}: experiment name mismatch"

        assert (
            config["targets"]["names"]
            == EXPECTED_TARGETS
        ), f"{experiment}: target order mismatch"

        configured_scaler = (
            ROOT / config["targets"]["scaler"]
        )

        assert (
            configured_scaler.resolve()
            == scaler_path.resolve()
        ), f"{experiment}: scaler path mismatch"

        checkpoint_path = resolve_checkpoint(config)

        assert (
            checkpoint_path.name
            == EXPECTED_CHECKPOINTS[experiment]
        ), f"{experiment}: checkpoint filename mismatch"

        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)

        if checkpoint_path.stat().st_size == 0:
            raise RuntimeError(
                f"{experiment}: empty checkpoint"
            )

        print(
            f"{experiment}: PASS | "
            f"{checkpoint_path.relative_to(ROOT)}"
        )

    print("\n[SAFETY]")
    print("TEST dataset accessed : NO")
    print("TEST predictions made : NO")
    print("Checkpoints loaded    : NO")
    print("Models instantiated   : NO")
    print("Training performed    : NO")

    print("=" * 76)
    print("UNIFIED EVALUATOR PREFLIGHT V1: PASS")


def main():
    parser = argparse.ArgumentParser()

    mode = parser.add_mutually_exclusive_group(
        required=True
    )

    mode.add_argument(
        "--preflight",
        action="store_true",
        help="Run safe configuration preflight.",
    )

    mode.add_argument(
        "--validation",
        action="store_true",
        help="Reproduce validation metrics for E1-E5.",
    )

    args = parser.parse_args()

    if args.preflight:
        run_preflight()
        return

    if args.validation:
        from experiments.conference2_validation_runner import (
            run_validation,
        )
        run_validation()
        return

    parser.error("Unsupported evaluation mode.")


if __name__ == "__main__":
    main()
