"""
Conference 2 final held-out TEST evaluator.

Execute only after frozen-artifact verification.
Do not use TEST results for model selection or tuning.
"""

from pathlib import Path
import gc
import hashlib
import json

import numpy as np
import torch
import yaml

from torch.utils.data import DataLoader

from experiments.conference2_validation_runner import (
    TARGETS,
    CONFIGS,
    make_model,
    make_dataset,
    checkpoint_path,
    compute_metrics,
)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)
    return h.hexdigest()


def verify_freeze(root):
    freeze_path = (
        root / "config/conference2_evaluator_freeze.json"
    )

    manifest_path = (
        root / "config/conference2_pretest_freeze_manifest.json"
    )

    freeze = json.loads(
        freeze_path.read_text(encoding="utf-8")
    )

    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )

    for info in freeze["source_files"].values():
        path = root / info["path"]

        if sha256(path) != info["sha256"]:
            raise RuntimeError(
                f"Frozen evaluator artifact changed: {path}"
            )

    for info in manifest["experiments"].values():
        for label in ("config", "checkpoint"):
            path = root / info[label]

            if sha256(path) != info[f"{label}_sha256"]:
                raise RuntimeError(
                    f"Frozen experiment artifact changed: {path}"
                )

    for info in manifest["shared_artifacts"].values():
        path = root / info["path"]

        if sha256(path) != info["sha256"]:
            raise RuntimeError(
                f"Frozen shared artifact changed: {path}"
            )

    if freeze["selected_model"] != "E1":
        raise RuntimeError(
            "Frozen validation model selection changed"
        )

    if freeze["test_evaluated"] is not False:
        raise RuntimeError(
            "Invalid frozen TEST status"
        )

    return freeze


def run_test(root=None, batch_size=8):
    """
    Final evaluation entry point.

    TEST dataset is accessed only when this function
    is explicitly invoked.
    """
    root = Path.cwd() if root is None else Path(root)

    # Refuse repeated evaluation before opening TEST data.
    output = (
        root / "results/conference2_final_test_results.json"
    )

    if output.exists():
        raise RuntimeError(
            "Final TEST results already exist. "
            "Refusing repeated TEST evaluation."
        )

    verify_freeze(root)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    from src.evaluation.target_scaler import TargetScaler

    scaler = TargetScaler(
        root / "config/conference2_target_stats.json"
    )

    results = {}

    print("=" * 76)
    print("CONFERENCE 2 - FINAL HELD-OUT TEST EVALUATION")
    print("=" * 76)
    print("Device:", device)
    print("Selected model frozen: E1")
    print("TEST used for model selection: NO")

    for exp, filename in CONFIGS.items():
        with (root / "config" / filename).open(
            encoding="utf-8"
        ) as f:
            cfg = yaml.safe_load(f)

        test_cfg = dict(cfg)
        test_cfg["data"] = dict(cfg["data"])

        test_cfg["data"]["val_csv"] = (
            cfg["data"]["test_csv"]
        )

        dataset = make_dataset(
            exp,
            test_cfg,
            root,
            scaler,
        )

        if len(dataset) != 524:
            raise RuntimeError(
                f"{exp}: expected 524 TEST samples, "
                f"found {len(dataset)}"
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

                predictions.append(
                    pred_original.numpy()
                )

                ground_truth.append(
                    batch["target"].numpy()
                )

        y_pred = np.concatenate(
            predictions,
            axis=0,
        )

        y_true = np.concatenate(
            ground_truth,
            axis=0,
        )

        if y_pred.shape != (524, 5):
            raise RuntimeError(
                f"{exp}: unexpected prediction shape"
            )

        if not (
            np.isfinite(y_pred).all()
            and np.isfinite(y_true).all()
        ):
            raise RuntimeError(
                f"{exp}: non-finite values"
            )

        normalized_mae, per_target = compute_metrics(
            y_true,
            y_pred,
            scaler,
        )

        results[exp] = {
            "samples": 524,
            "normalized_mae": normalized_mae,
            "per_target": per_target,
        }

        print(
            f"{exp}: normalized MAE = "
            f"{normalized_mae:.8f}"
        )

        del model, checkpoint, dataset, loader
        gc.collect()

        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = (
        root / "results/conference2_final_test_results.json"
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output.exists():
        raise RuntimeError(
            "Final TEST result already exists. "
            "Refusing to overwrite."
        )

    output.write_text(
        json.dumps(
            {
                "protocol": "CONFERENCE2_FINAL_TEST_V1",
                "selected_model": "E1",
                "test_used_for_model_selection": False,
                "target_names": TARGETS,
                "experiments": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nFinal results:", output)
    print("FINAL TEST EVALUATION: COMPLETE")

    return results
