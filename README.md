# NutriChat ICICyTA — Multimodal Fusion Experiments

Research code and recorded results for experiments E1–E5 comparing image-only, image-text, image-clinical, three-modality concatenation, and attention fusion for food nutrient regression. Clinical profiles are **synthetic**, not real patient records.

## Repository contents
- `scripts/`: supplied Python training, evaluation, data-loader, and clinical preprocessing scripts.
- `config/`: supplied E1–E5 YAML configurations, clinical preprocessing statistics, and target statistics.
- `src/multimodal/`: E4 concatenation and E5 attention model definitions.
- `requirements.txt`: indicative Python dependencies; not a fully locked environment.
- `results/training/`: frozen run configurations, run summaries, and training histories for E1–E5.
- `results/final/`: final held-out test metrics and comparison table.

## Recorded evaluation
| Experiment | Test normalized MAE |
|---|---:|
| E1 | 0.201803 |
| E2 | 0.198327 |
| E3 | 0.202514 |
| E4 | 0.193453 |
| E5 | 0.212341 |

E1 was selected based on validation, not test. E4 had the lowest observed held-out test nMAE; this is a descriptive comparison, not a statistical or clinical superiority claim.

## Reproducibility status
This is a **research audit package**, not yet a fully self-contained runnable reproduction. The original data files, images, model checkpoints, remaining implementation modules (`src/dataset/`, `src/evaluation/`, and any E1–E3 model modules), an exact environment lockfile, and path adjustments may be needed. Do not claim the experiments can be reproduced end-to-end from this repository alone. Review scripts for local absolute paths and other dependencies before public release.

The clinical profiles were synthetically generated for computational context experiments; they are not validated clinical patient–meal pairs.

## Important path check
Some training scripts calculate `PROJECT_ROOT = Path(__file__).resolve().parents[2]`, which may point above this repository when scripts are placed directly in `scripts/`. Before running, verify imports and adjust the project root to the repository directory. These scripts are supplied as audit artifacts, not guaranteed turnkey execution.

## Installation and Environment

The experiments were implemented in Python using PyTorch and
Hugging Face Transformers.

Create a virtual environment and install the dependencies:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

The requirements file specifies PyTorch 2.5.1,
Torchvision 0.20.1, and Transformers 4.46.3.
Other package versions are not fully pinned.

Installation alone does not reproduce the reported results.
The required dataset, preprocessing artifacts, and
model checkpoints must also be available.

## Dataset Provenance and Reproducibility Limitations

The experiments use 3,490 samples divided into
2,443 training, 523 validation, and 524 held-out test samples.

A dataset provenance audit was conducted to examine
consistency between the recovered multimodal dataset and
the available frozen-record reference.

All 3,490 record identifiers were matched, and the 11
shared non-identifier columns were identical across
the compared records, with zero mismatched cells.

However, the recovered multimodal CSV does not match
the SHA-256 checksum recorded in the frozen manifest.
Consequently, byte-identical recovery of the complete
historical frozen dataset has not been established.

The comparison covers only the columns available in
the frozen-record reference and does not establish
equivalence of every column in the recovered dataset.

Supporting audit files:

- `results/conference2_provenance_audit.json`
- `results/conference2_provenance_audit.md`
- `config/conference2_pretest_freeze_manifest.json`

Clinical contexts used in the multimodal experiments
are synthetic and must not be interpreted as real
patient records or evidence of clinical effectiveness.

Model selection was performed exclusively using
validation normalized mean absolute error (nMAE).
E1 was selected based on the lowest validation nMAE.

E4 achieved the lowest held-out test nMAE, which is
reported as a descriptive result rather than used
for retrospective model selection.

The repository documents the experiment implementation
and available results. Full numerical reproduction
requires the relevant datasets and checkpoints, which
are not included in this lightweight source package.

## Execution Paths and Repository Structure

The canonical training scripts for experiments E1–E5 are located under `experiments/`.

| Experiment | Canonical training script |
|---|---|
| E1 | `experiments/exp01_vision/train_conference2_e1_image.py` |
| E2 | `experiments/exp02_multimodal/train_conference2_e2_image_text.py` |
| E3 | `experiments/exp02_multimodal/train_conference2_e3_image_clinical.py` |
| E4 | `experiments/exp02_multimodal/train_conference2_e4_image_text_clinical.py` |
| E5 | `experiments/exp02_multimodal/train_conference2_e5_attention_fusion.py` |

These five scripts resolve the repository root using `Path(__file__).resolve().parents[2]`. This path has been verified against the repository directory structure.

The `scripts/` directory contains additional historical or convenience copies. The canonical training implementations are located in `experiments/`.

The validation and test runners use the current working directory as their default root unless a root argument is supplied. They should therefore be invoked from the repository root when using their default settings.

The original datasets, image files, and trained checkpoints are not included in this lightweight repository. Full numerical reproduction requires access to these external artifacts.

The held-out test metrics in `results/` represent recorded experimental results. Re-running the held-out test is not necessary to inspect the reported performance.
