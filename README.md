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

All 3,490 record identifiers were matched, and The initial recovery audit confirmed matching record identifiers and 11 shared non-identifier columns. A subsequent comprehensive comparison of the regenerated frozen dataset against the recovered dataset confirmed zero mismatches across all 39 shared non-identifier columns for the 3,490 records.

The original local multimodal CSV copy did not match the SHA-256 checksum recorded in the historical frozen manifest due to serialization inconsistencies. However, independent re-execution of the preserved multimodal text generator, using the original grounding input, templates, configuration, and fixed random seed of 42, successfully regenerated a dataset whose SHA-256 checksum exactly matched the frozen manifest. A record-level comparison with the recovered dataset further confirmed matching identifiers and zero mismatches across 39 shared non-identifier columns. Therefore, byte-identical regeneration of the frozen dataset has been established under the audited conditions, while exact model-training reproducibility remains unverified.

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

The canonical training scripts are located under `experiments/`:

| Experiment | Canonical training script |
|---|---|
| E1 | `experiments/exp01_vision/train_conference2_e1_image.py` |
| E2 | `experiments/exp02_multimodal/train_conference2_e2_image_text.py` |
| E3 | `experiments/exp02_multimodal/train_conference2_e3_image_clinical.py` |
| E4 | `experiments/exp02_multimodal/train_conference2_e4_image_text_clinical.py` |
| E5 | `experiments/exp02_multimodal/train_conference2_e5_attention_fusion.py` |

These five scripts resolve the repository root using
`Path(__file__).resolve().parents[2]`. This path was
verified against the published directory structure.

The `scripts/` directory contains additional historical
or convenience copies. Use the canonical scripts in
`experiments/` when inspecting the training implementation.

Validation and test runners use the current working
directory as the default root unless a root argument
is supplied. Run commands from the repository root
when using these entry points.

The dataset, image files, and trained checkpoints are
not included in this lightweight repository. Training
and evaluation commands require these external artifacts
and are not guaranteed to execute successfully without
the original data layout.

The held-out test results supplied in `results/` are
historical recorded results. Re-running the held-out
test is not required to inspect the reported metrics.

## Synthetic Data Provenance and Reproducibility Audit

This repository accompanies the Conference 2 manuscript,
"Image-Based Food Nutrient Prediction: An Evaluation of Visual
and Auxiliary Context Fusion Strategies."

### Synthetic clinical profiles

Clinical profiles were synthetically generated using a fixed
random seed of 42. An independent generator execution
reproduced all 3,490 reference profiles, with zero mismatches
across 17 compared clinical and profile attributes.

The synthetic clinical contexts do not represent observed
patient-meal associations and must not be interpreted as
real patient records or evidence of clinical validity.

### Synthetic multimodal text

The preserved multimodal text generator was independently
executed using the original grounding input, text templates,
design configuration, and random seed 42.

The generated dataset contained 3,490 unique records and
3,490 unique user utterances.

The resulting CSV matched the SHA-256 checksum recorded
in the historical frozen manifest:

070bcdbc965b276bad97e4c70ed79d13646d1b7a77c6af7149dae40fb62c7d39

A comparison against the recovered multimodal dataset
confirmed matching record identifiers and zero mismatches
across 39 shared non-identifier columns.

### Model selection and evaluation

Five configurations were evaluated:

- E1: Image-only
- E2: Image + Text
- E3: Image + Synthetic Clinical Context
- E4: Image + Text + Synthetic Clinical Context (Concatenation)
- E5: Attention-based Multimodal Fusion

Model selection was based exclusively on validation
normalized mean absolute error (nMAE).

E1 achieved the lowest validation nMAE (0.182347)
and was selected according to the predefined criterion.

E4 achieved the lowest held-out test nMAE (0.193453),
which is reported descriptively and was not used
for retrospective model selection.

### Scope and limitations

The synthetic data-generation procedures have been
independently reproduced under the audited conditions.

However, the complete E1-E5 training and evaluation
pipeline has not been independently rerun.

The reported model metrics are retained historical
evaluation results. Data-generation reproducibility
must not be interpreted as exact model-level
reproducibility or clinical validation.

The grounding dataset and model checkpoints may not
be included in the public repository. Their absence
limits end-to-end reproducibility from the repository alone.

### Dataset attribution

This research uses Nutrition5k as a source dataset:

Thames, Q., et al. (2021).
Nutrition5k: Towards Automatic Nutritional Understanding
of Generic Food.
Proceedings of the IEEE/CVF Conference on Computer
Vision and Pattern Recognition (CVPR).

Official dataset repository:
https://github.com/google-research-datasets/Nutrition5k

Nutrition5k is distributed under the Creative Commons
Attribution 4.0 International (CC BY 4.0) license.
See the official dataset repository for applicable
license terms and attribution requirements.

Synthetic clinical profiles and multimodal text
queries were generated for computational research
and are not part of the original Nutrition5k dataset.
