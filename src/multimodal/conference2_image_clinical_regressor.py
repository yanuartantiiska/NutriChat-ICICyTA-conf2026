import torch
import torch.nn as nn
from transformers import AutoModel


class ImageClinicalNutritionRegressor(nn.Module):
    """
    Conference 2 - E3
    Image + synthetic clinical-context nutrient regression.

    Architecture
    ------------
    Image:
        ViT CLS 768 -> 256

    Clinical:
        16 -> 256 -> 256

    Fusion:
        concat [256 + 256] = 512

    Regression:
        512 -> 256 -> 5

    Notes
    -----
    Clinical profiles are synthetic contextual scenarios.
    They are not naturally observed patient-meal pairs.
    """

    def __init__(
        self,
        image_model_name="google/vit-base-patch16-224",
        clinical_input_dim=16,
        projection_dim=256,
        regression_hidden_dim=256,
        num_targets=5,
        dropout=0.2,
    ):
        super().__init__()

        self.image_model_name = image_model_name
        self.clinical_input_dim = clinical_input_dim
        self.projection_dim = projection_dim
        self.num_targets = num_targets

        # --------------------------------------------------
        # IMAGE ENCODER
        # Trainable, matching E1/E2 policy.
        # --------------------------------------------------

        self.image_encoder = AutoModel.from_pretrained(
            image_model_name
        )

        image_hidden = (
            self.image_encoder.config.hidden_size
        )

        if image_hidden != 768:
            raise RuntimeError(
                f"Expected ViT hidden size 768, "
                f"got {image_hidden}"
            )

        self.image_projection = nn.Sequential(
            nn.Linear(
                image_hidden,
                projection_dim,
            ),
            nn.GELU(),
        )

        # --------------------------------------------------
        # CLINICAL ENCODER
        # 16-D frozen preprocessing representation
        # -> trainable 256-D representation.
        # --------------------------------------------------

        self.clinical_encoder = nn.Sequential(
            nn.Linear(
                clinical_input_dim,
                256,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                projection_dim,
            ),
            nn.GELU(),
        )

        # --------------------------------------------------
        # REGRESSION HEAD
        # Same fused dimensionality as E2:
        # 256 image + 256 second modality = 512.
        # --------------------------------------------------

        fused_dim = (
            projection_dim * 2
        )

        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                fused_dim,
                regression_hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                regression_hidden_dim,
                num_targets,
            ),
        )

    def forward(
        self,
        pixel_values,
        clinical_features,
    ):
        # --------------------------------------------------
        # Image CLS representation
        # --------------------------------------------------

        image_outputs = self.image_encoder(
            pixel_values=pixel_values
        )

        image_cls = (
            image_outputs
            .last_hidden_state[:, 0, :]
        )

        image_embedding = (
            self.image_projection(
                image_cls
            )
        )

        # --------------------------------------------------
        # Clinical representation
        # --------------------------------------------------

        if clinical_features.ndim != 2:
            raise RuntimeError(
                "clinical_features must have "
                "shape [batch, features]."
            )

        if (
            clinical_features.shape[1]
            != self.clinical_input_dim
        ):
            raise RuntimeError(
                "Expected clinical input dimension "
                f"{self.clinical_input_dim}, got "
                f"{clinical_features.shape[1]}."
            )

        clinical_embedding = (
            self.clinical_encoder(
                clinical_features
            )
        )

        # --------------------------------------------------
        # Concatenation fusion
        # --------------------------------------------------

        fused = torch.cat(
            [
                image_embedding,
                clinical_embedding,
            ],
            dim=1,
        )

        expected_fused_dim = (
            self.projection_dim * 2
        )

        if fused.shape[1] != expected_fused_dim:
            raise RuntimeError(
                f"Expected fused dimension "
                f"{expected_fused_dim}, got "
                f"{fused.shape[1]}."
            )

        # --------------------------------------------------
        # Multitarget normalized regression
        # --------------------------------------------------

        predictions = (
            self.regression_head(
                fused
            )
        )

        return predictions
