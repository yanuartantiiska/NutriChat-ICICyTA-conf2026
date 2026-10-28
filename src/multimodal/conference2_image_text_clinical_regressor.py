import torch
import torch.nn as nn
from transformers import AutoModel


class ImageTextClinicalNutritionRegressor(nn.Module):
    """E4 three-modality concatenation nutrient regressor."""

    def __init__(
        self,
        image_model_name="google/vit-base-patch16-224",
        text_model_name="indobenchmark/indobert-base-p1",
        image_hidden_dim=768,
        text_hidden_dim=768,
        clinical_input_dim=16,
        projection_dim=256,
        clinical_hidden_dim=256,
        regression_hidden_dim=256,
        dropout=0.2,
        num_targets=5,
        freeze_text=True,
    ):
        super().__init__()

        self.image_encoder = AutoModel.from_pretrained(
            image_model_name
        )

        self.text_encoder = AutoModel.from_pretrained(
            text_model_name
        )

        self.freeze_text = freeze_text
        self.clinical_input_dim = clinical_input_dim

        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            self.text_encoder.eval()

        self.image_projection = nn.Sequential(
            nn.Linear(image_hidden_dim, projection_dim),
            nn.GELU(),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden_dim, projection_dim),
            nn.GELU(),
        )

        self.clinical_encoder = nn.Sequential(
            nn.Linear(clinical_input_dim, clinical_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(clinical_hidden_dim, projection_dim),
            nn.GELU(),
        )

        self.fused_dim = 3 * projection_dim

        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.fused_dim, regression_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regression_hidden_dim, num_targets),
        )

    def train(self, mode=True):
        super().train(mode)

        if self.freeze_text:
            self.text_encoder.eval()

        return self

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        clinical_features,
    ):
        if (
            clinical_features.ndim != 2
            or clinical_features.shape[1] != self.clinical_input_dim
        ):
            raise ValueError(
                "Clinical features must have shape "
                f"(batch, {self.clinical_input_dim})."
            )

        image_outputs = self.image_encoder(
            pixel_values=pixel_values
        )
        image_cls = image_outputs.last_hidden_state[:, 0, :]

        if self.freeze_text:
            with torch.no_grad():
                text_outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        text_cls = text_outputs.last_hidden_state[:, 0, :]

        image_features = self.image_projection(image_cls)
        text_features = self.text_projection(text_cls)
        clinical_features = self.clinical_encoder(
            clinical_features
        )

        fused = torch.cat(
            [
                image_features,
                text_features,
                clinical_features,
            ],
            dim=1,
        )

        if fused.shape[1] != self.fused_dim:
            raise RuntimeError("Unexpected fusion dimension.")

        return self.regression_head(fused)
