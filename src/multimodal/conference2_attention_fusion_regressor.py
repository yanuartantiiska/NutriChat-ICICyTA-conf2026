import torch
from torch import nn

from src.multimodal.conference2_image_text_clinical_regressor import (
    ImageTextClinicalNutritionRegressor,
)


class ImageTextClinicalAttentionRegressor(
    ImageTextClinicalNutritionRegressor
):
    """
    E5: ViT + frozen IndoBERT + Clinical MLP.

    Three projected modality embeddings are treated as tokens
    and fused using multi-head self-attention.

    Output: five nutrient regression predictions.
    """

    def __init__(
        self,
        image_model_name,
        text_model_name,
        image_hidden_dim=768,
        text_hidden_dim=768,
        clinical_input_dim=16,
        projection_dim=256,
        clinical_hidden_dim=256,
        regression_hidden_dim=256,
        dropout=0.2,
        num_targets=5,
        freeze_text=True,
        num_heads=8,
        num_layers=1,
        attention_dropout=0.2,
    ):
        super().__init__(
            image_model_name=image_model_name,
            text_model_name=text_model_name,
            image_hidden_dim=image_hidden_dim,
            text_hidden_dim=text_hidden_dim,
            clinical_input_dim=clinical_input_dim,
            projection_dim=projection_dim,
            clinical_hidden_dim=clinical_hidden_dim,
            regression_hidden_dim=regression_hidden_dim,
            dropout=dropout,
            num_targets=num_targets,
            freeze_text=freeze_text,
        )

        assert projection_dim % num_heads == 0
        assert num_layers >= 1

        # ViT pooler is unused because forward() reads
        # last_hidden_state[:, 0, :] directly.
        # Freeze only the unused pooler parameters.
        if self.image_encoder.pooler is not None:
            for param in self.image_encoder.pooler.parameters():
                param.requires_grad_(False)

        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=projection_dim,
                num_heads=num_heads,
                dropout=attention_dropout,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.attention_norms = nn.ModuleList([
            nn.LayerNorm(projection_dim)
            for _ in range(num_layers)
        ])

        self.attention_dropout = nn.Dropout(dropout)

        # Remove the unused E4 concatenation regression head.
        # E5 uses attention_regression_head exclusively.
        del self.regression_head

        self.attention_regression_head = nn.Sequential(
            nn.Linear(projection_dim, regression_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(regression_hidden_dim, num_targets),
        )

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
        clinical_features,
    ):
        image_outputs = self.image_encoder(
            pixel_values=pixel_values
        )
        image_cls = image_outputs.last_hidden_state[:, 0, :]

        with torch.no_grad():
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            text_cls = text_outputs.last_hidden_state[:, 0, :]

        image_token = self.image_projection(image_cls)
        text_token = self.text_projection(text_cls)

        clinical_token = self.clinical_encoder(
            clinical_features
        )

        assert clinical_token.shape[-1] == image_token.shape[-1], (
            "Clinical and image token dimensions differ."
        )
        assert text_token.shape[-1] == image_token.shape[-1], (
            "Text and image token dimensions differ."
        )

        tokens = torch.stack(
            [image_token, text_token, clinical_token],
            dim=1,
        )

        assert tokens.ndim == 3
        assert tokens.shape[1] == 3
        assert tokens.shape[2] == 256

        for attention, norm in zip(
            self.attention_layers,
            self.attention_norms,
        ):
            attended, _ = attention(
                tokens,
                tokens,
                tokens,
                need_weights=False,
            )
            tokens = norm(
                tokens + self.attention_dropout(attended)
            )

        fused = tokens.mean(dim=1)

        predictions = self.attention_regression_head(
            fused
        )

        return predictions
