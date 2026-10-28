import torch
import torch.nn as nn
from transformers import ViTModel


class NutriViTRegressor(nn.Module):

    def __init__(
        self,
        model_name="google/vit-base-patch16-224",
        num_targets=4,
        dropout=0.2,
    ):
        super().__init__()

        self.backbone = ViTModel.from_pretrained(
            model_name,
            add_pooling_layer=False,
        )

        hidden_size = (
            self.backbone.config.hidden_size
        )

        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                hidden_size,
                256,
            ),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(
                256,
                num_targets,
            ),
        )

    def forward(
        self,
        pixel_values,
    ):
        outputs = self.backbone(
            pixel_values=pixel_values
        )

        cls_embedding = (
            outputs.last_hidden_state[:, 0, :]
        )

        predictions = self.regressor(
            cls_embedding
        )

        return predictions