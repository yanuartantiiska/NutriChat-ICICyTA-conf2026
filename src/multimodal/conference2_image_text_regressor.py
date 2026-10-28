import torch
import torch.nn as nn
from transformers import AutoModel


class ImageTextNutritionRegressor(nn.Module):
    """
    Conference-2 E2 Image+Text nutrient regression.

    Image:
        ViT CLS representation [B, 768], trainable.

    Text:
        IndoBERT CLS representation [B, 768], frozen.

    Fusion:
        image 768 -> 256
        text  768 -> 256
        concatenation -> 512
        regression head -> 5 normalized nutrition targets
    """

    def __init__(
        self,
        image_model_name="google/vit-base-patch16-224",
        text_model_name="indobenchmark/indobert-base-p1",
        projection_dim=256,
        regression_hidden=256,
        num_targets=5,
        dropout=0.2,
        freeze_text=True,
    ):
        super().__init__()

        # ---------------------------------------------------------
        # Encoders
        # ---------------------------------------------------------
        self.image_encoder = AutoModel.from_pretrained(
            image_model_name
        )

        self.text_encoder = AutoModel.from_pretrained(
            text_model_name
        )

        image_hidden = self.image_encoder.config.hidden_size
        text_hidden = self.text_encoder.config.hidden_size

        # ---------------------------------------------------------
        # Freeze IndoBERT
        # ---------------------------------------------------------
        self.freeze_text = bool(freeze_text)

        if self.freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # ---------------------------------------------------------
        # Modality projections
        # ---------------------------------------------------------
        self.image_projection = nn.Sequential(
            nn.Linear(image_hidden, projection_dim),
            nn.GELU(),
        )

        self.text_projection = nn.Sequential(
            nn.Linear(text_hidden, projection_dim),
            nn.GELU(),
        )

        # ---------------------------------------------------------
        # Regression head
        # ---------------------------------------------------------
        fused_dim = projection_dim * 2

        self.regression_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused_dim, regression_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(regression_hidden, num_targets),
        )

    def train(self, mode=True):
        """
        Keep frozen IndoBERT in eval mode even while the rest of
        the model is training. This prevents dropout inside the
        frozen text encoder from introducing stochastic features.
        """
        super().train(mode)

        if self.freeze_text:
            self.text_encoder.eval()

        return self

    def encode_image(self, pixel_values):
        outputs = self.image_encoder(
            pixel_values=pixel_values
        )

        # ViT [CLS]
        return outputs.last_hidden_state[:, 0, :]

    def encode_text(
        self,
        input_ids,
        attention_mask,
    ):
        if self.freeze_text:
            with torch.no_grad():
                outputs = self.text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        else:
            outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

        # IndoBERT [CLS]
        return outputs.last_hidden_state[:, 0, :]

    def forward(
        self,
        pixel_values,
        input_ids,
        attention_mask,
    ):
        image_embedding = self.encode_image(
            pixel_values
        )

        text_embedding = self.encode_text(
            input_ids,
            attention_mask,
        )

        image_token = self.image_projection(
            image_embedding
        )

        text_token = self.text_projection(
            text_embedding
        )

        fused = torch.cat(
            [image_token, text_token],
            dim=-1,
        )

        predictions = self.regression_head(
            fused
        )

        return predictions