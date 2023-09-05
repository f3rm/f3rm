import torch
from torch import nn
from torchtyping import TensorType


class FeatureRenderer(nn.Module):
    """Just a weighted sum."""

    @classmethod
    def forward(
        cls,
        feature: TensorType["bs":..., "num_samples", "num_channels"],
        weights: TensorType["bs":..., "num_samples", 1],
    ) -> TensorType["bs":..., "num_classes"]:
        output = torch.sum(weights * feature, dim=-2)
        return output
