from dataclasses import dataclass, field
from typing import Dict, List, Type

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from torch.nn import Parameter

from f3rm.field import FeatureField
from f3rm.renderer import FeatureRenderer


@dataclass
class FeatureFieldModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: FeatureFieldModel)


class FeatureFieldModel(NerfactoModel):
    config: FeatureFieldModelConfig

    feature_field: FeatureField
    feature_renderer: FeatureRenderer

    def populate_modules(self):
        super().populate_modules()

        # Create feature field
        feature_dim = self.kwargs["metadata"]["feature_dim"]
        if feature_dim <= 0:
            raise ValueError(f"Feature dimensionality must be positive, not {feature_dim}")

        self.feature_field = FeatureField(feature_dim)
        self.feature_renderer = FeatureRenderer()

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        # param_groups["feature_field"] = list(self.feature_field.parameters())
        return param_groups
