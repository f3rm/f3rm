from typing import Dict, Optional, Tuple

from jaxtyping import Float, Shaped
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.fields.base_field import Field
from torch import Tensor


class FeatureField(Field):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.feature_dim = feature_dim
        print("Created FeatureField")

    def get_density(
        self, ray_samples: RaySamples
    ) -> Tuple[Shaped[Tensor, "*batch 1"], Float[Tensor, "*batch num_features"]]:
        raise NotImplementedError

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        pass
