import gc
from dataclasses import dataclass, field
from typing import Dict, Literal, Tuple, Type

import numpy as np
import torch
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE

from f3rm.features.clip_extract import extract_clip_features
from f3rm.features.dino_extract import extract_dino_features


@dataclass
class FeatureDataManagerConfig(VanillaDataManagerConfig):
    _target: Type = field(default_factory=lambda: FeatureDataManager)
    feature_type: Literal["CLIP", "DINO"] = "CLIP"


feat_type_to_extract_fn = {
    "CLIP": extract_clip_features,
    "DINO": extract_dino_features,
}


class FeatureDataManager(VanillaDataManager):
    config: FeatureDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        test_mode = kwargs["test_mode"]

        if self.config.feature_type not in feat_type_to_extract_fn:
            raise ValueError(f"Unknown feature type {self.config.feature_type}")
        extract_fn = feat_type_to_extract_fn[self.config.feature_type]

        # Get image filenames, if inference mode only use one image to get feature dimensionality
        image_fnames = self.train_dataset.image_filenames + self.eval_dataset.image_filenames
        if test_mode == "inference":
            CONSOLE.print("Inference mode, only using one image to get feature dimensionality")
            image_fnames = image_fnames[:1]

        # Extract features
        CONSOLE.print(f"Extracting {self.config.feature_type} features for {len(image_fnames)} images...")
        features = extract_fn(image_fnames, self.device)

        # Set metadata, so we can initialize model with feature dimensionality
        self.train_dataset.metadata["feature_type"] = self.config.feature_type
        self.train_dataset.metadata["feature_dim"] = features.shape[-1]

        # If inference mode then we're done as we only needed to set the metadata to initialize the model
        if test_mode == "inference":
            return

        # Split into train and eval features
        self.train_features = features[: len(self.train_dataset)]
        self.eval_features = features[len(self.train_dataset) :]
        assert len(self.eval_features) == len(self.eval_dataset)

        # Determine scaling factors for nearest neighbor interpolation
        feat_h, feat_w = features.shape[1:3]
        im_h = set(self.train_dataset.cameras.image_height.squeeze().tolist())
        im_w = set(self.train_dataset.cameras.image_width.squeeze().tolist())
        assert len(im_h) == 1, "All images must have the same height"
        assert len(im_w) == 1, "All images must have the same width"
        im_h, im_w = im_h.pop(), im_w.pop()
        self.scale_h = feat_h / im_h
        self.scale_w = feat_w / im_w
        assert np.isclose(
            self.scale_h, self.scale_w, atol=1.5e-3
        ), f"Scales must be similar, got h={self.scale_h} and w={self.scale_w}"

        # Garbage collect
        torch.cuda.empty_cache()
        gc.collect()

    def next_train(self, step: int) -> Tuple[RayBundle, Dict]:
        """Nearest neighbor interpolation of features"""
        ray_bundle, batch = super().next_train(step)
        ray_indices = batch["indices"]
        camera_idx = ray_indices[:, 0]
        y_idx = (ray_indices[:, 1] * self.scale_h).long()
        x_idx = (ray_indices[:, 2] * self.scale_w).long()
        batch["feature"] = self.train_features[camera_idx, y_idx, x_idx]
        return ray_bundle, batch

    def next_eval(self, step: int) -> Tuple[RayBundle, Dict]:
        """Nearest neighbor interpolation of features"""
        ray_bundle, batch = super().next_eval(step)
        ray_indices = batch["indices"]
        camera_idx = ray_indices[:, 0]
        y_idx = (ray_indices[:, 1] * self.scale_h).long()
        x_idx = (ray_indices[:, 2] * self.scale_w).long()
        batch["feature"] = self.eval_features[camera_idx, y_idx, x_idx]
        return ray_bundle, batch
