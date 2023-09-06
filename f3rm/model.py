from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.model_components.losses import (
    orientation_loss,
    pred_normal_loss,
    scale_gradients_by_distance_squared,
)
from nerfstudio.models.nerfacto import NerfactoModel, NerfactoModelConfig
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.viewer.server.viewer_elements import (
    ViewerButton,
    ViewerNumber,
    ViewerText,
)
from torch.nn import Parameter

from f3rm.feature_field import FeatureField, FeatureFieldHeadNames
from f3rm.pca_colormap import apply_pca_colormap_return_proj
from f3rm.renderer import FeatureRenderer


@dataclass
class FeatureFieldModelConfig(NerfactoModelConfig):
    _target: Type = field(default_factory=lambda: FeatureFieldModel)
    feat_loss_weight: float = 1e-3


@dataclass
class _GuiState:
    positives: List[str] = field(default_factory=list)
    pos_embed: Optional[torch.Tensor] = None
    negatives: List[str] = field(default_factory=list)
    neg_embed: Optional[torch.Tensor] = None
    softmax_temp: float = 0.1


class FeatureFieldModel(NerfactoModel):
    config: FeatureFieldModelConfig

    feature_field: FeatureField
    renderer_feature: FeatureRenderer
    pca_proj: Optional[torch.Tensor] = None
    gui_state: _GuiState = _GuiState()

    def populate_modules(self):
        super().populate_modules()

        # Create feature field
        feature_dim = self.kwargs["metadata"]["feature_dim"]
        if feature_dim <= 0:
            raise ValueError(f"Feature dimensionality must be positive, not {feature_dim}")

        self.feature_field = FeatureField(feature_dim=feature_dim, spatial_distortion=self.field.spatial_distortion)
        self.renderer_feature = FeatureRenderer()
        self.setup_gui()

    def setup_gui(self):
        def refresh_pca_proj(_: ViewerButton):
            self.pca_proj = None
            print("PCA projection set to None")

        self.btn_refresh_pca = ViewerButton("Refresh PCA Projection", cb_hook=refresh_pca_proj)

        # Setup GUI for language features if we're using CLIP
        if self.kwargs["metadata"]["feature_type"] == "CLIP":
            self.setup_language_gui()

    def setup_language_gui(self):
        from f3rm.features.clip import load, tokenize
        from f3rm.features.clip_extract import CLIPArgs

        # Load CLIP
        CONSOLE.print(
            f"Loading CLIP {CLIPArgs.model_name} for Nerfstudio viewer. "
            "If you run out of memory please open a GitHub issue on https://github.com/f3rm/f3rm"
        )
        device = self.kwargs["device"]
        clip_model, _ = load(CLIPArgs.model_name, device=device)
        gui_state = self.gui_state

        @torch.no_grad()
        def update_positives(element: ViewerText):
            # Embed positive texts with CLIP
            text = element.value
            gui_state.positives = [x.strip() for x in text.split(",") if x.strip()]
            if gui_state.positives:
                tokens = tokenize(gui_state.positives).to(device)
                gui_state.pos_embed = clip_model.encode_text(tokens).float()
                # Average embedding if we have multiple positives
                gui_state.pos_embed = gui_state.pos_embed.mean(dim=0, keepdim=True)
                gui_state.pos_embed /= gui_state.pos_embed.norm(dim=-1, keepdim=True)
            else:
                gui_state.pos_embed = None

        @torch.no_grad()
        def update_negatives(element: ViewerText):
            # Embed negative texts with CLIP
            text = element.value
            gui_state.negatives = [x.strip() for x in text.split(",") if x.strip()]
            if gui_state.negatives:
                tokens = tokenize(gui_state.negatives).to(device)
                gui_state.neg_embed = clip_model.encode_text(tokens).float()
                gui_state.neg_embed /= gui_state.neg_embed.norm(dim=-1, keepdim=True)
            else:
                gui_state.neg_embed = None

        def update_softmax(element: ViewerNumber):
            gui_state.softmax_temp = element.value
            print("Updated softmax temperature to", gui_state.softmax_temp)

        # Note: the GUI elements are shown based on alphabetical variable names
        self.hint_text = ViewerText(name="Note:", disabled=True, default_value="Use , to separate labels")
        self.lang_1_pos_text = ViewerText(name="Language (Positives)", default_value="", cb_hook=update_positives)
        self.lang_2_neg_text = ViewerText(name="Language (Negatives)", default_value="", cb_hook=update_negatives)
        self.softmax_temp = ViewerNumber(
            name="Softmax temperature", default_value=gui_state.softmax_temp, cb_hook=update_softmax
        )

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = super().get_param_groups()
        param_groups["feature_field"] = list(self.feature_field.parameters())
        return param_groups

    def get_outputs(self, ray_bundle: RayBundle):
        """Modified from nerfacto.get_outputs to include feature field outputs."""
        ray_samples: RaySamples
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        field_outputs = self.field.forward(ray_samples, compute_normals=self.config.predict_normals)
        if self.config.use_gradient_scaling:
            field_outputs = scale_gradients_by_distance_squared(field_outputs, ray_samples)

        weights = ray_samples.get_weights(field_outputs[FieldHeadNames.DENSITY])
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        rgb = self.renderer_rgb(rgb=field_outputs[FieldHeadNames.RGB], weights=weights)
        with torch.no_grad():
            depth = self.renderer_depth(weights=weights, ray_samples=ray_samples)
        expected_depth = self.renderer_expected_depth(weights=weights, ray_samples=ray_samples)
        accumulation = self.renderer_accumulation(weights=weights)

        # Feature outputs
        ff_outputs = self.feature_field(ray_samples)
        features = self.renderer_feature(features=ff_outputs[FeatureFieldHeadNames.FEATURE], weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "expected_depth": expected_depth,
            "feature": features,
        }

        if self.config.predict_normals:
            normals = self.renderer_normals(normals=field_outputs[FieldHeadNames.NORMALS], weights=weights)
            pred_normals = self.renderer_normals(field_outputs[FieldHeadNames.PRED_NORMALS], weights=weights)
            outputs["normals"] = self.normals_shader(normals)
            outputs["pred_normals"] = self.normals_shader(pred_normals)
        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list

        if self.training and self.config.predict_normals:
            outputs["rendered_orientation_loss"] = orientation_loss(
                weights.detach(), field_outputs[FieldHeadNames.NORMALS], ray_bundle.directions
            )

            outputs["rendered_pred_normal_loss"] = pred_normal_loss(
                weights.detach(),
                field_outputs[FieldHeadNames.NORMALS].detach(),
                field_outputs[FieldHeadNames.PRED_NORMALS],
            )

        for i in range(self.config.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.renderer_depth(weights=weights_list[i], ray_samples=ray_samples_list[i])

        return outputs

    def get_metrics_dict(self, outputs, batch):
        metrics_dict = super().get_metrics_dict(outputs, batch)
        # Compute feature error
        target_feats = batch["feature"].to(self.device)
        metrics_dict["feature_error"] = F.mse_loss(outputs["feature"], target_feats)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        # Compute feature loss
        target_feats = batch["feature"].to(self.device)
        loss_dict["feature_loss"] = self.config.feat_loss_weight * F.mse_loss(outputs["feature"], target_feats)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        # TODO: add feature outputs
        return metrics_dict, images_dict

    @torch.no_grad()
    def get_outputs_for_camera_ray_bundle(self, camera_ray_bundle: RayBundle) -> Dict[str, torch.Tensor]:
        outputs = super().get_outputs_for_camera_ray_bundle(camera_ray_bundle)

        # Compute PCA of features separately, so we can reuse the same projection matrix
        outputs["feature_pca"], self.pca_proj, *_ = apply_pca_colormap_return_proj(outputs["feature"], self.pca_proj)

        # Nothing else to do if not CLIP features or no positives
        if self.kwargs["metadata"]["feature_type"] != "CLIP" or not self.gui_state.positives:
            return outputs

        # Normalize CLIP features rendered by feature field
        clip_features = outputs["feature"]
        clip_features /= clip_features.norm(dim=-1, keepdim=True)

        # If there are no negatives, just show the cosine similarity with the positives
        if not self.gui_state.negatives:
            sims = clip_features @ self.gui_state.pos_embed.T
            # Show the mean similarity if there are multiple positives
            if sims.shape[-1] > 1:
                sims = sims.mean(dim=-1, keepdim=True)
            outputs["similarity"] = sims
            return outputs

        # Use paired softmax method as described in the paper with positive and negative texts
        text_embs = torch.cat([self.gui_state.pos_embed, self.gui_state.neg_embed], dim=0)
        raw_sims = clip_features @ text_embs.T

        # Broadcast positive label similarities to all negative labels
        pos_sims, neg_sims = raw_sims[..., :1], raw_sims[..., 1:]
        pos_sims = pos_sims.broadcast_to(neg_sims.shape)
        paired_sims = torch.cat([pos_sims, neg_sims], dim=-1)

        # Compute paired softmax
        probs = (paired_sims / self.gui_state.softmax_temp).softmax(dim=-1)[..., :1]
        torch.nan_to_num_(probs, nan=0.0)
        sims, _ = probs.min(dim=-1, keepdim=True)
        outputs["similarity"] = sims
        return outputs
