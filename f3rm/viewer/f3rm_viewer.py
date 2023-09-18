from asyncio import sleep
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import (
    AsyncIterable,
    DefaultDict,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch
from matplotlib.colors import to_rgb
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components.renderers import background_color_override_context
from nerfstudio.models.base_model import Model
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup
from params_proto import ParamsProto, Proto
from termcolor import cprint
from torch import Tensor
from torchtyping import TensorType
from vuer import Vuer
from vuer.events import ServerEvent
from vuer.schemas import Scene, group

from f3rm.model import viewer_utils
from f3rm.pca_colormap import apply_pca_colormap_return_proj
from f3rm.viewer.utils import (
    b64jpg,
    get_camera_from_three,
    get_colormap,
    rotation_matrix,
)

doc = Vuer(
    uri="ws://localhost:8012",
    domain="https://dash.ml/vuer",
    queries=dict(
        reconnect=True,
        debug=True,
        rotation="-90,0,-90",
    ),
    cors="*",
)

Vector3 = Tuple[float, float, float]


class Worker(ParamsProto):
    checkpoint: str = Proto(env="CHECKPOINT", help="checkpoint to load")


# while not Worker.checkpoint:
#     Worker.checkpoint = input("Enter checkpoint: ").strip()


def load_nerfstudio_pipeline(load_dir: str) -> Pipeline:
    load_config = Path(load_dir) / "config.yml"
    print(f"Loading Nerfstudio pipeline from {load_config}")
    _, pipeline, _, _ = eval_setup(
        load_config,
        eval_num_rays_per_chunk=None,
        test_mode="inference",
    )
    return pipeline


async def async_render(model: Model, rb: RayBundle, rays_per_chunk: int) -> AsyncIterable[Dict[str, Tensor]]:
    for i in range(0, len(rb), rays_per_chunk):
        start_idx = i
        end_idx = i + rays_per_chunk

        ray_bundle = rb.get_row_major_sliced_ray_bundle(start_idx, end_idx)
        outputs = model(ray_bundle=ray_bundle)
        yield outputs


async def render(
    pipeline: Pipeline,
    camera: Cameras,
    channels: List[str],
    use_aabb: bool,
    aabb_min: Optional[Vector3],
    aabb_max: Optional[Vector3],
    alpha_threshold: Optional[float],
    colormap: str,
    gain: float,
    normalize: bool,
    clip: Tuple[int, int],
    rotation: Vector3,
    position: Vector3,
    bg_color: str = "#000000",
    **__,
):
    primary_channel = channels[0]
    if primary_channel == "features":
        # temporary hack
        primary_channel = "feature"

    # Handle aabb crop
    if use_aabb and aabb_min is not None and aabb_max is not None:
        aabb = torch.FloatTensor([aabb_min, aabb_max])
        # If min bounds are greater than max bounds, set min bound to 1cm below max bound
        aabb_min_gt = aabb[0] >= aabb[1]
        if aabb_min_gt.any():
            print(f"Invalid aabb box with min bounds {aabb_min} and max_bounds {aabb_max}")
            aabb[0][aabb_min_gt] = aabb[1][aabb_min_gt] - 0.01
            print("Set min bounds to", aabb[0])
        aabb_box = SceneBox(aabb=aabb)
    else:
        aabb_box = None

    # Process background color
    bg_rgb = torch.Tensor(to_rgb(bg_color)).to(pipeline.device)

    # Generate rays and apply rotation and translation
    camera = camera.to(pipeline.device)
    camera_ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=aabb_box)
    if rotation is not None:
        mat = rotation_matrix(*rotation)
        c2w = torch.FloatTensor(mat).to(pipeline.device)
        camera_ray_bundle.origins @= c2w
        camera_ray_bundle.directions @= c2w
    if position is not None:
        position_tensor = torch.FloatTensor(position).to(pipeline.device)
        camera_ray_bundle.origins += position_tensor

    # Forward pass through the model
    outputs = defaultdict(list)
    with background_color_override_context(bg_rgb), torch.no_grad():
        async for output_chunk in async_render(pipeline.model, camera_ray_bundle, rays_per_chunk=1 << 13):
            terminate = yield None
            if terminate:
                return
            for k, v in output_chunk.items():
                outputs[k].append(v)

    # Concat and reshape outputs
    height, width = camera.height.item(), camera.width.item()
    for key, value in outputs.items():
        outputs[key] = torch.cat(value).view(height, width, -1)

    # Process to get outputs
    alpha = outputs["accumulation"]
    if alpha_threshold is not None:
        alpha[alpha < alpha_threshold] = 0
    alpha_np = alpha.squeeze().cpu().numpy()
    alpha_encoded = b64jpg(alpha_np)

    if primary_channel not in outputs:
        raise ValueError(f"Invalid primary channel {primary_channel}")
    primary_output = outputs[primary_channel]

    if primary_channel == "feature":
        # apply pca
        primary_output, viewer_utils.pca_proj, *_ = apply_pca_colormap_return_proj(
            primary_output, viewer_utils.pca_proj
        )

    if primary_output.shape[-1] == 3:
        rgb = primary_output
        rgba = torch.cat([rgb, alpha], dim=-1)
    elif primary_output.shape[-1] == 1:
        cmap = get_colormap(colormap, normalize=normalize, clip=clip, gain=gain)

        rgba = cmap(primary_output.cpu().numpy())
        rgba = rgba[:, :, 0, :]
        rgba[:, :, 3] = alpha_np
    else:
        raise ValueError(f"Invalid primary output shape {primary_output.shape}")
    rgb_encoded = b64jpg(rgba.cpu().numpy())

    # FIXME: the camera height and width is different from the frontend one by rounding error.
    yield ServerEvent(etype="RENDER", data={"rgb": rgb_encoded, "alpha": alpha_encoded})


@doc.spawn(start=True)
async def run_eval(ws_id):
    pipeline = load_nerfstudio_pipeline(
        "/home/william/workspace/vqn/f3rm-public-release/outputs/stata_office/f3rm/2023-09-14_164333"
    )
    pipeline.eval()

    scene = Scene(
        group(key="playground"),
        # set camera position
        key="scene",
    )

    doc.set @ scene

    while True:
        await sleep(0)
        event = doc.pop()

        if not event:
            continue

        if event == "CAMERA_MOVE":
            value = deepcopy(event.value)
            doc.clear()

            world = value.pop("world")
            camera = value.pop("camera")
            # print(camera)
            channels = value.pop("channels", "rgb")

            progressive = value.pop("progressive", 1)
            progressive = np.clip(progressive, 0, 1)

            height = camera["height"]
            width = camera["width"]

            camera = get_camera_from_three([camera], width=width, height=height)

            if progressive < 1:
                quick_cam = deepcopy(camera)
                quick_cam.rescale_output_resolution(progressive)

                async for render_response in render(
                    pipeline,
                    # last_event.client_id,
                    quick_cam,
                    # move model to a model cache, reload.
                    # model,
                    channels=channels,
                    position=world["position"],
                    rotation=world["rotation"],
                    scale=world["scale"],
                    # snapshot=snapshot,
                    **value,
                ):
                    if isinstance(render_response, ServerEvent):
                        # print("rendering")
                        doc @ render_response
                        # logger.since('low-res-render')

            await sleep(0.0001)
            if "CAMERA_MOVE" in doc.downlink_queue:
                # print("skip long render:", logger.since('long-render'))
                continue

            async for render_response in render(
                pipeline,
                # last_event.client_id,
                camera,
                # move model to a model cache, reload.
                # model,
                channels=channels,
                position=world["position"],
                rotation=world["rotation"],
                scale=world["scale"],
                # snapshot=snapshot,
                **value,
            ):
                if isinstance(render_response, ServerEvent):
                    doc @ render_response

                await sleep(0.0001)
                if "CAMERA_MOVE" in doc.downlink_queue:
                    # print("terminate long render:", logger.since('long-render'))
                    break
        elif event == "RESET_PCA_MAP":
            viewer_utils.reset_pca_proj()

    cprint("this now dies", "red")
