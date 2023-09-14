from asyncio import sleep
from collections import deque
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
from matplotlib.colors import to_rgb
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.model_components.renderers import background_color_override_context
from nerfstudio.pipelines.base_pipeline import Pipeline
from vuer import Vuer
from vuer.events import ServerEvent
from vuer.schemas import Scene, group

from f3rm.viewer.load import load_nerfstudio_pipeline
from f3rm.viewer.utils import b64jpg, get_camera_from_three, rotation_matrix

doc = Vuer(
    uri="ws://localhost:8012",
    domain="https://dash.ml/demos/vqn-dash/three",
    queries=dict(
        reconnect=True,
        debug=True,
    ),
    cors="*",
)

Vector3 = Tuple[float, float, float]


async def render(
    pipeline: Pipeline,
    camera: Dict,
    use_aabb: bool,
    aabb_min: Optional[Vector3],
    aabb_max: Optional[Vector3],
    alpha_threshold: Optional[float],
    rotation=None,
    bg_color: str = "#000000",
    **__,
):
    # Get camera poses based from three.js camera
    height = int(camera["height"])
    width = int(camera["aspect"] * height)
    camera = get_camera_from_three([camera], width, height).to(pipeline.device)

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
    rgb = to_rgb(bg_color)
    bg_rgb = torch.Tensor(rgb).to(pipeline.device)

    # Generate rays and apply rotation and translation
    camera_ray_bundle = camera.generate_rays(camera_indices=0, aabb_box=aabb_box)
    if rotation is not None:
        mat = rotation_matrix(*rotation)
        c2w = torch.FloatTensor(mat).to(pipeline.device)
        camera_ray_bundle.origins @= c2w
        camera_ray_bundle.directions @= c2w

    with background_color_override_context(bg_rgb), torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

    rgb_np = outputs["rgb"].cpu().numpy()
    # Use accumulation as proxy for alpha
    alpha = outputs["accumulation"]
    if alpha_threshold is not None:
        alpha[alpha <= alpha_threshold] = 0
    alpha_np = alpha.cpu().numpy()

    rgb64 = b64jpg(rgb_np)
    alpha64 = b64jpg(alpha_np.squeeze())

    return ServerEvent(
        etype="RENDER",
        # client_id=client_id,
        data={
            "rgb": rgb64,
            "alpha": alpha64,
            "camera": camera,
        },
    )


@doc.spawn(start=True)
async def run_eval(ws_id):
    pipeline = load_nerfstudio_pipeline(
        "/home/william/workspace/vqn/f3rm-public-release/outputs/stata_office/f3rm/2023-09-14_164333"
    )
    pipeline.eval()
    print("Open viewer: https://dash.ml/demos/vqn-dash/three/?ws=ws://localhost:8012")

    scene = Scene(
        group(key="playground"),
        # set camera position
        key="scene",
    )

    doc.set @ scene

    history = deque(maxlen=10)

    while True:
        event = doc.pop()
        if event:
            history.append(event)
            print("\r", len(doc.downlink_queue))

        if event == "CAMERA_MOVE":
            last_event = history[-1]
            value = deepcopy(last_event.value)
            doc.clear()
            print(value)

            world = value.pop("world")
            camera = value.pop("camera")
            channels = value.pop("channels", "rgb")
            # progressive = value.pop("progressive", None)
            # print(progressive)
            progressive = 0.25
            if progressive:
                camera = deepcopy(camera)
                camera["height"] = progressive * camera["height"]
                camera["width"] = camera["aspect"] * camera["height"]
            # snapshot = event.etype == "SNAPSHOT"

            m = await render(
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
            )
            doc @ m

        await sleep(0)

    cprint("this now dies", "red")
