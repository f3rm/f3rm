import json
import os.path
import time
from asyncio import sleep
from datetime import datetime

import numpy as np
import open3d as o3d
from vuer import Vuer, VuerSession
from vuer.schemas import DefaultScene, Gripper, Movable, PointCloud

from f3rm_robot.load import load_nerfstudio_outputs
from f3rm_robot.optimize import get_scene_pcd

app = Vuer()

# Load point cloud
_valid_exts = {".xyz", ".xyzn", ".xyzrgb", ".pts", ".ply", ".pcd", "config.yml"}

while True:
    load_path = input("=> Enter path to point cloud or nerfstudio outputs: ").strip().rstrip("/")
    if not any(load_path.endswith(ext) for ext in _valid_exts):
        print(f"Invalid file extension! Must be one of: {_valid_exts}")
    elif not os.path.exists(load_path):
        print(f"{load_path} does not exist!")
    else:
        break

if load_path.endswith("config.yml"):
    load_state = load_nerfstudio_outputs(load_path)
    pcd = get_scene_pcd(load_state, num_points=100_000, voxel_size=0.005)
else:
    pcd = o3d.io.read_point_cloud(load_path)
    print(f"Loaded point cloud with {len(pcd.points)} points from {load_path}")

if pcd.is_empty():
    print("WARNING: Point cloud is empty!")


def save_gripper_states(gripper_states: dict, save_path: str):
    # Note: Three.js uses xyzw quaternion convention: https://threejs.org/docs/#api/en/math/Quaternion
    demo_dict = {
        "task": "my_task_name",
        "demo_labels": [f"demo_{i}" for i in range(len(gripper_states))],
        "demo_poses": [
            {"translation": state["position"], "quat_xyzw": state["quaternion"]} for state in gripper_states.values()
        ],
    }
    with open(save_path, "w") as f:
        json.dump(demo_dict, f, indent=2)


@app.spawn(start=True)
async def main(session: VuerSession):
    # Set point cloud in vuer
    session.set @ DefaultScene(
        PointCloud(
            key="scene",
            vertices=np.array(pcd.points),
            colors=np.array(pcd.colors),
            # size=0.005 + 0.002,
        ),
        up=[0, 0, 1],  # z-up
    )

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = f"{timestamp}_scene_demo.json"

    # Setup gripper states and add initial gripper
    gripper_states = {}

    def add_gripper():
        # Need to rotate so it matches the z-up convention
        session.upsert @ Movable(Gripper(key=f"gripper-{len(gripper_states)}", rotation=[np.pi / 2, np.pi / 2, 0]))

    add_gripper()
    metadata = {"handled_event_keys": set(), "last_save_time": None}

    @app.add_handler("OBJECT_MOVE")
    async def handler(event, _):
        # If the gripper has moved far enough from the origin, add a new gripper
        # Each event has a unique key which has a one-to-one mapping to one of the grippers
        position = event.value["position"]
        dist_from_origin = np.linalg.norm(position)
        if dist_from_origin > 0.10 and event.key not in metadata["handled_event_keys"]:
            add_gripper()
            metadata["handled_event_keys"].add(event.key)

        # Update the gripper state and save to disk
        gripper_states[event.key] = {
            "position": position,
            "quaternion": event.value["quaternion"],
        }
        save_gripper_states(gripper_states, save_path)
        metadata["last_save_time"] = time.perf_counter()

    print("=> Move the grippers to label the demos! Ctrl+C when done.")
    print(f"View scene at {app.get_url()}")
    print(f"Saving gripper states to {save_path}")

    while True:
        if metadata["last_save_time"] is not None:
            duration = time.perf_counter() - metadata["last_save_time"]
            print(f"\rTime since last gripper event: {duration:.2f}s", end="")
        await sleep(0.01)
