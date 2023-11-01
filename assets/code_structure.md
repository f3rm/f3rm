# F3RM Code Structure

The F3RM codebase is split into two modules:

1. [`f3rm`](#f3rm): the feature field implementation and Nerfstudio integration.
2. [`f3rm_robot`](#f3rm_robot): the language-guided pose optimization code.

## [`f3rm`](/f3rm)

- [`features/`](/f3rm/features)
    - Code for extracting CLIP and DINO features from images.
    - We interpolate the position encoding for both the CLIP and DINO ViT, so the patch-level features maintain the same
      aspect ratio as the input images.
    - See [`clip_extract.py`](/f3rm/features/clip_extract.py) and [`dino_extract.py`](/f3rm/features/dino_extract.py)
      for the main entrypoints.
- [`scripts/`](/f3rm/scripts)
    - Demo scripts for extracting CLIP and DINO features ([`demo_clip_features.py`](/f3rm/scripts/demo_clip_features.py)
      and [`demo_extract_features.py`](/f3rm/scripts/demo_extract_features.py)).
    - Script for downloading example datasets ([`download_datasets.py`](/f3rm/scripts/download_datasets.py)).
- [`f3rm_config.py`](/f3rm/f3rm_config.py)
    - Nerfstudio method configuration for F3RM. This allows `f3rm` to show up as a method in `ns-train`.
- [`feature_datamanager.py`](/f3rm/feature_datamanager.py)
    - Datamanager for extracting features from images online, and interpolating them when we sample rays.
- [`feature_field.py`](/f3rm/feature_field.py)
    - [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) definition of the feature field, which is a multiresolution
      hash encoding with a MLP head.
- [`model.py`](/f3rm/model.py)
    - Nerfstudio model class for a feature field. You can specify the hyperparameters for the feature field here.
    - The implementation for the GUI controls, including the language-based querying, is provided here.
- [`pca_colormap.py`](/f3rm/pca_colormap.py)
    - Computing the PCA colormap of the high-dimensional features, so we can visualize them as an RGB image.
- [`renderer.py`](/f3rm/renderer.py)
    - Simple weighted sum for rendering features along a ray.

## [`f3rm_robot`](/f3rm_robot)

- [`assets`](/f3rm_robot/assets)
    - Contains the Panda gripper mesh used for collision checking and the task embeddings.
- [`examples`](/f3rm_robot/examples)
    - Contains an example script for generating task embeddings.
- [`args.py`](/f3rm_robot/args.py)
    - Arguments for language-guided pose optimization.
- [`collision.py`](/f3rm_robot/collision.py)
    - Collision checking grasps against a NeRF and associated arguments.
- [`field_adapter.py`](/f3rm_robot/field_adapter.py)
    - Adapter for a feature field to support querying by `(x, y, z)` coordinates in the world frame.
- [`initial_proposals.py`](/f3rm_robot/initial_proposals.py)
    - Utilities for generating initial proposals for the pose optimization, including marching cubes masking.
- [`load.py`](/f3rm_robot/load.py)
    - Utilities for loading trained feature fields using Nerfstudio.
- [`optimize.py`](/f3rm_robot/optimize.py)
    - Main entrypoint for language-guided pose optimization.
    - Contains the code for retrieving relevant demonstrations, initial proposal generation, and the optimization loop.
    - See `f3rm-optimize --help` for usage.
- [`task.py`](/f3rm_robot/task.py)
    - Model class for a task and loading the task embeddings
- [`utils.py`](/f3rm_robot/utils.py)
    - Utilities for generating heatmaps, gripper meshes, and sampling point clouds.
- [`visualizer.py`](/f3rm_robot/visualizer.py)
    - Wrapper for visualizer. We provide a [Viser](https://github.com/nerfstudio-project/viser)-based visualizer.

**Note:** type hints for PyTorch tensors are provided on a best effort basis. They may be incorrect. 