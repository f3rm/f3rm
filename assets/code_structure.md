# F3RM Code Structure

- [`f3rm/features`](/f3rm/features)
    - Code for extracting CLIP and DINO features from images.
    - We interpolate the position encoding for both the CLIP and DINO ViT, so the patch-level features maintain the same
      aspect ratio as the input images.
    - See [`clip_extract.py`](/f3rm/features/clip_extract.py) and [`dino_extract.py`](/f3rm/features/dino_extract.py)
      for the main entrypoints.
- [`f3rm/scripts`](/f3rm/scripts)
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
