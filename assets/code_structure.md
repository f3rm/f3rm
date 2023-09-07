# F3RM Code Structure

- `f3rm/features`
    - Code for extracting CLIP and DINO features from images.
    - We interpolate the position encoding for both the CLIP and DINO ViT, so the patch-level features maintain the same
      aspect ratio as the input images.
    - See `clip_extract.py` and `dino_extract.py` for the main entrypoints.
- `f3rm/scripts`
- `f3rm_config.py`
    - Nerfstudio method configuration for F3RM. This allows `f3rm` to show up as a method in `ns-train`.
- `feature_datamanager.py`
    - Datamanager for extracting features from images online, and interpolating them when we sample rays.
- `feature_field.py`
    - [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) definition of the feature field, which is a multiresolution
      hash encoding with a MLP head.
- `model.py`
    - Nerfstudio model class for a feature field. You can specify the hyperparameters for the feature field here.
    - The implementation for the GUI controls, including the language-based querying, is provided here.
- `pca_colormap.py`
    - Computing the PCA colormap of the high-dimensional features, so we can visualize them as an RGB image.
- `renderer.py`
    - Simple weighted sum for rendering features along a ray.
