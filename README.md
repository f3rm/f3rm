# Distilled Feature Fields Enable Few-Shot Language-Guided Manipulation

### [🌐 Project Website](https://f3rm.github.io) | [📝 Paper](https://arxiv.org/abs/2308.07931) | [🎥 Video](https://www.youtube.com/watch?v=PA9rWWVWsc4)

We distill features from 2D foundation models into 3D feature fields, and enable few-shot language-guided manipulation
that generalizes across object poses, shapes, appearances and categories.

**[William Shen](https://shen.nz)<sup>\*1</sup>, [Ge Yang](https://www.episodeyang.com/)<sup>\*2</sup>,
[Alan Yu](https://www.linkedin.com/in/alan-yu1/)<sup>1</sup>,
[Jansen Wong](https://www.linkedin.com/in/jansenwong/)<sup>1</sup>,
[Leslie Kaelbling](https://people.csail.mit.edu/lpk/)<sup>1</sup>,
[Phillip Isola](https://people.csail.mit.edu/phillipi/)<sup>1</sup>**<br>
<sup>1 </sup>[MIT CSAIL](https://www.csail.mit.edu/),
<sup>2 </sup>[Institute of AI and Fundamental Interactions (IAIFI)](https://iaifi.org/)<br>
<sup>* </sup>Indicates equal contribution

## Code

We currently release our implementation for training feature fields. We will release our custom viewer and the robot
manipulation code soon.

F3RM is built on top of [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) following their
[guide for adding new methods](https://docs.nerf.studio/en/latest/developer_guides/new_methods.html).

### Installation

**Note:** this repo requires an NVIDIA GPU with CUDA 11.7+ for NeRF and feature field distillation.

```bash
# Clone the repo
git clone https://github.com/f3rm/f3rm.git
cd f3rm

# Create conda environment. Feel free to use a different package manager
conda create -n f3rm python=3.10
conda activate f3rm

# Install torch per instructions here: https://pytorch.org/get-started/locally/
# Choose the CUDA version that matches your GPU
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install tiny-cuda-nn
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

# Install F3RM project and dependencies (including nerfstudio)
pip install -e .

# Install command line completions for nerfstudio
ns-install-cli

# Test your installation, and check 'f3rm' is a valid method
ns-train --help
```

## Usage

### Downloading Example Datasets

We provide example datasets which you can download using the `f3rm-download-data` command. By default, the script will
download all the datasets into the `datasets/` directory relative to your current directory.

Run `f3rm-download-data -h` to see how to change these options.

### Training a Feature Field

We provide the functionality to train a NeRF and distill features in parallel. The default features we distill are CLIP
features. You can distill DINO features by setting `--pipeline.datamanager.feature-type DINO`.

```bash
ns-train f3rm --data <data_folder>
```

You can try F3RM with the example datasets which you can download following the
instructions [here](#downloading-example-datasets). Alternatively, you can prepare your own datasets following the
instructions in the [Nerfstudio documentation](https://docs.nerf.studio/en/latest/quickstart/custom_dataset.html).

### Using our Custom Viewer

Coming soon...

### Using the Nerfstudio Viewer

Screenshot

How to use

If you distill CLIP features, these appear...

Please see [viewer.md](viewer.md) for more details.

### Extracting CLIP and DINO Features

We provide scripts to demonstrate how to extract CLIP and DINO features from their respective vision models. You can
use these features for your own NeRF pipeline or for other downstream applications.

- Run `python scripts/demo_extract_features.py` for a general demo on how to extract CLIP and DINO features.
  This will create a plot showing the PCA of the CLIP and DINO features. The plot is saved
  to `demo_extract_features.png`.
- For details on how to extract CLIP features and compare the extracted features with CLIP text embeddings, run
  `python scripts/demo_clip_features.py`. This script will create a plot showing the similarity heatmaps for a given
  text query, and will save a plot to `demo_clip_features-{text_query}.png`.

## Citation

If you find our work useful, please consider citing:

```
@article{shen2023F3RM,
    title={Distilled Feature Fields Enable Few-Shot Language-Guided Manipulation},
    author={Shen, William and Yang, Ge and Yu, Alan and Wong, Jansen and Kaelbling, Leslie Pack, and Isola, Phillip},
    journal={arXiv preprint:2308.07931},
    year={2023}
}
```
