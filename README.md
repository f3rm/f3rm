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
**The NeRF and feature distillation code will be released soon.**

We currently provide our implementation for extracting CLIP and DINO features. See `scripts/demo_extract_features.py`
for a demo.

### Installation
**Note:** this repo will eventually require an NVIDIA GPU with CUDA 11.7+ for NeRF and feature field distillation.

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

# Install F3RM project and dependencies
pip install -e .
```

### Usage
**Extracting CLIP and DINO Features**

Run `python scripts/demo_extract_features.py` for a demo on how to extract CLIP and DINO features.
This will create a plot showing the PCA of the CLIP and DINO features. The plot is saved to `demo_extract_features.png`.

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
