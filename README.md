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

## Installation
**Note:** this project requires a NVIDIA GPU with CUDA 11.8+

```bash
# Clone the repo
git clone https://github.com/f3rm/f3rm.git
cd f3rm

# Create conda environment. Feel free to use a different package manager
conda create -n f3rm python=3.10
conda activate f3rm

# Install Nerfstudio per instructions here: 
# https://docs.nerf.studio/en/latest/quickstart/installation.html#dependencies
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install ninja git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
pip install nerfstudio

# Install F3RM project and dependencies
pip install -e .

# Test your installation
# TODO: add test
```

## Code coming soon
Check back for updates.

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
