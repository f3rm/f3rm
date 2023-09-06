import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from f3rm.features.clip_extract import extract_clip_features
from f3rm.features.dino_extract import extract_dino_features
from f3rm.pca_colormap import apply_pca_colormap

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGE_DIR = os.path.join(_MODULE_DIR, "images")

image_paths = [os.path.join(_IMAGE_DIR, name) for name in ["frame_1.png", "frame_2.png", "frame_3.png"]]


@torch.no_grad()
def demo_extract_features():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    clip_embeddings = extract_clip_features(image_paths, device)
    # Convert to float as torch PCA doesn't support half on GPU
    clip_embeddings = clip_embeddings.float()
    clip_pca = apply_pca_colormap(clip_embeddings).cpu().numpy()

    dino_embeddings = extract_dino_features(image_paths, device)
    dino_pca = apply_pca_colormap(dino_embeddings).cpu().numpy()

    # Visualize the embeddings
    plt.figure()
    plt.suptitle("CLIP (2nd row) and DINO (3rd row) Features PCA")
    for i, (image_path, clip_pca_, dino_pca_) in enumerate(zip(image_paths, clip_pca, dino_pca)):
        plt.subplot(3, len(image_paths), i + 1)
        plt.imshow(Image.open(image_path))
        plt.title(os.path.basename(image_path))
        plt.axis("off")

        plt.subplot(3, len(image_paths), len(image_paths) + i + 1)
        plt.imshow(clip_pca_)
        plt.axis("off")

        plt.subplot(3, len(image_paths), 2 * len(image_paths) + i + 1)
        plt.imshow(dino_pca_)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("demo_extract_features.png")
    print("Saved plot to demo_extract_features.png")
    plt.show()


if __name__ == "__main__":
    demo_extract_features()
