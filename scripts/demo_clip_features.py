import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from f3rm.features import clip
from f3rm.features.clip import tokenize
from f3rm.features.clip_extract import CLIPArgs, extract_clip_features
from f3rm.pca_colormap import apply_pca_colormap

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGE_DIR = os.path.join(_MODULE_DIR, "images")

image_paths = [
    os.path.join(_IMAGE_DIR, name)
    for name in ["frame_1.png", "frame_2.png", "frame_3.png"]
]
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@torch.no_grad()
def demo_clip_features():
    # Extract the patch-level features for the images
    clip_embeddings = extract_clip_features(image_paths, device)

    # Load the CLIP model so we can get text embeddings
    model, _ = clip.load(CLIPArgs.model_name, device=device)

    # Encode text queries
    text_queries = ["teddy bear", "mug", "scissors"]
    tokens = tokenize(text_queries).to(device)
    text_embeddings = model.encode_text(tokens)

    # Normalize embeddings
    clip_embeddings /= clip_embeddings.norm(dim=-1, keepdim=True)
    text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)

    # Compute similarities
    all_sims = clip_embeddings @ text_embeddings.T
    all_sims = all_sims.float()

    # Visualize
    plt.figure()
    cmap = plt.get_cmap("turbo")

    for idx, sims in enumerate(all_sims):
        plt.subplot(len(image_paths), len(text_queries) + 1, idx)
        plt.imshow(Image.open(image_paths[idx]))

        sims = sims.permute(2, 0, 1).cpu().numpy()
        for query, sim in zip(text_queries, sims):
            # Normalize similarity
            sim = (sim - sim.min()) / (sim.max() - sim.min())
            heatmap = cmap(sim)
            # heatmap = Image.fromarray((heatmap * 255).astype(np.uint8))

            plt.subplot()


if __name__ == "__main__":
    demo_clip_features()
