import os

import matplotlib.pyplot as plt
import torch
from PIL import Image

from f3rm.features import clip
from f3rm.features.clip import tokenize
from f3rm.features.clip_extract import CLIPArgs, extract_clip_features

_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_IMAGE_DIR = os.path.join(_MODULE_DIR, "images")

image_paths = [os.path.join(_IMAGE_DIR, name) for name in ["frame_1.png", "frame_2.png", "frame_3.png"]]


@torch.no_grad()
def demo_clip_features(text_query: str) -> None:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Extract the patch-level features for the images
    clip_embs = extract_clip_features(image_paths, device)
    clip_embs /= clip_embs.norm(dim=-1, keepdim=True)

    # Load the CLIP model so we can get text embeddings
    model, _ = clip.load(CLIPArgs.model_name, device=device)

    # Encode text query
    tokens = tokenize(text_query).to(device)
    text_embs = model.encode_text(tokens)
    text_embs /= text_embs.norm(dim=-1, keepdim=True)

    # Compute similarities
    sims = clip_embs @ text_embs.T
    sims = sims.squeeze()

    # Visualize
    plt.figure()
    cmap = plt.get_cmap("turbo")
    for idx, (image_path, sim) in enumerate(zip(image_paths, sims)):
        plt.subplot(2, len(image_paths), idx + 1)
        plt.imshow(Image.open(image_path))
        plt.title(os.path.basename(image_path))
        plt.axis("off")

        plt.subplot(2, len(image_paths), len(image_paths) + idx + 1)
        sim_norm = (sim - sim.min()) / (sim.max() - sim.min())
        heatmap = cmap(sim_norm.cpu().numpy())
        plt.imshow(heatmap)
        plt.axis("off")

    plt.tight_layout()
    plt.suptitle(f'Similarity to language query "{text_query}"')

    text_label = text_query.replace(" ", "-")
    plt_fname = f"demo_clip_features_{text_label}.png"
    plt.savefig(plt_fname)
    print(f"Saved plot to {plt_fname}")
    plt.show()


if __name__ == "__main__":
    demo_clip_features(text_query="teddy bear")
