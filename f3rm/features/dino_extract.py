import gc
from typing import List

import torch
from einops import rearrange
from tqdm import tqdm


class DINOArgs:
    model_type: str = "dino_vits8"
    load_size: int = 224
    stride: int = 4
    facet: str = "key"
    layer: int = 11
    bin: bool = False
    batch_size: int = 4


_supported_dino_models = {"dino_vits8", "dino_vits16", "dino_vitb8", "dino_vitb16"}


@torch.no_grad()
def extract_dino_features(image_paths: List[str], device: torch.device) -> torch.Tensor:
    from f3rm.features.dino.dino_vit_extractor import ViTExtractor

    assert (
        DINOArgs.model_type in _supported_dino_models
    ), f"Model type must be one of {_supported_dino_models}, not {DINOArgs.model_type}"

    extractor = ViTExtractor(DINOArgs.model_type, DINOArgs.stride, device=device)
    print(f"Loaded DINO model {DINOArgs.model_type}")

    # Preprocess images
    preprocessed_images = [
        extractor.preprocess(image_path, DINOArgs.load_size)[0]
        for image_path in image_paths
    ]
    preprocessed_images = torch.cat(preprocessed_images, dim=0).to(device)
    print(
        f"Preprocessed {len(image_paths)} images to shape {preprocessed_images.shape}"
    )

    # Extract DINO features in batches
    embeddings = []
    for i in tqdm(
        range(0, len(preprocessed_images), DINOArgs.batch_size),
        desc="Extracting DINO features",
    ):
        batch = preprocessed_images[i : i + DINOArgs.batch_size]
        embeddings.append(
            extractor.extract_descriptors(
                batch, DINOArgs.layer, DINOArgs.facet, DINOArgs.bin
            )
        )
    embeddings = torch.cat(embeddings, dim=0)

    # Reshape embeddings to have shape (batch, height, width, channels))
    height, width = extractor.num_patches
    embeddings = rearrange(embeddings, "b 1 (h w) c -> b h w c", h=height, w=width)
    print(f"Extracted DINO embeddings of shape {embeddings.shape}")

    # Delete and clear memory to be safe
    del extractor
    del preprocessed_images
    torch.cuda.empty_cache()
    gc.collect()

    return embeddings


if __name__ == "__main__":
    extract_dino_features(
        [
            f"/home/william/workspace/vqn/datasets/instant-feature/datasets/panda/open_ended/multi_lang/demo/trial_002/scene_00002/images/frame_{i + 1:05d}.png"
            for i in range(50)
        ],
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
    )
