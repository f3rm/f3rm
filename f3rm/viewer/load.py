from pathlib import Path

from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils.eval_utils import eval_setup


def load_nerfstudio_pipeline(load_dir: str) -> Pipeline:
    load_config = Path(load_dir) / "config.yml"
    print(f"Loading Nerfstudio pipeline from {load_config}")
    _, pipeline, _, _ = eval_setup(
        load_config,
        eval_num_rays_per_chunk=None,
        test_mode="inference",
    )
    return pipeline


# loaded_state = torch.load(load_path, map_location="cpu")
# # load the checkpoints for pipeline, optimizers, and gradient scalar
# self.pipeline.load_pipeline(loaded_state["pipeline"], loaded_state["step"])
# self.optimizers.load_optimizers(loaded_state["optimizers"])
# if "schedulers" in loaded_state and self.config.load_scheduler:
#     self.optimizers.load_schedulers(loaded_state["schedulers"])
# self.grad_scaler.load_state_dict(loaded_state["scalers"])
# CONSOLE.print(f"Done loading Nerfstudio checkpoint from {load_path}")


if __name__ == "__main__":
    pipeline = load_nerfstudio_pipeline(
        "/home/william/workspace/vqn/f3rm-public-release/outputs/stata_office/f3rm/2023-09-14_164333"
    )

    model = pipeline.model
