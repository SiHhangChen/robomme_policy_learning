"""Dataset transforms for the isolated Omega suite."""

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        if np.abs(image).max() > 1.0:
            raise ValueError("Floating-point images must be normalized to [0, 1]")
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class OmegaRoboMMEInputs(transforms.DataTransformFn):
    """Convert the three native MemBench camera views to Pi05 inputs."""

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        if "images" in data:
            image_dict = data["images"]
            base_image = _parse_image(image_dict["agentview_right"])
            right_image = _parse_image(image_dict["agentview_left"])
            wrist_image = _parse_image(image_dict["eye_in_hand"])
        else:
            base_image = _parse_image(data["image"] if "image" in data else data["observation/image"])
            right_image = _parse_image(
                data["right_image"] if "right_image" in data else data["observation/right_image"]
            )
            wrist_image = _parse_image(
                data["wrist_image"] if "wrist_image" in data else data["observation/wrist_image"]
            )
        state = data["state"] if "state" in data else data["observation/state"]
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "right_wrist_0_rgb": right_image,
                "left_wrist_0_rgb": wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
            },
            "omega_memory": data["omega_memory"],
            "omega_memory_mask": data["omega_memory_mask"],
        }
        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        # The shared Pi05 tokenizer removes these fields when symbolic memory
        # is disabled, so keep the RoboMME keys in the transformed sample.
        inputs["simple_subgoal"] = data.get("simple_subgoal")
        inputs["grounded_subgoal"] = data.get("grounded_subgoal")
        return inputs


@dataclasses.dataclass(frozen=True)
class OmegaRoboMMEOutputs(transforms.DataTransformFn):
    """Return the native 13-dimensional MemBench hybrid action."""

    action_dim: int = 13

    def __call__(self, data: dict) -> dict:
        if "actions" not in data:
            return data
        return {"actions": np.asarray(data["actions"])[:, : self.action_dim]}


@dataclasses.dataclass(frozen=True)
class OmegaSliceState(transforms.DataTransformFn):
    """Keep the 30-dim state used by the pretrained MemBench Pi05 baseline."""

    keep_dim: int = 30

    def __call__(self, data: dict) -> dict:
        if "state" in data:
            data = {**data, "state": np.asarray(data["state"])[..., : self.keep_dim]}
        return data


__all__ = ["OmegaRoboMMEInputs", "OmegaRoboMMEOutputs", "OmegaSliceState"]
