"""Three-view adapter for the RoboMME challenge interface.

The legacy ``challenge_interface.policy`` adapter is intentionally unchanged
for two-view checkpoints.  Use this adapter with ``deploy_omega.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import override

from mme_vla_omega_suite.policies.omega_policy import OmegaPolicy


def pack_state(joint_state: np.ndarray, gripper_state: np.ndarray) -> np.ndarray:
    return np.concatenate([joint_state, gripper_state[:1]], axis=0, dtype=np.float32)


def _required_list(inputs: dict[str, Any], names: tuple[str, ...], camera: str) -> list[np.ndarray]:
    for name in names:
        if name in inputs:
            values = inputs[name]
            if not isinstance(values, list | tuple):
                raise TypeError(f"{name} for {camera} must be a list of images")
            return [np.asarray(value) for value in values]
    raise KeyError(f"Missing {camera} history; expected one of {names}")


class MyOmegaPolicy_for_CVPR_Challenge:  # noqa: N801 - challenge loader imports this public name
    """Wrap :class:`OmegaPolicy` for the three-view challenge payload."""

    def __init__(self, model: OmegaPolicy, **_: Any):
        self._inner_policy = model
        self.chunk_size = 16

    @override
    def infer(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        left = _required_list(inputs, ("front_rgb_list", "agentview_left_list"), "agentview_left")
        right = _required_list(
            inputs,
            ("agentview_right_list", "right_rgb_list", "right_image_list"),
            "agentview_right",
        )
        eye = _required_list(inputs, ("wrist_rgb_list", "eye_in_hand_list"), "eye_in_hand")
        if not (len(left) == len(right) == len(eye)):
            raise ValueError(
                "Three-view challenge histories must have equal lengths: "
                f"left={len(left)}, right={len(right)}, eye_in_hand={len(eye)}"
            )

        joint_states = inputs["joint_state_list"]
        gripper_states = inputs["gripper_state_list"]
        if len(joint_states) != len(left) or len(gripper_states) != len(left):
            raise ValueError("State and image histories must have equal lengths")
        states = [pack_state(joint, gripper) for joint, gripper in zip(joint_states, gripper_states, strict=True)]

        # Challenge payloads conventionally include the current frame in each
        # list.  It is passed to infer below, so only earlier observations are
        # added to the strict-past memory bank.
        if len(left) > 1:
            self._inner_policy.add_buffer(
                {
                    "omega_images": {
                        "agentview_left": np.stack(left[:-1], axis=0).astype(np.uint8),
                        "agentview_right": np.stack(right[:-1], axis=0).astype(np.uint8),
                        "eye_in_hand": np.stack(eye[:-1], axis=0).astype(np.uint8),
                    }
                }
            )

        task_goal = inputs["task_goal"]
        prompt = task_goal[0] if isinstance(task_goal, list | tuple) else task_goal
        element = {
            "observation/image": left[-1],
            "observation/right_image": right[-1],
            "observation/wrist_image": eye[-1],
            "observation/state": states[-1],
            "prompt": str(prompt).lower(),
        }
        if inputs.get("simple_subgoal") is not None:
            element["simple_subgoal"] = inputs["simple_subgoal"]
            element["grounded_subgoal"] = inputs.get("grounded_subgoal", inputs["simple_subgoal"])
        outputs = self._inner_policy.infer(element)
        return {"actions": outputs["actions"][: self.chunk_size, :]}

    @override
    def reset(self) -> None:
        self._inner_policy.reset()
