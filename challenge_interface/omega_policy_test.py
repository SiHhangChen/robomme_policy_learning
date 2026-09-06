from __future__ import annotations

import numpy as np
import pytest

from challenge_interface.omega_policy import MyOmegaPolicy_for_CVPR_Challenge


class _FakePolicy:
    def __init__(self):
        self.buffers = []
        self.infers = []

    def add_buffer(self, payload):
        self.buffers.append(payload)

    def infer(self, payload):
        self.infers.append(payload)
        return {"actions": np.zeros((20, 8), dtype=np.float32)}

    def reset(self):
        pass


def _inputs(length=2):
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    return {
        "front_rgb_list": [image] * length,
        "agentview_right_list": [image + 1] * length,
        "wrist_rgb_list": [image + 2] * length,
        "joint_state_list": [np.zeros(7, dtype=np.float32)] * length,
        "gripper_state_list": [np.zeros(1, dtype=np.float32)] * length,
        "task_goal": ["pick"],
    }


def test_adapter_adds_only_history_before_current():
    fake = _FakePolicy()
    output = MyOmegaPolicy_for_CVPR_Challenge(fake).infer(_inputs())
    assert output["actions"].shape == (16, 8)
    assert fake.buffers[0]["omega_images"]["agentview_right"].shape[0] == 1
    assert fake.infers[0]["observation/right_image"].shape == (4, 4, 3)


def test_adapter_requires_right_view():
    values = _inputs()
    values.pop("agentview_right_list")
    with pytest.raises(KeyError, match="agentview_right"):
        MyOmegaPolicy_for_CVPR_Challenge(_FakePolicy()).infer(values)
