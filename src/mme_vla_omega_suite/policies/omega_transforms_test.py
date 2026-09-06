import numpy as np
import pytest

from mme_vla_omega_suite.policies.omega_transforms import OmegaRoboMMEInputs
from openpi.models import model as _model


def test_omega_inputs_keep_shared_symbolic_memory_placeholders():
    data = {
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "right_image": np.full((4, 4, 3), 2, dtype=np.uint8),
        "wrist_image": np.ones((4, 4, 3), dtype=np.uint8),
        "state": np.zeros((8,), dtype=np.float32),
        "omega_memory": np.zeros((2, 3, 4, 17, 2048), dtype=np.float16),
        "omega_memory_mask": np.ones((2,), dtype=np.bool_),
        "simple_subgoal": None,
        "grounded_subgoal": None,
        "prompt": "task",
    }
    transformed = OmegaRoboMMEInputs(model_type=_model.ModelType.PI05)(data)
    assert transformed["simple_subgoal"] is None
    assert transformed["grounded_subgoal"] is None
    assert transformed["omega_memory"].shape == (2, 3, 4, 17, 2048)
    assert tuple(transformed["image"]) == ("base_0_rgb", "right_wrist_0_rgb", "left_wrist_0_rgb")
    assert all(transformed["image_mask"].values())


def test_omega_inputs_require_the_third_camera():
    data = {
        "image": np.zeros((4, 4, 3), dtype=np.uint8),
        "wrist_image": np.ones((4, 4, 3), dtype=np.uint8),
        "state": np.zeros((8,), dtype=np.float32),
        "omega_memory": np.zeros((2, 3, 4, 17, 2048), dtype=np.float16),
        "omega_memory_mask": np.ones((2,), dtype=np.bool_),
    }
    with pytest.raises(KeyError, match="right_image"):
        OmegaRoboMMEInputs(model_type=_model.ModelType.PI05)(data)


def test_omega_inputs_accept_online_observation_keys():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    data = {
        "observation/image": image,
        "observation/right_image": image,
        "observation/wrist_image": image,
        "observation/state": np.zeros((8,), dtype=np.float32),
        "omega_memory": np.zeros((2, 3, 4, 17, 2048), dtype=np.float16),
        "omega_memory_mask": np.ones((2,), dtype=np.bool_),
    }
    transformed = OmegaRoboMMEInputs(model_type=_model.ModelType.PI05)(data)
    assert transformed["state"].shape == (8,)
