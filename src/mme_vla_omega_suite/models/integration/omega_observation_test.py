import numpy as np

from mme_vla_omega_suite.models.integration.omega_observation import OmegaObservation


def test_omega_observation_accepts_image_and_memory_axes_independently():
    observation = OmegaObservation.from_dict(
        {
            "image": {"base_0_rgb": np.zeros((1, 224, 224, 3), dtype=np.float32)},
            "image_mask": {"base_0_rgb": np.ones((1,), dtype=np.bool_)},
            "state": np.zeros((1, 8), dtype=np.float32),
            "omega_memory": np.zeros((1, 2, 3, 4, 17, 2048), dtype=np.float32),
            "omega_memory_mask": np.ones((1, 2), dtype=np.bool_),
        }
    )

    assert observation.omega_memory.shape == (1, 2, 3, 4, 17, 2048)
    assert observation.omega_memory_mask.shape == (1, 2)
