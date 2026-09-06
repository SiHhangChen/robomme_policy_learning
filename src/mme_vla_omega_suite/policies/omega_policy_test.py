import numpy as np
import pytest

from mme_vla_omega_suite.policies.omega_policy import OmegaPolicy


def test_extract_camera_images_accepts_canonical_nested_three_views():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    images = {
        "agentview_left": image,
        "agentview_right": image,
        "eye_in_hand": image,
    }
    extracted = OmegaPolicy._extract_camera_images(  # noqa: SLF001
        {"images": images}, ("agentview_left", "agentview_right", "eye_in_hand")
    )
    assert tuple(extracted) == ("agentview_left", "agentview_right", "eye_in_hand")
    assert all(value.shape == (4, 4, 3) for value in extracted.values())


def test_extract_camera_images_rejects_two_view_input():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="agentview_right"):
        OmegaPolicy._extract_camera_images(  # noqa: SLF001
            {"observation/image": image, "observation/wrist_image": image},
            ("agentview_left", "agentview_right", "eye_in_hand"),
        )
