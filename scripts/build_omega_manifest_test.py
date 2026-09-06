import json

import h5py
import numpy as np
from PIL import Image
import pytest

from scripts.build_omega_manifest import build_manifest


def _write_episode(path):
    with h5py.File(path, "w") as data:
        episode = data.create_group("episode_0")
        for step in range(2):
            timestep = episode.create_group(f"timestep_{step}")
            obs = timestep.create_group("obs")
            obs.create_dataset("front_rgb", data=np.full((4, 5, 3), step, dtype=np.uint8))
            obs.create_dataset("agentview_right_rgb", data=np.full((4, 5, 3), step + 10, dtype=np.uint8))
            obs.create_dataset("wrist_rgb", data=np.full((4, 5, 3), step + 20, dtype=np.uint8))


def test_build_manifest_exports_real_three_view_frames(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _write_episode(raw / "demo.h5")

    manifest = build_manifest(raw, tmp_path / "omega", decision_stride=1)
    records = [json.loads(line) for line in manifest.read_text().splitlines()]
    assert len(records) == 2
    assert tuple(records[0]["images"]) == ("agentview_left", "agentview_right", "eye_in_hand")
    assert all((tmp_path / "omega" / "images").exists() for _ in [0])
    with Image.open(records[0]["images"]["agentview_right"]) as image:
        assert np.asarray(image).mean() == 10


def test_build_manifest_rejects_missing_right_camera(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    with h5py.File(raw / "demo.h5", "w") as data:
        timestep = data.create_group("episode_0").create_group("timestep_0")
        obs = timestep.create_group("obs")
        obs.create_dataset("front_rgb", data=np.zeros((2, 2, 3), dtype=np.uint8))
        obs.create_dataset("wrist_rgb", data=np.zeros((2, 2, 3), dtype=np.uint8))

    with pytest.raises(KeyError, match="agentview_right"):
        build_manifest(raw, tmp_path / "omega", decision_stride=1)
