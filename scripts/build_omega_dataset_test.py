from __future__ import annotations

import pickle

import h5py
import numpy as np
import pytest

from scripts.build_omega_dataset import build_dataset


def _make_episode(path, *, right_path="agentview_right_rgb"):
    with h5py.File(path, "w") as data:
        episode = data.create_group("episode_0")
        setup = episode.create_group("setup")
        setup.create_dataset("task_goal", data=np.asarray([b"pick"], dtype="S10"))
        for step in range(2):
            timestep = episode.create_group(f"timestep_{step}")
            obs = timestep.create_group("obs")
            obs.create_dataset("front_rgb", data=np.full((4, 4, 3), step, dtype=np.uint8))
            obs.create_dataset("wrist_rgb", data=np.full((4, 4, 3), step + 20, dtype=np.uint8))
            obs.create_dataset(right_path, data=np.full((4, 4, 3), step + 10, dtype=np.uint8))
            obs.create_dataset("joint_state", data=np.zeros(7, dtype=np.float32))
            obs.create_dataset("gripper_state", data=np.zeros(1, dtype=np.float32))
            action = timestep.create_group("action")
            action.create_dataset("joint_action", data=np.zeros(8, dtype=np.float32))
            info = timestep.create_group("info")
            info.create_dataset("is_video_demo", data=np.zeros((), dtype=np.bool_))
            info.create_dataset("is_completed", data=np.zeros((), dtype=np.bool_))
            info.create_dataset("is_subgoal_boundary", data=np.zeros((), dtype=np.bool_))
            for key in (
                "simple_subgoal",
                "grounded_subgoal",
                "simple_subgoal_online",
                "grounded_subgoal_online",
            ):
                info.create_dataset(key, data=np.asarray([b"pick"], dtype="S10"))


def test_build_dataset_injects_real_right_view(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_episode(raw / "sample.h5")
    output = tmp_path / "processed"

    class TinyProcessor:
        def __init__(self, *, preprocessed_data_path, **_):
            self.path = preprocessed_data_path

        def run(self):
            data = self.path + "/data"
            import pathlib

            pathlib.Path(data).mkdir(parents=True)
            sample = {
                "epis_idx": np.asarray([0]),
                "step_idx": np.asarray([1]),
            }
            with open(pathlib.Path(data) / "0.pkl", "wb") as handle:
                pickle.dump(sample, handle)
            pathlib.Path(self.path + "/meta").mkdir(parents=True)

    monkeypatch.setattr("scripts.build_omega_dataset.DatasetProcessor", TinyProcessor)
    build_dataset(raw, output)
    with (output / "data" / "0.pkl").open("rb") as handle:
        sample = pickle.load(handle)
    np.testing.assert_array_equal(sample["right_image"], np.full((4, 4, 3), 11, dtype=np.uint8))


def test_build_dataset_rejects_missing_right_view(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    _make_episode(raw / "sample.h5", right_path="other_rgb")
    with pytest.raises(KeyError, match="Missing right camera"):
        build_dataset(raw, tmp_path / "processed")
