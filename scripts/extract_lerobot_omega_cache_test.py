from __future__ import annotations

import argparse

import numpy as np
import pytest

from mme_vla_omega_suite.training.dataset import LeRobotV3Source
import scripts.extract_lerobot_omega_cache as extractor


def test_decision_keys_are_episode_ordered_and_keep_raw_identity():
    source = object.__new__(LeRobotV3Source)
    source._episode_positions = {  # noqa: SLF001
        3: np.asarray([3, 4, 5], dtype=np.int64),
        1: np.asarray([0, 1, 2], dtype=np.int64),
    }
    source._frames = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)  # noqa: SLF001
    source._indices = np.arange(6, dtype=np.int64)  # noqa: SLF001

    np.testing.assert_array_equal(
        source.decision_keys(2),
        np.asarray([[1, 0, 0], [1, 2, 2], [3, 0, 3], [3, 2, 5]], dtype=np.int64),
    )


class _FakeSource:
    def __init__(self, root, length: int):
        self.root = root.resolve()
        self._length = length

    def __len__(self):
        return self._length


def test_initialize_and_validate_resumable_cache(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    output_dir = tmp_path / "cache"
    checkpoint = tmp_path / "omega.pt"
    source = _FakeSource(dataset_root, length=4)
    keys = np.asarray([[0, 0, 0], [0, 2, 2]], dtype=np.int64)
    args = argparse.Namespace(
        output_dir=output_dir,
        checkpoint=checkpoint,
        image_resolution=32,
        decision_stride_frames=2,
        tolerance_s=0.051,
    )
    monkeypatch.setattr(extractor, "CACHED_LAYER_INDICES", (0, 1))
    monkeypatch.setattr(extractor, "TOKENS_PER_VIEW", 2)
    monkeypatch.setattr(extractor, "TOKEN_DIM", 3)

    extractor.initialize_cache(source, args, keys)
    paths = extractor.validate_cache(source, args, keys)

    assert np.load(paths["tokens"], mmap_mode="r").shape == (2, 3, 2, 2, 3)
    np.testing.assert_array_equal(np.load(paths["keys"]), keys)
    np.testing.assert_array_equal(np.load(paths["completed"]), np.zeros(2, dtype=np.uint8))


def test_validate_cache_rejects_different_decision_keys(tmp_path, monkeypatch):
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    source = _FakeSource(dataset_root, length=2)
    args = argparse.Namespace(
        output_dir=tmp_path / "cache",
        checkpoint=tmp_path / "omega.pt",
        image_resolution=32,
        decision_stride_frames=1,
        tolerance_s=0.051,
    )
    keys = np.asarray([[0, 0, 0]], dtype=np.int64)
    monkeypatch.setattr(extractor, "CACHED_LAYER_INDICES", (0,))
    monkeypatch.setattr(extractor, "TOKENS_PER_VIEW", 1)
    monkeypatch.setattr(extractor, "TOKEN_DIM", 2)
    extractor.initialize_cache(source, args, keys)

    with pytest.raises(ValueError, match="decision_keys mismatch"):
        extractor.validate_cache(source, args, np.asarray([[0, 1, 1]], dtype=np.int64))
