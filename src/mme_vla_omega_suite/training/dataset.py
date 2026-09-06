"""Local LeRobot v3 reader augmented with a causal VGGT-Omega cache.

This reader intentionally does not instantiate ``LeRobotDataset``. Different
environments contain incompatible LeRobot APIs (and some try to contact the
Hub when ``tasks.jsonl`` is absent), while this experiment is fully local.
Parquet is read with PyArrow and frames are decoded with PyAV.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
import json
import pathlib

import av
import numpy as np
import pyarrow.parquet as parquet

from mme_vla_omega_suite.shared.omega_memory import gather_strict_past

THREE_CAMERA_ORDER = ("agentview_left", "agentview_right", "eye_in_hand")
LE_ROBOT_CAMERA_KEYS = {
    "agentview_left": "observation.images.robot0_agentview_left",
    "agentview_right": "observation.images.robot0_agentview_right",
    "eye_in_hand": "observation.images.robot0_eye_in_hand",
}
_INDEX_COLUMNS = (
    "observation.state",
    "action",
    "timestamp",
    "frame_index",
    "episode_index",
    "index",
    "task_index",
)


def canonical_camera_name(name: str) -> str:
    value = str(name).strip().lower()
    for prefix in ("observation.images.", "observation/", "observation."):
        value = value.removeprefix(prefix)
    value = value.removeprefix("robot0_")
    return {
        "front_rgb": "agentview_left",
        "front": "agentview_left",
        "agentview": "agentview_left",
        "wrist_rgb": "eye_in_hand",
        "wrist": "eye_in_hand",
        "eye_in_hand_rgb": "eye_in_hand",
    }.get(value, value)


def default_camera_order(num_views: int) -> tuple[str, ...]:
    if num_views != len(THREE_CAMERA_ORDER):
        raise ValueError(f"Omega requires three synchronized views, got omega_num_views={num_views}")
    return THREE_CAMERA_ORDER


def _read_table_columns(root: pathlib.Path) -> dict[str, np.ndarray]:
    """Read only numeric/index columns and return rows ordered by global index."""

    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No LeRobot parquet files found under {root / 'data'}")
    batches = [parquet.read_table(path, columns=list(_INDEX_COLUMNS)).to_pydict() for path in files]
    values = {
        key: np.asarray([item for batch in batches for item in batch[key]])
        for key in _INDEX_COLUMNS
    }
    order = np.argsort(values["index"].astype(np.int64), kind="stable")
    return {key: values[key][order] for key in _INDEX_COLUMNS}


def _load_tasks(root: pathlib.Path) -> dict[int, str]:
    path = root / "meta" / "tasks.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing LeRobot task metadata: {path}")
    table = parquet.read_table(path).to_pydict()
    return {int(index): str(task) for index, task in zip(table["task_index"], table["task"], strict=True)}


def _load_episode_video_metadata(root: pathlib.Path) -> dict[int, dict[str, tuple[int, int, float]]]:
    """Return camera -> (chunk, file, episode-start timestamp) per episode."""

    result: dict[int, dict[str, tuple[int, int, float]]] = {}
    files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata found under {root / 'meta/episodes'}")
    columns = ["episode_index"]
    for key in LE_ROBOT_CAMERA_KEYS.values():
        columns.extend(
            [
                f"videos/{key}/chunk_index",
                f"videos/{key}/file_index",
                f"videos/{key}/from_timestamp",
            ]
        )
    for path in files:
        table = parquet.read_table(path, columns=columns).to_pydict()
        for row in range(len(table["episode_index"])):
            episode = int(table["episode_index"][row])
            result[episode] = {}
            for camera, key in LE_ROBOT_CAMERA_KEYS.items():
                result[episode][camera] = (
                    int(table[f"videos/{key}/chunk_index"][row]),
                    int(table[f"videos/{key}/file_index"][row]),
                    float(table[f"videos/{key}/from_timestamp"][row]),
                )
    return result


class LeRobotV3Source:
    """Shared local LeRobot v3 index and synchronized video reader."""

    def __init__(self, dataset_path: str, *, video_tolerance_s: float = 0.051) -> None:
        self.root = pathlib.Path(dataset_path).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"LeRobot dataset directory does not exist: {self.root}")
        info_path = self.root / "meta" / "info.json"
        if not info_path.is_file():
            raise FileNotFoundError(f"Not a LeRobot v3 dataset (missing {info_path})")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        self.fps = float(info.get("fps", 20.0))
        self.video_tolerance_s = float(video_tolerance_s)
        if self.video_tolerance_s <= 0:
            raise ValueError("video_tolerance_s must be positive")

        columns = _read_table_columns(self.root)
        self._states = columns["observation.state"].astype(np.float32)
        self._actions = columns["action"].astype(np.float32)
        self._timestamps = columns["timestamp"].astype(np.float64).reshape(-1)
        self._frames = columns["frame_index"].astype(np.int64).reshape(-1)
        self._episodes = columns["episode_index"].astype(np.int64).reshape(-1)
        self._indices = columns["index"].astype(np.int64).reshape(-1)
        self._tasks = columns["task_index"].astype(np.int64).reshape(-1)
        if self._states.ndim != 2 or self._states.shape[1] != 37:
            raise ValueError(f"Expected observation.state [N,37], got {self._states.shape}")
        if self._actions.ndim != 2 or self._actions.shape[1] != 13:
            raise ValueError(f"Expected action [N,13], got {self._actions.shape}")
        if not np.array_equal(self._indices, np.arange(len(self._indices), dtype=np.int64)):
            raise ValueError("LeRobot data index must be contiguous and start at zero")
        self._tasks_by_index = _load_tasks(self.root)
        self._episode_videos = _load_episode_video_metadata(self.root)
        self._episode_positions: dict[int, np.ndarray] = {}
        self._episode_local_position = np.empty(len(self._indices), dtype=np.int32)
        for episode in np.unique(self._episodes):
            positions = np.flatnonzero(self._episodes == episode)
            positions = positions[np.argsort(self._frames[positions], kind="stable")]
            self._episode_positions[int(episode)] = positions
            self._episode_local_position[positions] = np.arange(len(positions), dtype=np.int32)
        self._video_containers: OrderedDict[tuple[str, int, int], av.container.InputContainer] = OrderedDict()
        self._video_cache_limit = 12

    def __len__(self) -> int:
        return len(self._indices)

    def __getstate__(self):
        state = self.__dict__.copy()
        for container in self._video_containers.values():
            container.close()
        state["_video_containers"] = OrderedDict()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._video_containers = OrderedDict()

    def decision_keys(self, stride: int, *, max_decisions: int = 0) -> np.ndarray:
        """Return cache identities as ``[episode, frame, raw_index]`` rows."""

        if stride <= 0:
            raise ValueError("stride must be positive")
        if max_decisions < 0:
            raise ValueError("max_decisions must be non-negative")
        rows: list[tuple[int, int, int]] = []
        for episode, positions in sorted(self._episode_positions.items()):
            for raw_index in positions[::stride]:
                rows.append((episode, int(self._frames[raw_index]), int(self._indices[raw_index])))
                if max_decisions and len(rows) >= max_decisions:
                    return np.asarray(rows, dtype=np.int64)
        return np.asarray(rows, dtype=np.int64).reshape(-1, 3)

    def decode_views(self, raw_index: int, camera_order: Sequence[str] = THREE_CAMERA_ORDER) -> list[np.ndarray]:
        if not 0 <= raw_index < len(self):
            raise IndexError(raw_index)
        episode = int(self._episodes[raw_index])
        timestamp = float(self._timestamps[raw_index])
        return [self._decode_frame(camera, episode, timestamp) for camera in camera_order]

    def frame_identity(self, raw_index: int) -> tuple[int, int, int]:
        if not 0 <= raw_index < len(self):
            raise IndexError(raw_index)
        return (
            int(self._episodes[raw_index]),
            int(self._frames[raw_index]),
            int(self._indices[raw_index]),
        )

    def _decode_frame(self, camera: str, episode: int, timestamp: float) -> np.ndarray:
        chunk, file_index, from_timestamp = self._episode_videos[episode][camera]
        key = (camera, chunk, file_index)
        container = self._video_containers.get(key)
        if container is None:
            video_key = LE_ROBOT_CAMERA_KEYS[camera]
            path = self.root / "videos" / video_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
            container = av.open(str(path))
            self._video_containers[key] = container
            if len(self._video_containers) > self._video_cache_limit:
                _, stale = self._video_containers.popitem(last=False)
                stale.close()
        else:
            self._video_containers.move_to_end(key)
        stream = container.streams.video[0]
        target = from_timestamp + timestamp
        container.seek(max(0, int(target / float(stream.time_base))), stream=stream, backward=True)
        previous = None
        for frame in container.decode(stream):
            previous = frame
            if frame.time >= target - 1e-3:
                break
        if previous is None:
            raise RuntimeError(f"Could not decode {camera} episode={episode} timestamp={timestamp}")
        if abs(float(previous.time) - target) > self.video_tolerance_s:
            raise RuntimeError(
                f"Decoded {camera} episode={episode} at {float(previous.time):.6f}s, "
                f"expected {target:.6f}s (tolerance {self.video_tolerance_s:.6f}s)"
            )
        return previous.to_ndarray(format="rgb24")


class OmegaRoboMMEDataset(LeRobotV3Source):
    """Random-access MemBench dataset with strict-past Omega memory."""

    def __init__(
        self,
        *,
        dataset_path: str,
        data_config,
        omega_cache_path: str | pathlib.Path,
        omega_memory_horizon: int,
        omega_stride: int,
        action_horizon: int,
        omega_num_views: int = 3,
        omega_camera_order: Sequence[str] | None = None,
        omega_num_layers: int = 4,
        omega_layer_indices: Sequence[int] = (4, 11, 17, 23),
        omega_tokens_per_view: int = 17,
        omega_input_dim: int = 2048,
        video_backend: str | None = None,
    ) -> None:
        del data_config, video_backend  # kept in the signature for loader compatibility
        super().__init__(dataset_path)
        self.action_horizon = int(action_horizon)
        self.omega_memory_horizon = int(omega_memory_horizon)
        self.omega_stride = int(omega_stride)
        self.omega_num_views = int(omega_num_views)
        self.omega_camera_order = tuple(
            str(name) for name in (omega_camera_order or default_camera_order(self.omega_num_views))
        )
        self.omega_num_layers = int(omega_num_layers)
        self.omega_layer_indices = tuple(int(index) for index in omega_layer_indices)
        self.omega_tokens_per_view = int(omega_tokens_per_view)
        self.omega_input_dim = int(omega_input_dim)
        if self.omega_memory_horizon <= 0 or self.omega_stride <= 0:
            raise ValueError("Omega horizon and stride must be positive")
        if len(self.omega_camera_order) != self.omega_num_views:
            raise ValueError("omega_camera_order must contain exactly omega_num_views names")
        if len(set(self.omega_camera_order)) != len(self.omega_camera_order):
            raise ValueError("omega_camera_order must not contain duplicates")
        if len(self.omega_layer_indices) != self.omega_num_layers:
            raise ValueError("omega_layer_indices must contain exactly omega_num_layers values")

        self._omega_cache_dir = pathlib.Path(omega_cache_path).expanduser().resolve()
        self._load_omega_cache(self._omega_cache_dir)

    def __getstate__(self):
        state = super().__getstate__()
        # A memmap is serialized as its complete 4 GB payload by pickle. Keep
        # only the cache location and reopen it in each spawned data worker.
        state["_omega_tokens"] = None
        state["_omega_keys"] = None
        state["_omega_rows"] = None
        return state

    def __setstate__(self, state):
        super().__setstate__(state)
        self._load_omega_cache(self._omega_cache_dir)

    def _load_omega_cache(self, cache_dir: pathlib.Path) -> None:
        metadata_path = cache_dir / "metadata.json"
        token_path = cache_dir / "omega_tokens.f16.npy"
        if not token_path.is_file():
            token_path = cache_dir / "scene_tokens.f16.npy"
        key_path = cache_dir / "decision_keys.i64.npy"
        missing = [str(path) for path in (metadata_path, token_path, key_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError("Incomplete Omega cache; missing " + ", ".join(missing))
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_kind") not in {
            "omega_camera_register_decisions",
            "single_timestep_intermediate_camera_register",
        }:
            raise ValueError(f"Unsupported Omega cache kind: {metadata.get('cache_kind')!r}")
        cache_stride = metadata.get("decision_stride", metadata.get("decision_stride_frames"))
        if int(cache_stride) != self.omega_stride:
            raise ValueError(f"Omega stride mismatch: expected {self.omega_stride}, got {cache_stride}")
        if metadata.get("cached_layer_indices") != list(self.omega_layer_indices):
            raise ValueError("Omega layer indices do not match cache metadata")
        if [canonical_camera_name(name) for name in metadata.get("camera_order", [])] != list(self.omega_camera_order):
            raise ValueError("Omega camera order does not match the three-view dataset contract")
        self._omega_tokens = np.load(token_path, mmap_mode="r")
        self._omega_keys = np.load(key_path, mmap_mode="r")
        expected_shape = (
            self.omega_num_views,
            self.omega_num_layers,
            self.omega_tokens_per_view,
            self.omega_input_dim,
        )
        if self._omega_tokens.shape[1:] != expected_shape or self._omega_tokens.dtype != np.float16:
            raise ValueError(f"Omega tokens must be [N,{','.join(map(str, expected_shape))}] float16")
        if self._omega_keys.ndim != 2 or self._omega_keys.shape[0] != self._omega_tokens.shape[0]:
            raise ValueError("Omega decision keys and token rows do not match")
        if self._omega_keys.shape[1] == 2:
            self._omega_rows = {
                (int(episode), int(decision)): row for row, (episode, decision) in enumerate(self._omega_keys)
            }
        elif self._omega_keys.shape[1] == 3:
            self._omega_rows = {
                (int(episode), int(frame) // self.omega_stride): row
                for row, (episode, frame, _raw) in enumerate(self._omega_keys)
            }
            for episode, frame, raw_index in self._omega_keys:
                cache_raw = int(raw_index)
                if not 0 <= cache_raw < len(self):
                    raise ValueError(f"Omega cache raw dataset index is out of range: {cache_raw}")
                if int(self._episodes[cache_raw]) != int(episode) or int(self._frames[cache_raw]) != int(frame):
                    raise ValueError(
                        "Omega cache is not aligned with the local dataset at "
                        f"raw index {cache_raw}: cache={(int(episode), int(frame))}, "
                        f"dataset={(int(self._episodes[cache_raw]), int(self._frames[cache_raw]))}"
                    )
        else:
            raise ValueError("Omega decision keys must have shape [N,2] or [N,3]")
        if len(self._omega_rows) != self._omega_keys.shape[0]:
            raise ValueError("Omega decision keys contain duplicates")

    def __getitem__(self, idx):
        index = int(idx)
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        episode = int(self._episodes[index])
        frame = int(self._frames[index])
        positions = self._episode_positions[episode]
        local_position = int(self._episode_local_position[index])
        action_rows = positions[np.minimum(np.arange(self.action_horizon) + local_position, len(positions) - 1)]
        data = {
            "observation.state": self._states[index],
            "action": self._actions[action_rows],
            "episode_index": np.asarray(episode, dtype=np.int64),
            "frame_index": np.asarray(frame, dtype=np.int64),
            "index": np.asarray(index, dtype=np.int64),
            "task_index": np.asarray(self._tasks[index], dtype=np.int64),
            "prompt": self._tasks_by_index[int(self._tasks[index])],
        }
        for camera, key in LE_ROBOT_CAMERA_KEYS.items():
            data[key] = self._decode_frame(camera, episode, float(self._timestamps[index]))
        data["omega_memory"], data["omega_memory_mask"] = gather_strict_past(
            self._omega_tokens,
            self._omega_rows,
            episode=episode,
            step=frame,
            horizon=self.omega_memory_horizon,
            stride=self.omega_stride,
        )
        return data


__all__ = [
    "LE_ROBOT_CAMERA_KEYS",
    "THREE_CAMERA_ORDER",
    "LeRobotV3Source",
    "OmegaRoboMMEDataset",
    "canonical_camera_name",
]
