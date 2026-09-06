"""Build RoboMME pickle data with the three camera views required by Omega.

The legacy RoboMME preprocessor intentionally stores only ``front_rgb`` and
``wrist_rgb``.  This entrypoint runs that processor and then adds a real right
agent-view image to every execution sample.  It keeps the original feature and
action generation code unchanged while making the Omega suite's three-camera
contract explicit.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
import pathlib
import pickle

import h5py

from mme_vla_suite.dataset_builder.build_robomme_dataset import DatasetProcessor

DEFAULT_RIGHT_CAMERA_CANDIDATES = (
    "obs/agentview_right",
    "obs/agentview_right_rgb",
    "obs/right_rgb",
    "obs/right_camera_rgb",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-path", type=pathlib.Path, required=True)
    parser.add_argument("--preprocessed-data-path", type=pathlib.Path, required=True)
    parser.add_argument(
        "--right-camera-key",
        default=None,
        help="Right-camera dataset path relative to each timestep, e.g. obs/agentview_right_rgb",
    )
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def _h5_files(raw_data_path: pathlib.Path) -> list[pathlib.Path]:
    # Match DatasetProcessor's os.listdir ordering so global episode ids stay aligned.
    return [
        raw_data_path / name
        for name in os.listdir(raw_data_path)
        if name.endswith(".h5")
    ]


def _episode_sources(raw_data_path: pathlib.Path, max_episodes: int | None) -> list[tuple[pathlib.Path, int]]:
    sources: list[tuple[pathlib.Path, int]] = []
    for path in _h5_files(raw_data_path):
        with h5py.File(path, "r") as data:
            episode_indices = sorted(
                int(name.removeprefix("episode_"))
                for name in data
                if name.startswith("episode_")
            )
            if max_episodes is not None:
                episode_indices = episode_indices[:max_episodes]
            sources.extend((path, episode_index) for episode_index in episode_indices)
    return sources


def _resolve_right_camera(episode: h5py.Group, candidates: tuple[str, ...]) -> str:
    for path in candidates:
        try:
            item = episode["timestep_0"][path]
        except KeyError:
            continue
        if isinstance(item, h5py.Dataset):
            return path
    raise KeyError(f"Missing right camera; tried {candidates}")


def _validate_right_cameras(
    sources: list[tuple[pathlib.Path, int]], candidates: tuple[str, ...]
) -> dict[int, tuple[pathlib.Path, int, str]]:
    resolved: dict[int, tuple[pathlib.Path, int, str]] = {}
    for global_episode, (path, episode_index) in enumerate(sources):
        with h5py.File(path, "r") as data:
            episode = data[f"episode_{episode_index}"]
            right_path = _resolve_right_camera(episode, candidates)
            for timestep_name in episode:
                if not timestep_name.startswith("timestep_"):
                    continue
                try:
                    item = episode[timestep_name][right_path]
                except KeyError as error:
                    raise KeyError(
                        f"Episode {global_episode} timestep {timestep_name} is missing right camera {right_path!r}"
                    ) from error
                if not isinstance(item, h5py.Dataset):
                    raise TypeError(
                        f"Episode {global_episode} timestep {timestep_name} right camera {right_path!r} is not a dataset"
                    )
            resolved[global_episode] = (path, episode_index, right_path)
    return resolved


def _load_pickle_samples(dataset_path: pathlib.Path) -> list[pathlib.Path]:
    data_path = dataset_path / "data"
    return sorted(data_path.glob("*.pkl"), key=lambda path: int(path.stem))


def _inject_right_images(
    dataset_path: pathlib.Path,
    episode_sources: Mapping[int, tuple[pathlib.Path, int, str]],
) -> int:
    updated = 0
    open_files: dict[pathlib.Path, h5py.File] = {}
    try:
        for sample_path in _load_pickle_samples(dataset_path):
            with sample_path.open("rb") as handle:
                sample = pickle.load(handle)
            episode = int(sample["epis_idx"].item())
            step = int(sample["step_idx"].item())
            if episode not in episode_sources:
                raise ValueError(f"Sample {sample_path} references unknown global episode {episode}")
            source_path, source_episode, right_path = episode_sources[episode]
            source = open_files.setdefault(source_path, h5py.File(source_path, "r"))
            image = source[f"episode_{source_episode}"][f"timestep_{step}"][right_path][()]
            sample["right_image"] = image
            with sample_path.open("wb") as handle:
                pickle.dump(sample, handle, protocol=pickle.HIGHEST_PROTOCOL)
            updated += 1
    finally:
        for source in open_files.values():
            source.close()
    return updated


def build_dataset(
    raw_data_path: pathlib.Path,
    preprocessed_data_path: pathlib.Path,
    *,
    right_camera_key: str | None = None,
    max_episodes: int | None = None,
    visualize: bool = False,
) -> pathlib.Path:
    raw_data_path = raw_data_path.expanduser().resolve()
    preprocessed_data_path = preprocessed_data_path.expanduser().resolve()
    if not raw_data_path.is_dir():
        raise FileNotFoundError(raw_data_path)
    if max_episodes is not None and max_episodes <= 0:
        raise ValueError("max_episodes must be positive")

    sources = _episode_sources(raw_data_path, max_episodes)
    if not sources:
        raise ValueError(f"No HDF5 episodes found under {raw_data_path}")
    candidates = (right_camera_key,) if right_camera_key else DEFAULT_RIGHT_CAMERA_CANDIDATES
    resolved = _validate_right_cameras(sources, candidates)

    DatasetProcessor(
        raw_data_path=str(raw_data_path),
        preprocessed_data_path=str(preprocessed_data_path),
        visualize=visualize,
        max_episodes=max_episodes,
    ).run()
    updated = _inject_right_images(preprocessed_data_path, resolved)
    metadata = {
        "camera_order": ["agentview_left", "agentview_right", "eye_in_hand"],
        "right_camera_candidates": list(candidates),
        "num_execution_samples_with_right_image": updated,
    }
    (preprocessed_data_path / "meta" / "omega_cameras.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"added right-camera images to {updated} execution samples")
    return preprocessed_data_path


def main() -> None:
    args = parse_args()
    build_dataset(
        args.raw_data_path,
        args.preprocessed_data_path,
        right_camera_key=args.right_camera_key,
        max_episodes=args.max_episodes,
        visualize=args.visualize,
    )


if __name__ == "__main__":
    main()
