"""Build a three-view Omega image manifest from RoboMME HDF5 episodes.

The regular RoboMME preprocessor intentionally keeps only the two images used
by the original policy.  Omega needs the complete synchronized camera tuple,
so this script is an independent export step.  It writes one RGB image per
camera and decision step and a JSONL manifest consumed by
``scripts/extract_omega_cache.py``.

The exporter fails when a required camera is absent.  In particular, it never
duplicates the left camera to stand in for ``agentview_right``.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import os
import pathlib

import h5py
import numpy as np
from PIL import Image

CAMERA_ORDER = ("agentview_left", "agentview_right", "eye_in_hand")
DEFAULT_CAMERA_CANDIDATES: dict[str, tuple[str, ...]] = {
    "agentview_left": (
        "obs/agentview_left",
        "obs/agentview_left_rgb",
        "obs/front_rgb",
    ),
    "agentview_right": (
        "obs/agentview_right",
        "obs/agentview_right_rgb",
        "obs/right_rgb",
        "obs/right_camera_rgb",
    ),
    "eye_in_hand": (
        "obs/eye_in_hand",
        "obs/eye_in_hand_rgb",
        "obs/wrist_rgb",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-path", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--decision-stride", type=int, default=50)
    parser.add_argument(
        "--camera-key",
        action="append",
        metavar="CAMERA=H5_PATH",
        help="Override a camera dataset path, e.g. agentview_right=obs/agentview_right_rgb",
    )
    parser.add_argument(
        "--manifest-name",
        default="omega_manifest.jsonl",
        help="Manifest filename relative to --output-dir",
    )
    return parser.parse_args()


def _parse_camera_overrides(values: list[str] | None) -> dict[str, str]:
    overrides: dict[str, str] = {}
    for value in values or []:
        camera, separator, h5_path = value.partition("=")
        if not separator or camera not in CAMERA_ORDER or not h5_path:
            raise ValueError(f"--camera-key must use CAMERA=H5_PATH with one of {CAMERA_ORDER}, got {value!r}")
        if camera in overrides:
            raise ValueError(f"Duplicate --camera-key for {camera!r}")
        overrides[camera] = h5_path
    return overrides


def _find_dataset(group: h5py.Group, candidates: tuple[str, ...], camera: str) -> tuple[str, h5py.Dataset]:
    for path in candidates:
        try:
            item = group[path]
        except KeyError:
            continue
        if not isinstance(item, h5py.Dataset):
            raise TypeError(f"HDF5 path {path!r} for {camera} is not a dataset")
        return path, item
    available = sorted(_walk_dataset_paths(group))
    raise KeyError(
        f"Missing required Omega camera {camera!r}; tried {candidates}. Available datasets include: {available[:30]}"
    )


def _walk_dataset_paths(group: h5py.Group, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for name, item in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(item, h5py.Dataset):
            paths.append(path)
        elif isinstance(item, h5py.Group):
            paths.extend(_walk_dataset_paths(item, path))
    return paths


def _as_uint8_rgb(value: np.ndarray, *, source: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3:
        raise ValueError(f"Camera dataset {source!r} must contain rank-3 images, got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    if image.shape[-1] == 4:
        image = image[..., :3]
    if image.shape[-1] != 3:
        raise ValueError(f"Camera dataset {source!r} must have 3 RGB channels, got {image.shape}")
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if float(np.nanmax(image)) <= 1.0 else 1.0
        image = image * scale
    return np.clip(image, 0, 255).astype(np.uint8)


def _episode_indices(data: h5py.File) -> list[int]:
    return sorted(int(name.removeprefix("episode_")) for name in data if name.startswith("episode_"))


def _timestep_indices(episode: h5py.Group) -> list[int]:
    return sorted(int(name.removeprefix("timestep_")) for name in episode if name.startswith("timestep_"))


def build_manifest(
    raw_data_path: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    decision_stride: int,
    camera_overrides: Mapping[str, str] | None = None,
    manifest_name: str = "omega_manifest.jsonl",
) -> pathlib.Path:
    if decision_stride <= 0:
        raise ValueError("decision_stride must be positive")
    if not raw_data_path.is_dir():
        raise FileNotFoundError(raw_data_path)
    # DatasetProcessor assigns global episode ids in os.listdir() order.  Keep
    # the same order here so cache keys line up with preprocessed pickles.
    h5_paths = [
        raw_data_path / name
        for name in os.listdir(raw_data_path)
        if name.endswith(".h5")
    ]
    if not h5_paths:
        raise FileNotFoundError(f"No .h5 files found under {raw_data_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / manifest_name
    global_episode = 0
    records = 0

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for h5_path in h5_paths:
            with h5py.File(h5_path, "r") as data:
                for episode_index in _episode_indices(data):
                    episode = data[f"episode_{episode_index}"]
                    timestep_indices = _timestep_indices(episode)
                    if not timestep_indices:
                        global_episode += 1
                        continue
                    for step in timestep_indices:
                        if step % decision_stride:
                            continue
                        timestep = episode[f"timestep_{step}"]
                        datasets: dict[str, tuple[str, h5py.Dataset]] = {}
                        for camera in CAMERA_ORDER:
                            candidates = (
                                (camera_overrides or {}).get(camera),
                                *DEFAULT_CAMERA_CANDIDATES[camera],
                            )
                            datasets[camera] = _find_dataset(
                                timestep,
                                tuple(path for path in candidates if path),
                                camera,
                            )
                        source_paths = [datasets[camera][0] for camera in CAMERA_ORDER]
                        if len(set(source_paths)) != len(source_paths):
                            raise ValueError(
                                f"Episode {global_episode}, step {step} maps multiple Omega cameras to the same "
                                f"HDF5 path: {source_paths}. Supply distinct --camera-key overrides."
                            )
                        images: dict[str, str] = {}
                        for camera in CAMERA_ORDER:
                            source, dataset = datasets[camera]
                            try:
                                frame = dataset[()]
                            except (IndexError, ValueError) as error:
                                raise ValueError(
                                    f"Camera {camera} dataset {source!r} has no frame {step} "
                                    f"for episode {global_episode}"
                                ) from error
                            image_path = image_dir / f"episode_{global_episode}" / f"step_{step}_{camera}.png"
                            image_path.parent.mkdir(parents=True, exist_ok=True)
                            Image.fromarray(_as_uint8_rgb(frame, source=source)).save(image_path)
                            images[camera] = str(image_path.resolve())
                        manifest.write(
                            json.dumps(
                                {"episode": global_episode, "step": step, "images": images},
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                        records += 1
                    global_episode += 1

    if records == 0:
        raise ValueError("No decision records were exported")
    print(f"wrote {records} three-view records to {manifest_path}")
    return manifest_path


def main() -> None:
    args = parse_args()
    build_manifest(
        args.raw_data_path,
        args.output_dir,
        decision_stride=args.decision_stride,
        camera_overrides=_parse_camera_overrides(args.camera_key),
        manifest_name=args.manifest_name,
    )


if __name__ == "__main__":
    main()
