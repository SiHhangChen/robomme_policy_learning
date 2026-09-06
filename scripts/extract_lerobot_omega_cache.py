"""Build a resumable VGGT-Omega cache directly from a local LeRobot v3 dataset.

The cache stores one synchronized three-view token bank every decision stride.
Training gathers strict-past rows from this bank, so the frozen PyTorch Omega
model never enters the JAX training process.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pathlib
import shutil
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch

from mme_vla_omega_suite.training.dataset import THREE_CAMERA_ORDER
from mme_vla_omega_suite.training.dataset import LeRobotV3Source

CACHE_FORMAT_VERSION = 1
CACHED_LAYER_INDICES = (4, 11, 17, 23)
TOKENS_PER_VIEW = 17
TOKEN_DIM = 2048


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=pathlib.Path, required=True)
    parser.add_argument("--omega-repo", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--image-resolution", type=int, default=256)
    parser.add_argument("--decision-stride-frames", type=int, default=50)
    parser.add_argument("--tolerance-s", type=float, default=0.051)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--flush-interval", type=int, default=4)
    parser.add_argument("--max-decisions", type=int, default=0, help="Debug limit; zero means all decisions")
    parser.add_argument("--initialize-only", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def cache_paths(output_dir: pathlib.Path) -> dict[str, pathlib.Path]:
    return {
        "tokens": output_dir / "scene_tokens.f16.npy",
        "keys": output_dir / "decision_keys.i64.npy",
        "completed": output_dir / "completed.u8.npy",
        "metadata": output_dir / "metadata.json",
    }


def initialize_cache(source: LeRobotV3Source, args: argparse.Namespace, keys: np.ndarray) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = cache_paths(args.output_dir)
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing cache files: {existing}")
    shape = (
        len(keys),
        len(THREE_CAMERA_ORDER),
        len(CACHED_LAYER_INDICES),
        TOKENS_PER_VIEW,
        TOKEN_DIM,
    )
    required_bytes = math.prod(shape) * np.dtype(np.float16).itemsize
    free_bytes = shutil.disk_usage(args.output_dir).free
    headroom = 2 * 2**30
    if free_bytes < required_bytes + headroom:
        raise OSError(
            f"Insufficient free space: need {required_bytes / 2**30:.2f} GiB plus 2 GiB headroom, "
            f"have {free_bytes / 2**30:.2f} GiB"
        )

    tokens = np.lib.format.open_memmap(paths["tokens"], mode="w+", dtype=np.float16, shape=shape)
    tokens[:] = 0
    tokens.flush()
    del tokens
    completed = np.lib.format.open_memmap(paths["completed"], mode="w+", dtype=np.uint8, shape=(len(keys),))
    completed[:] = 0
    completed.flush()
    del completed
    key_cache = np.lib.format.open_memmap(paths["keys"], mode="w+", dtype=np.int64, shape=keys.shape)
    key_cache[:] = keys
    key_cache.flush()
    del key_cache

    metadata = {
        "format_version": CACHE_FORMAT_VERSION,
        "cache_kind": "single_timestep_intermediate_camera_register",
        "dataset_root": str(source.root),
        "num_dataset_frames": len(source),
        "num_decision_rows": len(keys),
        "cached_shape": list(shape),
        "cached_per_decision_shape": list(shape[1:]),
        "dtype": "float16",
        "camera_order": list(THREE_CAMERA_ORDER),
        "token_layout": "camera_then_16_register_tokens",
        "cached_layer_indices": list(CACHED_LAYER_INDICES),
        "omega_output_dim": TOKEN_DIM,
        "image_resolution": args.image_resolution,
        "decision_stride_frames": args.decision_stride_frames,
        "video_tolerance_s": args.tolerance_s,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "required_bytes": required_bytes,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"initialized {shape[0]} rows with per-row shape {shape[1:]} "
        f"({required_bytes / 2**30:.2f} GiB) under {args.output_dir}",
        flush=True,
    )


def validate_cache(
    source: LeRobotV3Source, args: argparse.Namespace, keys: np.ndarray
) -> dict[str, pathlib.Path]:
    paths = cache_paths(args.output_dir)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing cache file {path}; run --initialize-only first")
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    per_row_shape = [len(THREE_CAMERA_ORDER), len(CACHED_LAYER_INDICES), TOKENS_PER_VIEW, TOKEN_DIM]
    required = {
        "format_version": CACHE_FORMAT_VERSION,
        "cache_kind": "single_timestep_intermediate_camera_register",
        "dataset_root": str(source.root),
        "num_dataset_frames": len(source),
        "num_decision_rows": len(keys),
        "cached_per_decision_shape": per_row_shape,
        "dtype": "float16",
        "camera_order": list(THREE_CAMERA_ORDER),
        "token_layout": "camera_then_16_register_tokens",
        "cached_layer_indices": list(CACHED_LAYER_INDICES),
        "decision_stride_frames": args.decision_stride_frames,
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise ValueError(f"Cache metadata mismatch for {key}: expected {expected!r}, got {metadata.get(key)!r}")

    cached_keys = np.load(paths["keys"], mmap_mode="r")
    if cached_keys.shape != keys.shape or cached_keys.dtype != np.int64 or not np.array_equal(cached_keys, keys):
        raise ValueError(
            f"decision_keys mismatch: expected {keys.shape}@int64 matching the dataset, "
            f"got {cached_keys.shape}@{cached_keys.dtype}"
        )
    tokens = np.load(paths["tokens"], mmap_mode="r")
    expected_shape = (len(keys), *per_row_shape)
    if tokens.shape != expected_shape or tokens.dtype != np.float16:
        raise ValueError(f"Token cache mismatch: expected {expected_shape}@float16, got {tokens.shape}@{tokens.dtype}")
    completed = np.load(paths["completed"], mmap_mode="r")
    if completed.shape != (len(keys),) or completed.dtype != np.uint8:
        raise ValueError(
            f"Completion bitmap mismatch: expected {(len(keys),)}@uint8, got {completed.shape}@{completed.dtype}"
        )
    if np.any(completed > 1):
        raise ValueError("Completion bitmap contains values other than zero and one")
    return paths


def load_checkpoint(path: pathlib.Path) -> dict[str, torch.Tensor]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    state: Any = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    if not isinstance(state, dict):
        raise TypeError(f"Expected a state dict in {path}, got {type(state)}")
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    return state


def preprocess_image(image: np.ndarray, resolution: int) -> torch.Tensor:
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected an HWC image, got {image.shape}")
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    if np.issubdtype(image.dtype, np.floating):
        scale = 255.0 if image.size and float(np.nanmax(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0.0, 255.0)
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    pil = pil.resize((resolution, resolution), Image.Resampling.BICUBIC)
    pixels = np.array(pil, dtype=np.float32) / 255.0
    return torch.from_numpy(pixels).permute(2, 0, 1).contiguous()


def decode_decision_images(source: LeRobotV3Source, raw_index: int, resolution: int) -> torch.Tensor:
    return torch.stack([preprocess_image(image, resolution) for image in source.decode_views(raw_index)], dim=0)


def extract(
    args: argparse.Namespace,
    source: LeRobotV3Source,
    keys: np.ndarray,
    paths: dict[str, pathlib.Path],
) -> None:
    sys.path.insert(0, str(args.omega_repo.expanduser().resolve()))
    from vggt_omega.models import VGGTOmega

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {args.device}, but CUDA is unavailable")
    model = VGGTOmega(enable_camera=False, enable_depth=False, enable_alignment=False)
    missing, unexpected = model.load_state_dict(load_checkpoint(args.checkpoint), strict=False)
    missing_aggregator = [name for name in missing if name.startswith("aggregator.")]
    unexpected_non_head = [
        name
        for name in unexpected
        if not name.startswith(("camera_head.", "dense_head.", "text_alignment_head."))
    ]
    if missing_aggregator or unexpected_non_head:
        raise RuntimeError(
            "Omega checkpoint is incompatible with its aggregator: "
            f"missing={missing_aggregator[:10]}, unexpected={unexpected_non_head[:10]}"
        )
    model = model.to(args.device).eval()
    token_cache = np.load(paths["tokens"], mmap_mode="r+")
    completed = np.load(paths["completed"], mmap_mode="r+")
    rows = np.arange(args.shard_index, len(keys), args.num_shards, dtype=np.int64)
    rows = rows[completed[rows] == 0]
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    print(
        f"shard={args.shard_index}/{args.num_shards} remaining={len(rows)} "
        f"batch={args.batch_size} resolution={args.image_resolution}",
        flush=True,
    )
    started = time.monotonic()
    for batch_start in range(0, len(rows), args.batch_size):
        batch_rows = rows[batch_start : batch_start + args.batch_size]
        images = []
        for cache_row in batch_rows:
            episode, frame, raw_index = (int(value) for value in keys[cache_row])
            if source.frame_identity(raw_index) != (episode, frame, raw_index):
                raise ValueError(f"Decision key row {cache_row} does not match its source frame")
            images.append(decode_decision_images(source, raw_index, args.image_resolution))
        batch = torch.stack(images, dim=0).to(args.device, non_blocking=True)
        with torch.inference_mode():
            aggregated, patch_token_start = model.aggregator(batch)
        selected = []
        for layer_index in CACHED_LAYER_INDICES:
            layer = aggregated[layer_index]
            if layer is None:
                raise RuntimeError(f"Omega did not return configured cached layer {layer_index}")
            selected.append(layer[:, :, :TOKENS_PER_VIEW, :])
        output = torch.stack(selected, dim=2)
        expected = (
            len(batch_rows),
            len(THREE_CAMERA_ORDER),
            len(CACHED_LAYER_INDICES),
            TOKENS_PER_VIEW,
            TOKEN_DIM,
        )
        if tuple(output.shape) != expected or patch_token_start != TOKENS_PER_VIEW:
            raise RuntimeError(
                f"Unexpected Omega output {tuple(output.shape)} / patch start {patch_token_start}; expected {expected}"
            )
        token_cache[batch_rows] = output.float().cpu().numpy().astype(np.float16)
        completed[batch_rows] = 1
        processed = batch_start + len(batch_rows)
        if processed == len(rows) or processed // args.batch_size % args.flush_interval == 0:
            token_cache.flush()
            completed.flush()
        if processed == len(rows) or processed % max(args.batch_size * 10, 1) == 0:
            elapsed = max(time.monotonic() - started, 1e-6)
            rate = processed / elapsed
            eta = (len(rows) - processed) / max(rate, 1e-6)
            print(
                f"shard={args.shard_index} {processed}/{len(rows)} rate={rate:.2f} rows/s eta={eta / 3600:.2f}h",
                flush=True,
            )
    token_cache.flush()
    completed.flush()


def validate_args(args: argparse.Namespace) -> None:
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num-shards > 0 and 0 <= shard-index < num-shards")
    if args.image_resolution <= 0 or args.image_resolution % 16:
        raise ValueError("image-resolution must be a positive multiple of 16")
    if args.decision_stride_frames <= 0 or args.batch_size <= 0 or args.flush_interval <= 0:
        raise ValueError("decision stride, batch size, and flush interval must be positive")
    if args.max_decisions < 0 or args.tolerance_s <= 0:
        raise ValueError("max-decisions must be non-negative and tolerance-s must be positive")
    if args.initialize_only and (args.verify_only or args.resume):
        raise ValueError("--initialize-only cannot be combined with --verify-only or --resume")
    if args.verify_only and args.resume:
        raise ValueError("--verify-only cannot be combined with --resume")


def main() -> None:
    args = parse_args()
    validate_args(args)
    source = LeRobotV3Source(str(args.dataset_root), video_tolerance_s=args.tolerance_s)
    keys = source.decision_keys(args.decision_stride_frames, max_decisions=args.max_decisions)
    if not len(keys):
        raise ValueError("No decision rows were found")
    if args.initialize_only:
        initialize_cache(source, args, keys)
        return
    paths = validate_cache(source, args, keys)
    if args.verify_only:
        completed = np.load(paths["completed"], mmap_mode="r")
        done = int(np.count_nonzero(completed == 1))
        print(f"completed={done}/{len(completed)} cache={args.output_dir}")
        if done != len(completed):
            raise RuntimeError(f"Cache is incomplete: {len(completed) - done} rows missing")
        return
    if not args.resume:
        raise ValueError("Refusing to write an initialized cache without --resume")
    extract(args, source, keys, paths)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()
