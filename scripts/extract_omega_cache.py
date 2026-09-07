"""Extract per-decision VGGT-Omega camera/register tokens.

The input is a JSONL manifest with one record per synchronized timestep:
``{"episode": 0, "step": 0, "images": {"agentview_left": "...", ...}}``.
Records are sampled at ``--decision-stride`` and written as
``[N, V, L, 17, 2048]`` float16 tokens.  Keeping this extractor separate from
JAX training prevents the frozen PyTorch Omega model from entering the trainer.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
from PIL import Image
import torch

DEFAULT_CAMERA_ORDER = ("agentview_left", "agentview_right", "eye_in_hand")
LAYER_INDICES = (4, 11, 17, 23)
TOKEN_DIM = 2048
TOKENS_PER_VIEW = 17


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=pathlib.Path, required=True)
    parser.add_argument("--omega-repo", type=pathlib.Path, required=True)
    parser.add_argument("--checkpoint", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--image-resolution", type=int, default=256)
    parser.add_argument("--decision-stride", type=int, default=50)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--camera",
        dest="camera_order",
        action="append",
        help="Canonical camera name, repeat once per synchronized view (default: three benchmark views)",
    )
    return parser.parse_args()


def canonical_camera_name(name: str) -> str:
    value = str(name).strip().lower()
    for prefix in ("observation.images.", "observation/", "observation."):
        value = value.removeprefix(prefix)
    value = value.removeprefix("robot0_")
    aliases = {
        "front_rgb": "agentview_left",
        "front": "agentview_left",
        "agentview": "agentview_left",
        "wrist_rgb": "eye_in_hand",
        "wrist": "eye_in_hand",
        "eye_in_hand_rgb": "eye_in_hand",
    }
    return aliases.get(value, value)


def load_manifest(path: pathlib.Path, stride: int, camera_order: tuple[str, ...]) -> list[dict]:
    records = []
    manifest_root = path.parent.resolve()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                episode = int(record["episode"])
                step = int(record["step"])
                images = record["images"]
                if not isinstance(images, dict):
                    raise TypeError("images must be a mapping")
                canonical_images = {}
                for name, value in images.items():
                    canonical_name = canonical_camera_name(name)
                    if canonical_name in canonical_images:
                        raise ValueError(f"duplicate camera alias {canonical_name!r}")
                    canonical_images[canonical_name] = value
                paths = []
                for name in camera_order:
                    image_path = pathlib.Path(canonical_images[name])
                    if not image_path.is_absolute():
                        image_path = manifest_root / image_path
                    paths.append(image_path)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"Invalid manifest record at line {line_number}") from error
            for image_path in paths:
                if not image_path.is_file():
                    raise FileNotFoundError(
                        f"Manifest record at line {line_number} references missing image {image_path}"
                    )
            if step % stride == 0:
                records.append({"episode": episode, "step": step, "paths": paths})
    records.sort(key=lambda item: (item["episode"], item["step"]))
    if not records:
        raise ValueError("Manifest contains no decision records")
    return records


def preprocess(path: pathlib.Path, resolution: int) -> torch.Tensor:
    with Image.open(path) as source_image:
        image = source_image.convert("RGB").resize((resolution, resolution), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def load_model(repo: pathlib.Path, checkpoint: pathlib.Path, device: str):
    sys.path.insert(0, str(repo.resolve()))
    from vggt_omega.models import VGGTOmega

    model = VGGTOmega(enable_camera=False, enable_depth=False, enable_alignment=False)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model"):
        if isinstance(state, dict) and isinstance(state.get(key), dict):
            state = state[key]
    if state and all(str(key).startswith("module.") for key in state):
        state = {str(key).removeprefix("module."): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing_aggregator = [name for name in missing if name.startswith("aggregator.")]
    unexpected_non_head = [
        name for name in unexpected if not name.startswith(("camera_head.", "dense_head.", "text_alignment_head."))
    ]
    if missing_aggregator or unexpected_non_head:
        raise RuntimeError(
            f"Incompatible Omega checkpoint: missing={missing_aggregator[:5]}, unexpected={unexpected_non_head[:5]}"
        )
    return model.to(device).eval()


def main() -> None:
    args = parse_args()
    if args.decision_stride <= 0 or args.image_resolution % 16:
        raise ValueError("decision-stride must be positive and image-resolution divisible by 16")
    camera_order = tuple(canonical_camera_name(name) for name in (args.camera_order or DEFAULT_CAMERA_ORDER))
    if not camera_order or len(set(camera_order)) != len(camera_order):
        raise ValueError("--camera must specify unique camera names")
    records = load_manifest(args.manifest, args.decision_stride, camera_order)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    token_path = args.output_dir / "omega_tokens.f16.npy"
    key_path = args.output_dir / "decision_keys.i64.npy"
    metadata_path = args.output_dir / "metadata.json"
    keys = np.asarray([(item["episode"], item["step"] // args.decision_stride) for item in records], dtype=np.int64)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError("Manifest contains duplicate episode/decision keys")
    tokens = np.lib.format.open_memmap(
        token_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(records), len(camera_order), len(LAYER_INDICES), TOKENS_PER_VIEW, TOKEN_DIM),
    )
    key_cache = np.lib.format.open_memmap(key_path, mode="w+", dtype=np.int64, shape=keys.shape)
    key_cache[:] = keys
    key_cache.flush()
    model = load_model(args.omega_repo, args.checkpoint, args.device)
    for row, record in enumerate(records):
        images = torch.stack([preprocess(path, args.image_resolution) for path in record["paths"]])
        with torch.inference_mode():
            aggregated, patch_token_start = model.aggregator(images[None].to(args.device))
        if patch_token_start != TOKENS_PER_VIEW:
            raise RuntimeError(f"Unexpected Omega patch-token start: {patch_token_start}")
        selected = [aggregated[index][:, :, :TOKENS_PER_VIEW, :] for index in LAYER_INDICES]
        output = torch.stack(selected, dim=2)
        expected = (1, len(camera_order), len(LAYER_INDICES), TOKENS_PER_VIEW, TOKEN_DIM)
        if tuple(output.shape) != expected:
            raise RuntimeError(f"Unexpected Omega output {tuple(output.shape)}, expected {expected}")
        tokens[row] = output[0].float().cpu().numpy().astype(np.float16)
    tokens.flush()
    metadata_path.write_text(
        json.dumps(
            {
                "cache_kind": "omega_camera_register_decisions",
                "format_version": 1,
                "decision_stride": args.decision_stride,
                "camera_order": list(camera_order),
                "cached_layer_indices": list(LAYER_INDICES),
                "token_layout": "camera_then_16_register_tokens",
                "dtype": "float16",
                "shape": list(tokens.shape),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {tokens.shape} Omega cache to {args.output_dir}")


if __name__ == "__main__":
    main()
