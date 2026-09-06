"""Frozen VGGT-Omega register-token encoder used by online policy inference."""

from __future__ import annotations

from collections import deque
import math
import pathlib
import sys
from typing import ClassVar

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as functional


class FrozenVGGTOmegaSceneEncoder:
    CAMERA_ORDER: ClassVar[tuple[str, ...]] = (
        "agentview_left",
        "agentview_right",
        "eye_in_hand",
    )

    def __init__(
        self,
        *,
        omega_repo: str | pathlib.Path,
        checkpoint: str | pathlib.Path,
        device: str = "cuda:0",
        image_resolution: int = 256,
        camera_order: tuple[str, ...] | None = None,
        output_tokens_per_view: int = 17,
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
        temporal_window_size: int = 1,
        temporal_stride_calls: int = 1,
    ) -> None:
        self._device = device
        self._image_resolution = int(image_resolution)
        self._camera_order = tuple(camera_order or self.CAMERA_ORDER)
        if not self._camera_order or len(set(self._camera_order)) != len(self._camera_order):
            raise ValueError("camera_order must contain unique camera names")
        self._output_tokens_per_view = int(output_tokens_per_view)
        self._cached_layer_indices = tuple(int(index) for index in cached_layer_indices)
        self._temporal_window_size = int(temporal_window_size)
        self._temporal_stride_calls = int(temporal_stride_calls)
        if self._image_resolution % 16:
            raise ValueError("VGGT-Omega image resolution must be divisible by 16")
        if self._output_tokens_per_view != 17:
            raise ValueError("Omega action conditioning requires all 17 camera/register tokens")
        if not self._cached_layer_indices:
            raise ValueError("cached_layer_indices must not be empty")
        if self._temporal_window_size <= 0:
            raise ValueError("temporal_window_size must be positive")
        if self._temporal_stride_calls <= 0:
            raise ValueError("temporal_stride_calls must be positive")
        # Each entry is one synchronized three-camera policy-decision snapshot.
        # Keeping the raw preprocessed views lets Omega jointly aggregate the
        # exact causal window used by the offline training cache.
        history_capacity = (self._temporal_window_size - 1) * self._temporal_stride_calls + 1
        self._history: deque[list[torch.Tensor]] = deque(maxlen=history_capacity)

        omega_repo = pathlib.Path(omega_repo).expanduser().resolve()
        checkpoint = pathlib.Path(checkpoint).expanduser().resolve()
        if not omega_repo.is_dir() or not checkpoint.is_file():
            raise FileNotFoundError(f"Omega repo/checkpoint missing: {omega_repo}, {checkpoint}")
        sys.path.insert(0, str(omega_repo))
        from vggt_omega.models import VGGTOmega

        model = VGGTOmega(enable_camera=False, enable_depth=False, enable_alignment=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        for key in ("state_dict", "model"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
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
                "Incompatible VGGT-Omega checkpoint: "
                f"missing={missing_aggregator[:10]}, unexpected={unexpected_non_head[:10]}"
            )
        self._model = model.to(self._device).eval()

    def encode(self, images: dict[str, np.ndarray]) -> np.ndarray:
        if set(images) != set(self._camera_order):
            raise ValueError(f"Expected synchronized views {self._camera_order}, got {tuple(images)}")
        tensors = [self._preprocess(images[name]) for name in self._camera_order]
        self._history.append(tensors)
        # Select the current snapshot and older snapshots at a fixed number of
        # policy calls.  For example, replan=25 with stride_calls=2 preserves
        # the same 50-environment-step spacing used by a replan=50 cache.
        sampled_history = list(self._history)[:: -self._temporal_stride_calls]
        sampled_history = sampled_history[: self._temporal_window_size][::-1]
        tensors = [view for snapshot in sampled_history for view in snapshot]
        max_height = max(image.shape[-2] for image in tensors)
        max_width = max(image.shape[-1] for image in tensors)
        padded = []
        for image in tensors:
            h_pad = max_height - image.shape[-2]
            w_pad = max_width - image.shape[-1]
            padded.append(
                functional.pad(
                    image,
                    (w_pad // 2, w_pad - w_pad // 2, h_pad // 2, h_pad - h_pad // 2),
                    value=1.0,
                )
            )
        omega_input = torch.stack(padded)[None].to(self._device)
        with torch.inference_mode():
            aggregated, patch_token_start = self._model.aggregator(omega_input)
        selected = []
        for layer_index in self._cached_layer_indices:
            layer = aggregated[layer_index]
            if layer is None:
                raise RuntimeError(f"Omega did not return configured cached layer {layer_index}")
            selected.append(layer[:, :, :17, :])
        output = torch.stack(selected, dim=2)
        if patch_token_start != 17:
            raise RuntimeError(f"Unexpected Omega patch-token start {patch_token_start}")
        # The final three Omega frames are the current left/right/wrist views.
        # Earlier decision snapshots influence them through Omega's aggregator,
        # but only the current views are returned to the policy connector.
        current = output[0, -len(self._camera_order) :].float()
        tokens = current.cpu().numpy()
        if tokens.shape != (len(self._camera_order), len(self._cached_layer_indices), 17, 2048):
            raise RuntimeError(f"Unexpected VGGT-Omega token shape: {tokens.shape}")
        return tokens.astype(np.float32, copy=False)

    @property
    def camera_order(self) -> tuple[str, ...]:
        return self._camera_order

    def reset(self) -> None:
        """Clear temporal images at an episode boundary."""
        self._history.clear()

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"Expected HWC/CHW image, got {array.shape}")
        if array.shape[0] in (1, 3, 4) and array.shape[-1] not in (1, 3, 4):
            array = np.transpose(array, (1, 2, 0))
        if np.issubdtype(array.dtype, np.floating):
            scale = 255.0 if float(np.nanmax(array)) <= 1.0 else 1.0
            array = np.clip(array * scale, 0.0, 255.0)
        image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
        width, height = image.size
        aspect = height / max(width, 1)
        if aspect < 0.5:
            crop_width = min(width, max(1, round(height / 0.5)))
            left = max((width - crop_width) // 2, 0)
            image = image.crop((left, 0, left + crop_width, height))
        elif aspect > 2.0:
            crop_height = min(height, max(1, round(width * 2.0)))
            top = max((height - crop_height) // 2, 0)
            image = image.crop((0, top, width, top + crop_height))

        width, height = image.size
        aspect = height / max(width, 1)
        token_count = (self._image_resolution // 16) ** 2
        width_patches = max(1, round(math.sqrt(token_count / aspect)))
        height_patches = max(1, round(token_count / width_patches))
        image = image.resize((width_patches * 16, height_patches * 16), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)
