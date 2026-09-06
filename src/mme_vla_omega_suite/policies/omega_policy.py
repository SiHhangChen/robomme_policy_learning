"""Online policy wrapper for the Omega-conditioned action expert."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from typing_extensions import override

from mme_vla_omega_suite.models.integration.omega_observation import OmegaObservation
from mme_vla_omega_suite.models.integration.omega_pi0 import OmegaPi0
from openpi import transforms as _transforms
from openpi.shared import nnx_utils


class OmegaPolicy:
    """Policy wrapper with strict-past online Omega memory semantics."""

    def __init__(
        self,
        model: OmegaPi0,
        *,
        seed: int = 42,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        omega_scene_encoder: Any | None = None,
    ) -> None:
        self._model = model
        self._seed = int(seed)
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._omega_scene_encoder = omega_scene_encoder
        self._omega_horizon = model.config.omega_memory_horizon
        self._omega_history: deque[np.ndarray] = deque(maxlen=self._omega_horizon)
        self._sample_actions = nnx_utils.module_jit(model.sample_actions)
        self.reset()

    @override
    def infer(self, obs: dict) -> dict:
        inputs = jax.tree.map(lambda x: x, obs)
        current_scene = None
        if "omega_memory" not in inputs:
            if self._omega_scene_encoder is None:
                raise ValueError("Omega online inference requires a frozen scene encoder")
            current_scene = self._omega_scene_encoder.encode(
                self._extract_camera_images(inputs, self._omega_scene_encoder.camera_order)
            )
            memory, memory_mask = self._history_before_current(current_scene)
            inputs["omega_memory"] = memory
            inputs["omega_memory_mask"] = memory_mask

        inputs = self._input_transform(inputs)
        observation = OmegaObservation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs))
        self._rng, sample_rng = jax.random.split(self._rng)
        start_time = time.monotonic()
        outputs = {
            "state": observation.state,
            "actions": self._sample_actions(sample_rng, observation, **self._sample_kwargs),
        }
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)
        outputs["infer_time_ms"] = (time.monotonic() - start_time) * 1000

        # The current decision becomes history only after its action has been
        # generated, matching the offline strict-past cache contract.
        if current_scene is not None:
            self._omega_history.append(np.asarray(current_scene, dtype=np.float32))
        return outputs

    def _history_before_current(self, current_scene: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current_scene = np.asarray(current_scene)
        expected = (
            self._omega_horizon,
            self._model.config.omega_num_views,
            len(self._model.config.omega_layer_indices),
            self._model.config.omega_tokens_per_view,
            self._model.config.omega_input_dim,
        )
        if current_scene.shape != expected[1:]:
            raise ValueError(f"Unexpected current Omega scene shape: {current_scene.shape}; expected {expected[1:]}")
        memory = np.zeros(expected, dtype=np.float16)
        mask = np.zeros((self._omega_horizon,), dtype=np.bool_)
        history = list(self._omega_history)
        count = min(len(history), self._omega_horizon)
        if count:
            memory[-count:] = np.asarray(history[-count:], dtype=np.float16)
            mask[-count:] = True
        return memory, mask

    @staticmethod
    def _extract_camera_images(inputs: dict, camera_order: Sequence[str]) -> dict[str, np.ndarray]:
        aliases = {
            "agentview_left": (
                "observation/image",
                "image",
                "agentview_left",
                "observation.images.agentview_left",
                "observation.images.robot0_agentview_left",
            ),
            "agentview_right": (
                "observation/right_image",
                "right_image",
                "agentview_right",
                "observation.images.agentview_right",
                "observation.images.robot0_agentview_right",
            ),
            "eye_in_hand": (
                "observation/wrist_image",
                "wrist_image",
                "eye_in_hand",
                "observation.images.eye_in_hand",
                "observation.images.robot0_eye_in_hand",
            ),
        }
        images = {}
        nested_images = inputs.get("images")
        if isinstance(nested_images, dict):
            inputs = {**inputs, **nested_images}
        for camera in camera_order:
            if camera not in aliases:
                raise ValueError(f"No online image alias is configured for Omega camera '{camera}'")
            names = aliases[camera]
            for name in names:
                if name in inputs:
                    value = inputs[name]
                    if isinstance(value, dict):
                        continue
                    images[camera] = np.asarray(value)
                    break
            else:
                raise ValueError(
                    f"Missing Omega camera '{camera}'. Expected one of {names}; "
                    "the online policy requires synchronized three-view input."
                )
        return images

    @override
    def reset(self) -> None:
        self._omega_history.clear()
        if self._omega_scene_encoder is not None and hasattr(self._omega_scene_encoder, "reset"):
            self._omega_scene_encoder.reset()
        self._rng = jax.random.key(self._seed)

    def add_buffer(self, obs: dict) -> None:
        """Append already-observed strict-past Omega snapshots.

        ``omega_images`` is a mapping from each canonical camera name to a
        time-major array/list.  Every supplied snapshot is treated as past
        context; the current observation must be passed to :meth:`infer` and
        is appended only after its action is sampled.  A caller may instead
        provide precomputed ``omega_memory`` and ``omega_memory_mask`` using
        the same unbatched layout as the model input.
        """

        if "omega_memory" in obs:
            memory = np.asarray(obs["omega_memory"])
            mask = np.asarray(obs.get("omega_memory_mask"), dtype=np.bool_)
            expected = (
                self._omega_horizon,
                self._model.config.omega_num_views,
                len(self._model.config.omega_layer_indices),
                self._model.config.omega_tokens_per_view,
                self._model.config.omega_input_dim,
            )
            if memory.shape != expected:
                raise ValueError(f"Unexpected omega_memory shape {memory.shape}; expected {expected}")
            if mask.shape != (self._omega_horizon,):
                raise ValueError(f"Unexpected omega_memory_mask shape {mask.shape}; expected {(self._omega_horizon,)}")
            self._omega_history.clear()
            self._omega_history.extend(np.asarray(memory[mask], dtype=np.float32))
            return

        omega_images = obs.get("omega_images")
        if omega_images is None:
            raise ValueError(
                "Omega add_buffer requires 'omega_images' with all three synchronized views "
                "or precomputed 'omega_memory'."
            )
        if self._omega_scene_encoder is None:
            raise ValueError("Cannot encode omega_images without a frozen Omega scene encoder")
        images = self._extract_camera_sequences(omega_images, self._omega_scene_encoder.camera_order)
        for index in range(len(next(iter(images.values())))):
            snapshot = {camera: values[index] for camera, values in images.items()}
            encoded = self._omega_scene_encoder.encode(snapshot)
            self._omega_history.append(np.asarray(encoded, dtype=np.float32))

    @classmethod
    def _extract_camera_sequences(cls, images: dict, camera_order: Sequence[str]) -> dict[str, np.ndarray]:
        if not isinstance(images, dict):
            raise TypeError("omega_images must be a mapping from camera names to image sequences")
        extracted: dict[str, np.ndarray] = {}
        for camera in camera_order:
            value = images.get(camera)
            if value is None:
                raise ValueError(f"Missing Omega history camera {camera!r}")
            array = np.asarray(value)
            if array.ndim == 3:
                array = array[None, ...]
            if array.ndim != 4:
                raise ValueError(f"Omega history camera {camera!r} must be [N,H,W,C] or one image, got {array.shape}")
            extracted[camera] = array
        lengths = {array.shape[0] for array in extracted.values()}
        if len(lengths) != 1:
            raise ValueError(f"Omega history camera sequences must have equal lengths, got {sorted(lengths)}")
        return extracted

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata
