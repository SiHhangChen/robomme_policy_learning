"""Checkpoint loading for the isolated Omega suite."""

from __future__ import annotations

import logging
import pathlib
from typing import Any

import jax.numpy as jnp

from mme_vla_omega_suite.policies.omega_policy import OmegaPolicy
from mme_vla_omega_suite.policies.omega_scene_encoder import FrozenVGGTOmegaSceneEncoder
from mme_vla_omega_suite.training import config as _config
from openpi import transforms
from openpi.models import model as _model
from openpi.training import checkpoints as _checkpoints


def create_trained_policy(
    train_config: _config.OmegaTrainConfig,
    checkpoint_dir: pathlib.Path | str,
    seed: int = 42,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
) -> OmegaPolicy:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    repack_transforms = repack_transforms or transforms.Group()
    logging.info("Loading Omega model from %s", checkpoint_dir)
    model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    scene_encoder = None
    if train_config.omega_repo and train_config.omega_checkpoint:
        scene_encoder = FrozenVGGTOmegaSceneEncoder(
            omega_repo=train_config.omega_repo,
            checkpoint=train_config.omega_checkpoint,
            device=train_config.omega_device,
            camera_order=train_config.model.omega_camera_order,
            output_tokens_per_view=train_config.model.omega_tokens_per_view,
            cached_layer_indices=train_config.model.omega_layer_indices,
        )

    return OmegaPolicy(
        model,
        seed=seed,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        omega_scene_encoder=scene_encoder,
    )
