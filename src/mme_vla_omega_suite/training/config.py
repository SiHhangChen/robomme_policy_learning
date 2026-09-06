"""Independent training configs for the Omega-conditioned RoboMME suite."""

from __future__ import annotations

import dataclasses
import difflib
import json
import os
import pathlib

import numpy as np
from typing_extensions import override
import tyro

from mme_vla_omega_suite.models.integration.omega_pi0 import OmegaPi0Config
from mme_vla_omega_suite.policies.omega_transforms import OmegaRoboMMEInputs
from mme_vla_omega_suite.policies.omega_transforms import OmegaRoboMMEOutputs
from mme_vla_omega_suite.policies.omega_transforms import OmegaSliceState
from mme_vla_suite.training.config import AssetsConfig
from mme_vla_suite.training.config import DataConfig
from mme_vla_suite.training.config import DataConfigFactory
from mme_vla_suite.training.config import ModelTransformFactory
from mme_vla_suite.training.config import TrainConfig
from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import normalize as _normalize
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders

DEFAULT_DATASET_PATH = os.getenv(
    "OMEGA_DATASET_PATH", "/data1/shared_workspace/chensihang/dataset/membench/wa05_200seeds_v061"
)
DEFAULT_OMEGA_CACHE_PATH = os.getenv(
    "OMEGA_CACHE_PATH", "/data1/shared_workspace/chensihang/dataset/baseline/vggt-omega/wa05_vggt_cache"
)
TS01_DATASET_PATH = "/data1/shared_workspace/chensihang/dataset/membench/ts01_200seeds_v061"
TS01_OMEGA_CACHE_PATH = "/data1/shared_workspace/chensihang/dataset/baseline/vggt-omega/ts01_vggt_cache"
WA01_DATASET_PATH = os.getenv(
    "OMEGA_WA01_DATASET_PATH", "/data1/shared_workspace/chensihang/dataset/membench/wa01_200seeds_v061"
)
WA01_OMEGA_CACHE_PATH = os.getenv(
    "OMEGA_WA01_CACHE_PATH", "/data1/shared_workspace/chensihang/dataset/baseline/vggt-omega/wa01_vggt_cache"
)
DEFAULT_PI05_PARAMS_PATH = os.getenv(
    "PI05_BASE_PARAMS_PATH",
    "/data1/shared_workspace/tangzhipeng/ckpts/openpi/openpi_assets/pi05_base/params",
)


@dataclasses.dataclass(frozen=True)
class OmegaRoboMMEDataConfig(DataConfigFactory):
    """MemBench LeRobot transforms with the additional Omega cache fields."""

    state_keep_dim: int = 30
    action_sequence_keys: tuple[str, ...] = ("action",)

    @staticmethod
    def _repack_transform() -> _transforms.Group:
        return _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "agentview_left": "observation.images.robot0_agentview_left",
                            "agentview_right": "observation.images.robot0_agentview_right",
                            "eye_in_hand": "observation.images.robot0_eye_in_hand",
                        },
                        "state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                        "omega_memory": "omega_memory",
                        "omega_memory_mask": "omega_memory_mask",
                    }
                )
            ]
        )

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[OmegaSliceState(self.state_keep_dim), OmegaRoboMMEInputs(model_type=model_config.model_type)],
            outputs=[OmegaRoboMMEOutputs(action_dim=13)],
        )
        base_config = self.create_base_config(assets_dirs, model_config)
        # Prefer statistics generated from this local dataset. This keeps the
        # Omega experiment independent from the old RoboMME pkl statistics.
        dataset_root = pathlib.Path(self.repo_id).expanduser()
        if base_config.norm_stats is None and dataset_root.is_dir():
            stats_path = dataset_root / "meta" / "stats.json"
            if stats_path.is_file():
                payload = json.loads(stats_path.read_text(encoding="utf-8"))
                payload = payload.get("norm_stats", payload)

                def make_stats(key: str, dim: int) -> _normalize.NormStats:
                    values = payload[key]
                    return _normalize.NormStats(
                        mean=np.asarray(values["mean"], dtype=np.float32).reshape(-1)[:dim],
                        std=np.asarray(values["std"], dtype=np.float32).reshape(-1)[:dim],
                        q01=np.asarray(values["q01"], dtype=np.float32).reshape(-1)[:dim],
                        q99=np.asarray(values["q99"], dtype=np.float32).reshape(-1)[:dim],
                    )

                base_config = dataclasses.replace(
                    base_config,
                    norm_stats={
                        "state": make_stats("observation.state", self.state_keep_dim),
                        "actions": make_stats("action", 13),
                    },
                )
        return dataclasses.replace(
            base_config,
            repack_transforms=self._repack_transform(),
            data_transforms=data_transforms,
            model_transforms=ModelTransformFactory()(model_config),
            action_sequence_keys=self.action_sequence_keys,
            prompt_from_task=False,
        )


@dataclasses.dataclass(frozen=True)
class OmegaTrainConfig(TrainConfig):
    """TrainConfig carrying the external frozen Omega cache location."""

    dataset_path: str = DEFAULT_DATASET_PATH
    omega_cache_path: str = DEFAULT_OMEGA_CACHE_PATH
    omega_stride: int = 50
    omega_repo: str = os.getenv("OMEGA_REPO", "")
    omega_checkpoint: str = os.getenv("OMEGA_CHECKPOINT", "")
    omega_device: str = os.getenv("OMEGA_DEVICE", "cuda:0")

    def __post_init__(self) -> None:
        super().__post_init__()
        # Keep dataset statistics and samples tied together when the dataset
        # path is overridden from the command line.
        if self.data.repo_id != self.dataset_path:
            object.__setattr__(self, "data", dataclasses.replace(self.data, repo_id=self.dataset_path))


_TS01_MODEL = OmegaPi0Config(
    pi05=True,
    action_horizon=50,
    max_token_len=200,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    omega_memory_horizon=8,
    omega_num_views=3,
    omega_camera_order=("agentview_left", "agentview_right", "eye_in_hand"),
    omega_layer_indices=(4, 11, 17, 23),
    omega_tokens_per_view=17,
    omega_input_dim=2048,
    omega_output_dim=1024,
)

_WA01_MODEL = OmegaPi0Config(
    pi05=True,
    action_horizon=50,
    max_token_len=200,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    omega_memory_horizon=8,
    omega_num_views=3,
    omega_camera_order=("agentview_left", "agentview_right", "eye_in_hand"),
    omega_layer_indices=(4, 11, 17, 23),
    omega_tokens_per_view=17,
    omega_input_dim=2048,
    omega_output_dim=1024,
)

_WA01_ACTION_CONDITION_MODEL = dataclasses.replace(_WA01_MODEL, omega_conditioning="action")

_CONFIGS = [
    OmegaTrainConfig(
        name="mme_vla_omega_wa01_action_condition",
        model=_WA01_ACTION_CONDITION_MODEL,
        data=OmegaRoboMMEDataConfig(
            repo_id=WA01_DATASET_PATH,
            assets=AssetsConfig(asset_id="wa01_200seeds_v061"),
        ),
        dataset_path=WA01_DATASET_PATH,
        omega_cache_path=WA01_OMEGA_CACHE_PATH,
        omega_stride=50,
        batch_size=48,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=_WA01_ACTION_CONDITION_MODEL.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(DEFAULT_PI05_PARAMS_PATH),
        num_train_steps=60_000,
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=None,
        fsdp_devices=3,
    ),
    OmegaTrainConfig(
        name="mme_vla_omega_wa01_action_modulation",
        model=_WA01_MODEL,
        data=OmegaRoboMMEDataConfig(
            repo_id=WA01_DATASET_PATH,
            assets=AssetsConfig(asset_id="wa01_200seeds_v061"),
        ),
        dataset_path=WA01_DATASET_PATH,
        omega_cache_path=WA01_OMEGA_CACHE_PATH,
        omega_stride=50,
        batch_size=48,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=_WA01_MODEL.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(DEFAULT_PI05_PARAMS_PATH),
        num_train_steps=60_000,
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=None,
        fsdp_devices=3,
    ),
    OmegaTrainConfig(
        name="mme_vla_omega_action_modulation",
        model=OmegaPi0Config(
            pi05=True,
            action_horizon=20,
            max_token_len=200,
            omega_memory_horizon=8,
            omega_num_views=3,
            omega_camera_order=("agentview_left", "agentview_right", "eye_in_hand"),
            omega_layer_indices=(4, 11, 17, 23),
            omega_tokens_per_view=17,
            omega_input_dim=2048,
            omega_output_dim=1024,
        ),
        data=OmegaRoboMMEDataConfig(
            repo_id=DEFAULT_DATASET_PATH,
            assets=AssetsConfig(asset_id="wa05_200seeds_v061"),
        ),
        batch_size=64,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=OmegaPi0Config().get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(DEFAULT_PI05_PARAMS_PATH),
        num_train_steps=80_000,
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=0.999,
        fsdp_devices=4,
    ),
    OmegaTrainConfig(
        name="mme_vla_omega_ts01_action_modulation",
        model=_TS01_MODEL,
        data=OmegaRoboMMEDataConfig(
            repo_id=TS01_DATASET_PATH,
            assets=AssetsConfig(asset_id="ts01_200seeds_v061"),
        ),
        dataset_path=TS01_DATASET_PATH,
        omega_cache_path=TS01_OMEGA_CACHE_PATH,
        omega_stride=50,
        batch_size=48,
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=10_000,
            peak_lr=5e-5,
            decay_steps=100_000,
            decay_lr=5e-5,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        freeze_filter=_TS01_MODEL.get_freeze_filter(),
        weight_loader=weight_loaders.CheckpointWeightLoader(DEFAULT_PI05_PARAMS_PATH),
        num_train_steps=60_000,
        save_interval=10_000,
        keep_period=10_000,
        num_workers=4,
        ema_decay=None,
        fsdp_devices=3,
    ),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> OmegaTrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> OmegaTrainConfig:
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT, n=1, cutoff=0.0)
        suggestion = f" Did you mean '{closest[0]}'?" if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{suggestion}")
    return _CONFIGS_DICT[config_name]
