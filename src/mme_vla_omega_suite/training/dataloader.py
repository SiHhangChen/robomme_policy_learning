"""
We implemented our own data loader,
which can be 5-10x faster than LeRobot dataloader and can avoid memory explosion issue
"""

from collections.abc import Sequence
import logging

import jax

from mme_vla_omega_suite.models.integration.omega_observation import OmegaObservation
from mme_vla_omega_suite.training.dataset import OmegaRoboMMEDataset
from openpi.models import model as _model
from openpi.training.data_loader import DataLoader
from openpi.training.data_loader import TorchDataLoader
from openpi.training.data_loader import transform_dataset


class DataLoaderImpl(DataLoader):
    def __init__(self, data_config, data_loader: TorchDataLoader):
        self._data_config = data_config
        self._data_loader = data_loader

    def data_config(self):
        return self._data_config

    def __iter__(self):
        for batch in self._data_loader:
            yield OmegaObservation.from_dict(batch), batch["actions"]


def create_data_loader(
    dataset_path: str,
    data_config,
    *,
    omega_cache_path: str,
    omega_memory_horizon: int,
    omega_stride: int,
    omega_num_views: int = 3,
    omega_camera_order: Sequence[str] | None = None,
    omega_layer_indices: Sequence[int] = (4, 11, 17, 23),
    omega_tokens_per_view: int = 17,
    omega_input_dim: int = 2048,
    action_horizon: int,
    batch_size: int,
    sharding: jax.sharding.Sharding | None = None,
    skip_norm_stats: bool = False,
    shuffle: bool = False,
    num_batches: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader[tuple[OmegaObservation, _model.Actions]]:
    dataset = OmegaRoboMMEDataset(
        dataset_path=dataset_path,
        data_config=data_config,
        omega_cache_path=omega_cache_path,
        omega_memory_horizon=omega_memory_horizon,
        omega_stride=omega_stride,
        action_horizon=action_horizon,
        omega_num_views=omega_num_views,
        omega_camera_order=omega_camera_order,
        omega_num_layers=len(tuple(omega_layer_indices)),
        omega_layer_indices=omega_layer_indices,
        omega_tokens_per_view=omega_tokens_per_view,
        omega_input_dim=omega_input_dim,
    )

    dataset = transform_dataset(dataset, data_config, skip_norm_stats=skip_norm_stats)

    local_batch_size = batch_size // jax.process_count()
    logging.info(f"local_batch_size: {local_batch_size}")

    data_loader = TorchDataLoader(
        dataset,
        local_batch_size=local_batch_size,
        sharding=sharding,
        shuffle=shuffle,
        num_batches=num_batches,
        num_workers=num_workers,
        seed=seed,
        framework="jax",
    )

    return DataLoaderImpl(data_config, data_loader)
