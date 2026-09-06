"""Read one real Omega training batch without initializing the model."""

from __future__ import annotations

import argparse

from mme_vla_omega_suite.training import config as omega_config
from mme_vla_omega_suite.training.dataloader import create_data_loader


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-name", default="mme_vla_omega_action_modulation")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    config = omega_config.get_config(args.config_name)
    data_config = config.data.create(config.assets_dirs, config.model)
    loader = create_data_loader(
        config.dataset_path,
        data_config,
        omega_cache_path=config.omega_cache_path,
        omega_memory_horizon=config.model.omega_memory_horizon,
        omega_stride=config.omega_stride,
        omega_num_views=config.model.omega_num_views,
        omega_camera_order=config.model.omega_camera_order,
        omega_layer_indices=config.model.omega_layer_indices,
        omega_tokens_per_view=config.model.omega_tokens_per_view,
        omega_input_dim=config.model.omega_input_dim,
        action_horizon=config.model.action_horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        num_batches=1,
    )
    observation, actions = next(iter(loader))
    print("dataset:", config.dataset_path)
    print("state:", observation.state.shape)
    print("actions:", actions.shape)
    print("images:", {key: value.shape for key, value in observation.images.items()})
    print("omega_memory:", observation.omega_memory.shape)
    print("omega_memory_mask:", observation.omega_memory_mask)


if __name__ == "__main__":
    main()
