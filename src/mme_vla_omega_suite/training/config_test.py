from mme_vla_omega_suite.training.config import get_config


def test_ts01_training_config_matches_prefix_baseline_contract():
    config = get_config("mme_vla_omega_ts01_action_modulation")

    assert config.dataset_path.endswith("/ts01_200seeds_v061")
    assert config.data.repo_id == config.dataset_path
    assert config.data.assets.asset_id == "ts01_200seeds_v061"
    assert config.omega_cache_path.endswith("/ts01_vggt_cache")
    assert config.model.paligemma_variant == "gemma_2b_lora"
    assert config.model.action_expert_variant == "gemma_300m_lora"
    assert config.model.action_horizon == 50
    assert config.model.omega_memory_horizon == 8
    assert config.model.omega_num_views == 3
    assert config.model.omega_layer_indices == (4, 11, 17, 23)
    assert config.omega_stride == 50
    assert config.batch_size == 48
    assert config.num_train_steps == 60_000
    assert config.fsdp_devices == 3
    assert config.ema_decay is None


def test_wa01_training_config_matches_action_modulation_contract():
    config = get_config("mme_vla_omega_wa01_action_modulation")

    assert config.dataset_path.endswith("/wa01_200seeds_v061")
    assert config.data.repo_id == config.dataset_path
    assert config.data.assets.asset_id == "wa01_200seeds_v061"
    assert config.omega_cache_path.endswith("/wa01_vggt_cache")
    assert config.model.paligemma_variant == "gemma_2b_lora"
    assert config.model.action_expert_variant == "gemma_300m_lora"
    assert config.model.action_horizon == 50
    assert config.model.omega_memory_horizon == 8
    assert config.model.omega_num_views == 3
    assert config.model.omega_layer_indices == (4, 11, 17, 23)
    assert config.omega_stride == 50
    assert config.batch_size == 48
    assert config.num_train_steps == 60_000
    assert config.fsdp_devices == 3
    assert config.ema_decay is None


def test_wa01_action_condition_config_uses_pooled_adarms_injection():
    config = get_config("mme_vla_omega_wa01_action_condition")

    assert config.dataset_path.endswith("/wa01_200seeds_v061")
    assert config.omega_cache_path.endswith("/wa01_vggt_cache")
    assert config.model.omega_conditioning == "action"
    assert config.model.action_horizon == 50
    assert config.model.omega_memory_horizon == 8
    assert config.model.omega_layer_indices == (4, 11, 17, 23)
    assert config.batch_size == 48
    assert config.num_train_steps == 60_000
    assert config.fsdp_devices == 3
    assert config.ema_decay is None
