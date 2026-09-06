from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from mme_vla_omega_suite.shared.omega_memory import OmegaMemoryEncoder
from mme_vla_omega_suite.shared.omega_memory import flatten_omega_mask
from mme_vla_omega_suite.shared.omega_memory import gather_strict_past


def test_gather_strict_past_is_oldest_to_newest_and_excludes_current():
    tokens = (10 + np.arange(6, dtype=np.float16)).reshape(6, 1, 1, 1, 1)
    rows = {(3, decision): decision for decision in range(6)}

    memory, mask = gather_strict_past(
        tokens,
        rows,
        episode=3,
        step=150,
        horizon=4,
        stride=50,
    )

    np.testing.assert_array_equal(memory[:, 0, 0, 0, 0], np.array([0, 10, 11, 12], dtype=np.float16))
    np.testing.assert_array_equal(mask, np.array([False, True, True, True]))


def test_gather_strict_past_left_pads_episode_warmup():
    tokens = (20 + np.arange(4, dtype=np.float16)).reshape(4, 1, 1, 1, 1)
    rows = {(0, decision): decision for decision in range(4)}

    memory, mask = gather_strict_past(
        tokens,
        rows,
        episode=0,
        step=50,
        horizon=4,
        stride=50,
    )

    np.testing.assert_array_equal(memory[:, 0, 0, 0, 0], np.array([0, 0, 0, 20], dtype=np.float16))
    np.testing.assert_array_equal(mask, np.array([False, False, False, True]))


def test_layer_fusion_preserves_history_view_token_order():
    encoder = OmegaMemoryEncoder(
        num_layers=2,
        input_dim=3,
        output_dim=5,
        rngs=nnx.Rngs(jax.random.key(0)),
        dtype=jnp.float32,
    )
    memory = jnp.arange(1 * 2 * 3 * 2 * 2 * 3, dtype=jnp.float32).reshape(1, 2, 3, 2, 2, 3)
    output = encoder(memory)
    assert output.shape == (1, 12, 5)
    # The flattened bank is H x V x T; every layer stays adjacent in the
    # shared projection's input channel dimension.
    fused = jnp.transpose(memory, (0, 1, 2, 4, 3, 5)).reshape(1, 12, 6)
    np.testing.assert_array_equal(
        np.asarray(encoder.fusion(fused)),
        np.asarray(output),
    )


def test_flatten_omega_mask_expands_each_history_slot():
    mask = jnp.asarray([[False, True]])
    np.testing.assert_array_equal(
        np.asarray(flatten_omega_mask(mask, num_views=3, tokens_per_view=2)),
        np.asarray([[False, False, False, False, False, False, True, True, True, True, True, True]]),
    )
