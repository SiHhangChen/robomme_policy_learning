"""Omega cache validation and layer-fusion encoding."""

from __future__ import annotations

from flax import nnx
import jax.numpy as jnp
import numpy as np

import openpi.shared.array_typing as at


class OmegaMemoryEncoder(nnx.Module):
    """Fuse the cached Omega layers into action-expert-width memory tokens.

    The projection is shared across every history slot, camera view, and
    camera/register token.  Layer identity is represented by the fixed channel
    concatenation order supplied by the cache metadata.
    """

    def __init__(
        self,
        *,
        num_layers: int,
        input_dim: int,
        output_dim: int,
        rngs: nnx.Rngs,
        dtype: at.DTypeLike = jnp.bfloat16,
    ) -> None:
        if num_layers <= 0 or input_dim <= 0 or output_dim <= 0:
            raise ValueError("Omega memory dimensions must be positive")
        self.num_layers = int(num_layers)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.fusion = nnx.Linear(
            self.num_layers * self.input_dim,
            self.output_dim,
            rngs=rngs,
            dtype=dtype,
        )

    def __call__(
        self,
        memory: at.Float[at.Array, "b h v l t d"],
    ) -> at.Float[at.Array, "b s e"]:
        if memory.ndim != 6:
            raise ValueError(f"Expected Omega memory [B,H,V,L,T,D], got {memory.shape}")
        batch, history, views, layers, tokens, dim = memory.shape
        if layers != self.num_layers or dim != self.input_dim:
            raise ValueError(
                "Omega cache dimensions do not match encoder: "
                f"expected L,D=({self.num_layers},{self.input_dim}), got ({layers},{dim})"
            )

        # [B,H,V,L,T,D] -> [B,H,V,T,L,D] -> [B,H,V,T,L*D].
        fused = jnp.transpose(memory, (0, 1, 2, 4, 3, 5))
        fused = fused.reshape(batch, history, views, tokens, layers * dim)
        fused = fused.reshape(batch, history * views * tokens, layers * dim)
        return self.fusion(fused)


def flatten_omega_mask(
    slot_mask: at.Bool[at.Array, "b h"],
    *,
    num_views: int,
    tokens_per_view: int,
) -> at.Bool[at.Array, "b s"]:
    """Expand one validity bit per history slot to the flattened token bank."""

    if slot_mask.ndim != 2:
        raise ValueError(f"Expected Omega slot mask [B,H], got {slot_mask.shape}")
    if num_views <= 0 or tokens_per_view <= 0:
        raise ValueError("Omega view/token counts must be positive")
    return jnp.repeat(slot_mask, int(num_views) * int(tokens_per_view), axis=1)


def gather_strict_past(
    tokens: np.ndarray,
    row_lookup: dict[tuple[int, int], int],
    *,
    episode: int,
    step: int,
    horizon: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather fixed-size oldest-to-newest history without current-step leakage."""

    if tokens.ndim != 5:
        raise ValueError(f"Expected cache [N,V,L,T,D], got {tokens.shape}")
    if horizon <= 0 or stride <= 0:
        raise ValueError("Omega horizon and stride must be positive")
    current_decision = int(step) // int(stride)
    memory = np.zeros((horizon, *tokens.shape[1:]), dtype=tokens.dtype)
    mask = np.zeros((horizon,), dtype=np.bool_)
    for slot, decision in enumerate(range(current_decision - horizon, current_decision)):
        if decision < 0:
            continue
        row = row_lookup.get((int(episode), decision))
        if row is None:
            raise ValueError(f"Missing Omega decision row {(episode, decision)}")
        memory[slot] = np.asarray(tokens[row], dtype=tokens.dtype)
        mask[slot] = True
    return memory, mask
