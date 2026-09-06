import dataclasses
import logging
from typing import Any

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from mme_vla_omega_suite.models.integration import omega_gemma as _gemma
from mme_vla_omega_suite.models.integration.omega_observation import OmegaObservation
from mme_vla_omega_suite.models.integration.omega_observation import preprocess_observation
from mme_vla_omega_suite.shared.omega_memory import OmegaMemoryEncoder
from mme_vla_omega_suite.shared.omega_memory import flatten_omega_mask
from mme_vla_suite.models.integration.utils import get_config
from openpi.models.model import Actions
from openpi.models.model import BaseModel
from openpi.models.pi0 import posemb_sincos
from openpi.models.pi0_config import Pi0Config
import openpi.models.siglip as _siglip
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils

logger = logging.getLogger("history-pi0")


def _zero_initializer(key, shape, dtype=jnp.float32):
    """Keep an optional Omega action-condition branch inert at initialization."""
    del key
    return jnp.zeros(shape, dtype)


def make_attn_mask(input_mask, mask_ar, mask_na=None):
    """Adapted from pi0.py

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    mask_na: bool[B, N]
       [[0 0 0 1 1 1 0 0 0 ...]]: The 1s tokens can not attend to the first three 0s.
       this is used for vision tokens not attend to memory tokens, but action tokens can

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    if mask_na is not None:
        mask_na = jnp.broadcast_to(mask_na, input_mask.shape)
        mask_not_attend = jnp.logical_and(
            jnp.logical_or(mask_na[:, None, :], mask_na[:, :, None]),
            einops.repeat(jnp.cumsum(mask_na, axis=1) <= 0, "b k -> b s k", s=mask_na.shape[1]),
        )
        # print(mask_not_attend)
        return jnp.where(mask_not_attend, jnp.zeros((), dtype=jnp.bool_), jnp.logical_and(attn_mask, valid_mask))
    return jnp.logical_and(attn_mask, valid_mask)


@dataclasses.dataclass(frozen=True)
class OmegaPi0Config(Pi0Config):
    """Pi05 configuration for the isolated VGGT-Omega suite."""

    omega_memory_horizon: int = 8
    # Both PaliGemma and the Omega conditioning cache use the same synchronized
    # three-camera contract.  The Omega suite does not silently pad a missing
    # right view because that would make the policy and cache disagree.
    omega_num_views: int = 3
    omega_camera_order: tuple[str, ...] = (
        "agentview_left",
        "agentview_right",
        "eye_in_hand",
    )
    omega_layer_indices: tuple[int, ...] = (4, 11, 17, 23)
    omega_tokens_per_view: int = 17
    omega_input_dim: int = 2048
    omega_output_dim: int = 1024
    # ``modulation`` uses per-layer action-token cross-attention. ``action``
    # pools the encoded strict-past memory and adds it to Pi05 AdaRMS.
    omega_conditioning: str = "modulation"
    # Kept for compatibility with RoboMME's generic Pi05 transform factory.
    use_history: bool = True
    history_config: str | None = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.omega_memory_horizon <= 0:
            raise ValueError("omega_memory_horizon must be positive")
        if self.omega_num_views <= 0 or self.omega_tokens_per_view <= 0:
            raise ValueError("Omega view/token counts must be positive")
        if len(self.omega_camera_order) != self.omega_num_views:
            raise ValueError(
                "omega_camera_order must contain exactly omega_num_views names: "
                f"got {len(self.omega_camera_order)} names for {self.omega_num_views} views"
            )
        if len(set(self.omega_camera_order)) != len(self.omega_camera_order):
            raise ValueError("omega_camera_order must not contain duplicates")
        if not self.omega_layer_indices:
            raise ValueError("omega_layer_indices must not be empty")
        if self.omega_input_dim <= 0 or self.omega_output_dim <= 0:
            raise ValueError("Omega feature dimensions must be positive")
        if self.omega_conditioning not in {"modulation", "action"}:
            raise ValueError(
                "omega_conditioning must be either 'modulation' or 'action', "
                f"got {self.omega_conditioning!r}"
            )
        if self.omega_conditioning == "action" and not self.pi05:
            raise ValueError("omega_conditioning='action' requires a Pi05 model")

    @override
    def create(self, rng: at.KeyArrayLike) -> "OmegaPi0":
        return OmegaPi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[OmegaObservation, Actions]:
        base_obs_spec, action_spec = super().inputs_spec(batch_size=batch_size)
        with at.disable_typechecking():
            observation_spec = OmegaObservation.from_base_obs(
                base_obs_spec,
                omega_memory=jax.ShapeDtypeStruct(
                    [
                        batch_size,
                        self.omega_memory_horizon,
                        self.omega_num_views,
                        len(self.omega_layer_indices),
                        self.omega_tokens_per_view,
                        self.omega_input_dim,
                    ],
                    jnp.float32,
                ),
                omega_memory_mask=jax.ShapeDtypeStruct(
                    [batch_size, self.omega_memory_horizon],
                    jnp.bool_,
                ),
            )
        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze the pretrained vision tower while training Omega adapters.

        This mirrors RoboMME's Pi05 policy: the SigLIP tower stays frozen,
        while the language/action experts and all newly introduced Omega
        projection, cross-attention, and modulation parameters remain
        trainable.  LoRA variants retain their normal base-weight freeze
        semantics.
        """

        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.extend((gemma_params_filter, nnx.Not(action_expert_params_filter)))
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(action_expert_params_filter)
            has_lora = True
        if has_lora:
            filters.extend((nnx.Not(nnx_utils.PathRegex(".*lora.*")), nnx.Not(nnx_utils.PathRegex(".*mem.*"))))
            return nnx.Any(nnx.All(*filters), nnx_utils.PathRegex(".*img.*"))
        return nnx_utils.PathRegex(".*img.*")


class OmegaPi0(BaseModel):
    def __init__(self, config: OmegaPi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        paligemma_config = get_config(config.paligemma_variant)
        action_expert_config = get_config(config.action_expert_variant)
        if config.omega_output_dim != action_expert_config.width:
            raise ValueError(
                "omega_output_dim must equal the action expert width for cross-attention: "
                f"got {config.omega_output_dim}, expected {action_expert_config.width}"
            )

        self.config = config
        self.use_history = True
        self.history_config = None
        self.omega_conditioning = config.omega_conditioning
        self.integration_type = "modulation" if self.omega_conditioning == "modulation" else "action_condition"
        self.representation_type = "omega"

        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
                integration_type=self.integration_type,
            )
        )
        llm.lazy_init(
            rngs=rngs,
            method="init",
            use_adarms=[False, True] if config.pi05 else [False, False],
            mem_mods=[False, True],
        )

        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.omega_memory_encoder = OmegaMemoryEncoder(
            num_layers=len(config.omega_layer_indices),
            input_dim=config.omega_input_dim,
            output_dim=action_expert_config.width,
            rngs=rngs,
            dtype=config.dtype,
        )
        if self.omega_conditioning == "action":
            self.omega_action_proj = nnx.Linear(
                action_expert_config.width,
                action_expert_config.width,
                kernel_init=_zero_initializer,
                bias_init=_zero_initializer,
                rngs=rngs,
            )
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)

        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_memory(self, obs: OmegaObservation):
        if obs.omega_memory is None or obs.omega_memory_mask is None:
            raise ValueError("Omega memory and omega_memory_mask are required")
        tokens = self.omega_memory_encoder(obs.omega_memory)
        input_mask = flatten_omega_mask(
            obs.omega_memory_mask,
            num_views=self.config.omega_num_views,
            tokens_per_view=self.config.omega_tokens_per_view,
        )
        ar_mask = [False] * tokens.shape[1]
        na_mask = [False] * tokens.shape[1]
        return tokens, input_mask, ar_mask, na_mask, None

    @at.typecheck
    def embed_action_condition(
        self,
        memory_tokens: at.Float[at.Array, "b s emb"],
        memory_mask: at.Bool[at.Array, "b s"],
    ) -> at.Float[at.Array, "b emb"]:
        """Pool valid Omega tokens into the Pi05 action AdaRMS condition."""
        if self.omega_conditioning != "action":
            raise RuntimeError("Omega action condition requested but disabled")
        weights = memory_mask.astype(jnp.float32)
        denominator = jnp.maximum(jnp.sum(weights, axis=1, keepdims=True), 1.0)
        pooled = jnp.sum(memory_tokens.astype(jnp.float32) * weights[..., None], axis=1) / denominator
        return self.omega_action_proj(pooled)

    @at.typecheck
    def embed_prefix(
        self, obs: OmegaObservation
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Bool[at.Array, " s"],
        Any | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        na_mask = []

        if self.integration_type == "context":
            (
                mem_tokens,
                mem_input_mask,
                mem_ar_mask,
                mem_na_mask,
                stats,
            ) = self.embed_memory(obs)
            if mem_tokens is not None:
                tokens.append(mem_tokens)
                input_mask.append(mem_input_mask)
                ar_mask += mem_ar_mask
                na_mask += mem_na_mask
        else:
            stats = None

        # embed images
        for i, name in enumerate(obs.images):
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # image tokens attend to each other
            if i == 0:
                ar_mask += [True] + ([False] * (image_tokens.shape[1] - 1))
            else:
                ar_mask += [False] * image_tokens.shape[1]
            na_mask += [True] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if self.use_history and self.representation_type == "symbolic":
            tokenized_inputs = self.PaliGemma.llm(obs.symbolic_tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.symbolic_tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
            na_mask += [False] * tokenized_inputs.shape[1]

        elif obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
            na_mask += [False] * tokenized_inputs.shape[1]

        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        na_mask = jnp.array(na_mask)

        return tokens, input_mask, ar_mask, na_mask, stats

    @at.typecheck
    def embed_suffix(
        self,
        obs: OmegaObservation,
        noisy_actions: Actions,
        timestep: at.Float[at.Array, " b"],
        *,
        omega_condition: at.Float[at.Array, "b emb"] | None = None,
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        na_mask = []

        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]
            na_mask += [False]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
            if self.omega_conditioning == "action":
                if omega_condition is None:
                    raise ValueError("omega_condition is required when action conditioning is enabled")
                adarms_cond = adarms_cond + omega_condition.astype(adarms_cond.dtype)
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        na_mask += [False] * self.action_horizon
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)

        ar_mask = jnp.array(ar_mask)
        na_mask = jnp.array(na_mask)
        return tokens, input_mask, ar_mask, na_mask, adarms_cond

    @override
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: OmegaObservation,
        actions: Actions,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = preprocess_observation(preprocess_rng, observation, train=train)

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        # one big forward pass of prefix + suffix at once
        omega_condition = None
        if self.omega_conditioning == "action":
            memory_tokens, memory_mask, _, _, _ = self.embed_memory(observation)
            omega_condition = self.embed_action_condition(memory_tokens, memory_mask)
        prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, stats = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, suffix_na_mask, adarms_cond = self.embed_suffix(
            observation, x_t, time, omega_condition=omega_condition
        )

        if self.integration_type == "expert":
            mem_tokens, mem_input_mask, mem_ar_mask, mem_na_mask, stats = self.embed_memory(observation)
            mem_ar_mask = jnp.array(mem_ar_mask)
            mem_na_mask = jnp.array(mem_na_mask)
            input_mask = jnp.concatenate([mem_input_mask, prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([mem_ar_mask, prefix_ar_mask, suffix_ar_mask], axis=0)
            na_mask = jnp.concatenate([mem_na_mask, prefix_na_mask, suffix_na_mask], axis=0)
        else:
            input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
            ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
            na_mask = jnp.concatenate([prefix_na_mask, suffix_na_mask], axis=0)

        if self.use_history and self.representation_type != "symbolic":
            attn_mask = make_attn_mask(input_mask, ar_mask, na_mask)
        else:
            attn_mask = make_attn_mask(input_mask, ar_mask)

        positions = jnp.cumsum(input_mask, axis=1) - 1

        if self.integration_type == "expert":
            (mem_out, prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [mem_tokens, prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, None, adarms_cond],
            )
        elif self.integration_type == "modulation":
            mem_seq, mem_mask, _, _, stats = self.embed_memory(observation)
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
                mem_seq=[None, mem_seq],
                mem_mask=[None, mem_mask],
            )
        else:
            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [prefix_tokens, suffix_tokens],
                mask=attn_mask,
                positions=positions,
                adarms_cond=[None, adarms_cond],
            )

        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        # import pdb; pdb.set_trace()

        return jnp.mean(jnp.square(v_t - u_t), axis=-1), stats

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: OmegaObservation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> Actions:
        observation = preprocess_observation(None, observation, train=False)
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        omega_condition = None
        if self.omega_conditioning == "action":
            memory_tokens, memory_mask, _, _, _ = self.embed_memory(observation)
            omega_condition = self.embed_action_condition(memory_tokens, memory_mask)

        if self.integration_type == "expert":
            mem_tokens, mem_input_mask, mem_ar_mask, mem_na_mask, _ = self.embed_memory(observation)
            vlm_tokens, vlm_mask, vlm_ar_mask, vlm_na_mask, _ = self.embed_prefix(observation)
            mem_ar_mask = jnp.array(mem_ar_mask)
            mem_na_mask = jnp.array(mem_na_mask)
            prefix_mask = jnp.concatenate([mem_input_mask, vlm_mask], axis=1)
            prefix_ar_mask = jnp.concatenate([mem_ar_mask, vlm_ar_mask], axis=0)
            prefix_na_mask = jnp.concatenate([mem_na_mask, vlm_na_mask], axis=0)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm(
                [mem_tokens, vlm_tokens, None],
                mask=prefix_attn_mask,
                positions=positions,
            )

        elif self.integration_type == "modulation":
            prefix_tokens, prefix_mask, prefix_ar_mask, _, _ = self.embed_prefix(observation)
            prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
            mem_seq, mem_mask, _, _, _ = self.embed_memory(observation)

        else:
            prefix_tokens, prefix_mask, prefix_ar_mask, prefix_na_mask, _ = self.embed_prefix(observation)
            if self.integration_type == "context":
                prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask, prefix_na_mask)
            else:
                prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)

        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, _, adarms_cond = self.embed_suffix(
                observation,
                x_t,
                jnp.broadcast_to(time, batch_size),
                omega_condition=omega_condition,
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_mask.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            if self.integration_type == "expert":
                (mem_out, prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, None, adarms_cond],
                )
                assert mem_out is None
            elif self.integration_type == "modulation":
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                    mem_seq=[None, mem_seq],
                    mem_mask=[None, mem_mask],
                )
            else:
                (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                    [None, suffix_tokens],
                    mask=full_attn_mask,
                    positions=positions,
                    kv_cache=kv_cache,
                    adarms_cond=[None, adarms_cond],
                )

            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0

    def vision_encode(self, images: at.Float[at.Array, "k v 224 224 3"]):
        shape = images.shape
        assert shape[-3:] == (
            224,
            224,
            3,
        ), f"x.shape: {shape}, should be (B, 224, 224, 3)"

        x = einops.rearrange(images, "k v h w c -> (k v) h w c")
        out, _ = self.PaliGemma.img(x, train=False)
        return einops.rearrange(out, "(k v) p d -> k v p d", k=shape[0], v=shape[1])
