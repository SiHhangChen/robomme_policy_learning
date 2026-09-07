import dataclasses
import functools
import logging
import platform
from typing import Any

import etils.epath as epath
import flax.nnx as nnx
from flax.training import common_utils
import flax.traverse_util as traverse_util
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import wandb

from mme_vla_omega_suite.models.integration.omega_observation import OmegaObservation
import mme_vla_omega_suite.models.integration.omega_pi0 as _model
import mme_vla_omega_suite.training.config as _config
import mme_vla_omega_suite.training.dataloader as _data_loader
import openpi.shared.array_typing as at
import openpi.shared.nnx_utils as nnx_utils
import openpi.training.checkpoints as _checkpoints
import openpi.training.optimizer as _optimizer
import openpi.training.sharding as sharding
import openpi.training.utils as training_utils
import openpi.training.weight_loaders as _weight_loaders


def init_logging():
    """Custom logging format for better readability."""
    level_mapping = {
        "DEBUG": "D",
        "INFO": "I",
        "WARNING": "W",
        "ERROR": "E",
        "CRITICAL": "C",
    }

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers[0].setFormatter(formatter)


def init_wandb(
    config: _config.OmegaTrainConfig,
    *,
    resuming: bool,
    log_code: bool = False,
    enabled: bool = True,
):
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")
    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
            entity="daiyp_umich",
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)

    if log_code:
        wandb.run.log_code(epath.Path(__file__).parent.parent)


def init_history_config(config: _config.OmegaTrainConfig):
    # Keep the original training lifecycle hook without writing a history
    # config that belongs to the legacy RoboMME suite.
    return None


def _load_weights_and_validate(loader: _weight_loaders.WeightLoader, params_shape: at.Params) -> at.Params:
    """Loads and validates the weights. Returns a loaded subset of the weights."""
    loaded_params = loader.load(params_shape)
    at.check_pytree_equality(expected=params_shape, got=loaded_params, check_shapes=True, check_dtypes=True)

    # Remove jax.ShapeDtypeStruct from the loaded params. This makes sure that only the loaded params are returned.
    return traverse_util.unflatten_dict(
        {k: v for k, v in traverse_util.flatten_dict(loaded_params).items() if not isinstance(v, jax.ShapeDtypeStruct)}
    )


def params_split(params, trainable_filter):
    memory_filter = params.filter(trainable_filter).filter(nnx.All(nnx.Param, nnx_utils.PathRegex(".*mem.*")))
    non_memory_filter = params.filter(trainable_filter).filter(
        nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*mem.*")))
    )
    return memory_filter, non_memory_filter


@at.typecheck
def init_train_state(
    config: _config.OmegaTrainConfig,
    init_rng: at.KeyArrayLike,
    mesh: jax.sharding.Mesh,
    *,
    resume: bool,
) -> tuple[training_utils.TrainState, Any]:
    tx = _optimizer.create_optimizer(config.optimizer, config.lr_schedule, weight_decay_mask=None)

    def init(rng: at.KeyArrayLike, partial_params: at.Params | None = None) -> training_utils.TrainState:
        rng, model_rng = jax.random.split(rng)
        # initialize the model (and its parameters).
        model = config.model.create(model_rng)

        # Merge the partial params into the model.
        if partial_params is not None:
            graphdef, state = nnx.split(model)
            state.replace_by_pure_dict(partial_params)
            model = nnx.merge(graphdef, state)

        params = nnx.state(model)
        # Convert frozen params to bfloat16.
        params = nnx_utils.state_map(
            params,
            config.freeze_filter,
            lambda p: p.replace(p.value.astype(jnp.bfloat16)),
        )

        logging.info(f"Total Model Size: {sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1024 / 1024} MB")
        logging.info(
            f"Trainable Model Size: {sum(x.size for x in jax.tree_util.tree_leaves(params.filter(config.trainable_filter))) / 1024 / 1024} MB"
        )

        memory_params, non_memory_params = params_split(params, config.trainable_filter)
        logging.info(
            f"Memory-related  Size: {sum(x.size for x in jax.tree_util.tree_leaves(memory_params)) / 1024 / 1024} MB"
        )
        logging.info(
            f"Non-Memory Size: {sum(x.size for x in jax.tree_util.tree_leaves(non_memory_params)) / 1024 / 1024} MB"
        )

        return training_utils.TrainState(
            step=0,
            params=params,
            model_def=nnx.graphdef(model),
            tx=tx,
            opt_state=tx.init(params.filter(config.trainable_filter)),
            ema_decay=config.ema_decay,
            ema_params=None if config.ema_decay is None else params,
        )

    train_state_shape = jax.eval_shape(init, init_rng)
    state_sharding = sharding.fsdp_sharding(train_state_shape, mesh, log=True)

    if resume:
        # The checkpoint manager restores the latest complete TrainState after
        # initialization. Do not load only params here: that would reset the
        # optimizer, EMA, and step counter.
        return train_state_shape, state_sharding

    partial_params = _load_weights_and_validate(config.weight_loader, train_state_shape.params.to_pure_dict())

    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # Initialize the train state and mix in the partial params.
    train_state = jax.jit(
        init,
        donate_argnums=(1,),  # donate the partial params buffer.
        in_shardings=replicated_sharding,
        out_shardings=state_sharding,
    )(init_rng, partial_params)

    return train_state, state_sharding


@at.typecheck
def train_step(
    config: _config.OmegaTrainConfig,
    rng: at.KeyArrayLike,
    state: training_utils.TrainState,
    batch: tuple[OmegaObservation, _model.Actions],
) -> tuple[training_utils.TrainState, dict[str, at.Array], Any]:
    model = nnx.merge(state.model_def, state.params)
    model.train()

    @at.typecheck
    def loss_fn(
        model: _model.OmegaPi0,
        rng: at.KeyArrayLike,
        observation: OmegaObservation,
        actions: _model.Actions,
    ):
        chunked_loss, stats = model.compute_loss(rng, observation, actions, train=True)
        return jnp.mean(chunked_loss), stats

    train_rng = jax.random.fold_in(rng, state.step)
    observation, actions = batch

    # Filter out frozen params.
    diff_state = nnx.DiffState(0, config.trainable_filter)
    (loss, stats), grads = nnx.value_and_grad(loss_fn, argnums=diff_state, has_aux=True)(
        model, train_rng, observation, actions
    )

    params = state.params.filter(config.trainable_filter)
    updates, new_opt_state = state.tx.update(grads, state.opt_state, params)
    new_params = optax.apply_updates(params, updates)

    # Update the model in place and return the new full state.
    nnx.update(model, new_params)
    new_params = nnx.state(model)

    new_state = dataclasses.replace(state, step=state.step + 1, params=new_params, opt_state=new_opt_state)

    if state.ema_decay is not None:
        new_state = dataclasses.replace(
            new_state,
            ema_params=jax.tree.map(
                lambda old, new: state.ema_decay * old + (1 - state.ema_decay) * new,
                state.ema_params,
                new_params,
            ),
        )

    # Filter out params that aren't kernels.
    kernel_params = nnx.state(
        model,
        nnx.All(
            nnx.Param,
            nnx.Not(nnx_utils.PathRegex(".*/(bias|scale|pos_embedding|input_embedding)")),
            lambda _, x: x.value.ndim > 1,
        ),
    )

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "param_norm": optax.global_norm(kernel_params),
        "llm_grad_norm": optax.global_norm(grads.PaliGemma.llm),
    }
    if config.model.use_history and hasattr(grads, "mem_encoder"):
        info["mem_enc_norm"] = optax.global_norm(grads.mem_encoder)

    return new_state, info, stats


def get_stats(stats_dict) -> dict[str, at.Array]:
    mask = stats_dict["mask"]
    if mask.ndim == 2:
        batch_size, sequence_length = mask.shape  # for rmt
    else:
        batch_size, _, sequence_length = mask.shape  # for ttt
        stats_dict = {k: v.mean(axis=1) for k, v in stats_dict.items()}

    dic = {}
    for key in stats_dict:
        dic[key] = [[] for _ in range(sequence_length)]

    mask = stats_dict["mask"]
    for batch_idx in range(batch_size):
        for step_idx in range(sequence_length):
            if mask[batch_idx, step_idx]:
                for k, v in stats_dict.items():
                    dic[k][step_idx].append(v[batch_idx, step_idx])

    stats = {}
    for k, v in dic.items():
        stats[k] = np.zeros((sequence_length,))
        for step in range(sequence_length):
            if len(v[step]) > 0:
                stats[k][step] = np.array(v[step]).mean()
    return stats


def main(config: _config.OmegaTrainConfig):
    init_logging()
    logging.info(f"Running on: {platform.node()}")
    logging.info(f"TrainConfig: {config}")

    if config.batch_size % jax.device_count() != 0:
        raise ValueError(
            f"Batch size {config.batch_size} must be divisible by the number of devices {jax.device_count()}."
        )

    jax.config.update(
        "jax_compilation_cache_dir",
        str(epath.Path(f"~/.cache/jax_{config.exp_name}").expanduser()),
    )

    rng = jax.random.key(config.seed)
    train_rng, init_rng = jax.random.split(rng)

    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    checkpoint_manager, resuming = _checkpoints.initialize_checkpoint_dir(
        config.checkpoint_dir,
        keep_period=config.keep_period,
        overwrite=config.overwrite,
        resume=config.resume,
    )
    init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
    init_history_config(config)
    data_config = config.data.create(config.assets_dirs, config.model)
    logging.info(f"data_config: {data_config}")

    data_loader = _data_loader.create_data_loader(
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
        sharding=data_sharding,
        shuffle=True,
        action_horizon=config.model.action_horizon,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        seed=config.seed,
    )
    train_state, train_state_sharding = init_train_state(config, init_rng, mesh, resume=resuming)
    if resuming:
        restore_step = config.resum_ckpt_id
        logging.info(
            "Resuming Omega training from %s checkpoint step",
            "latest" if restore_step is None else restore_step,
        )
        train_state = _checkpoints.restore_state(
            checkpoint_manager,
            train_state,
            data_loader,
            step=restore_step,
        )
    jax.block_until_ready(train_state)
    logging.info(
        f"Initialized train state:\n{training_utils.array_tree_to_info(train_state.params.filter(nnx.All(nnx.Param)))}"
    )

    start_step = int(train_state.step)
    data_iter = None
    if start_step < config.num_train_steps:
        # TorchDataLoader's shuffle generator is seeded at construction time.
        # Recreate only one iterator and consume the already-trained batches so
        # a resumed run follows the same sample order as an uninterrupted run.
        data_iter = iter(data_loader)
        if start_step:
            logging.info("Skipping %d already-consumed data batches for resume", start_step)
            for _ in range(start_step):
                next(data_iter)
            logging.info("Finished aligning data loader at step %d", start_step)
        batch = next(data_iter)
        logging.info(f"Initialized data loader at step {start_step}:\n{training_utils.array_tree_to_info(batch)}")

        # Log one batch for a visual sanity check without changing the loader
        # position or the training step.
        images_to_log = [
            wandb.Image(np.concatenate([np.array(img[i]) for img in batch[0].images.values()], axis=1))
            for i in range(min(1, len(next(iter(batch[0].images.values())))))
        ]
        wandb.log({"camera_views": images_to_log}, step=start_step)

    ptrain_step = jax.jit(
        functools.partial(train_step, config),
        in_shardings=(replicated_sharding, train_state_sharding, data_sharding),
        out_shardings=(train_state_sharding, replicated_sharding, replicated_sharding),
        donate_argnums=(1,),
    )

    pbar = tqdm.tqdm(
        range(start_step, config.num_train_steps),
        initial=start_step,
        total=config.num_train_steps,
        dynamic_ncols=True,
    )

    infos = []
    for step in pbar:
        with sharding.set_mesh(mesh):
            train_state, info, stats = ptrain_step(train_rng, train_state, batch)
        infos.append(info)
        if step % config.log_interval == 0:
            stacked_infos = common_utils.stack_forest(infos)
            reduced_info = jax.device_get(jax.tree.map(jnp.mean, stacked_infos))

            info_str = ", ".join(f"{k}={v:.4f}" for k, v in reduced_info.items())
            pbar.write(f"Step {step}: {info_str}")
            wandb.log(reduced_info, step=step)
            infos = []

        batch = next(data_iter)

        completed_step = step + 1
        if (completed_step % config.save_interval == 0 and completed_step > start_step) or completed_step == config.num_train_steps:
            _checkpoints.save_state(checkpoint_manager, train_state, data_loader, completed_step)

    logging.info("Waiting for checkpoint manager to finish")
    checkpoint_manager.wait_until_finished()


if __name__ == "__main__":
    main(_config.cli())
