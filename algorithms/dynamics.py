from collections import namedtuple
from dataclasses import dataclass, asdict
import os
import pickle
import warnings
from types import SimpleNamespace
from typing import Optional

from flax.core import frozen_dict
import flax.linen as nn
from flax.training.train_state import TrainState
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax
import tyro
import wandb

from utils import (
    DATASET_SAMPLING_STRATEGY,
    load_minari_dataset,
    transitions_from_minari,
    collect_run_provenance,
)
from termination_fns import get_termination_fn


os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    dataset: str = "mujoco/halfcheetah/medium-v0"
    dataset_fraction: float = 1.0
    algorithm: str = "dynamics"
    eval_interval: int = 10_000
    # --- Logging ---
    log: bool = False
    wandb_project: str = "unifloral"
    wandb_team: str = "flair"
    wandb_group: str = "debug"
    model_path: str = "dynamics_models"
    # --- Generic optimization ---
    lr: float = 0.001
    batch_size: int = 256
    # --- Dynamics training ---
    n_layers: int = 4
    layer_size: int = 200
    num_ensemble: int = 7
    num_elites: int = 5
    num_epochs: int = 400
    logvar_diff_coef: float = 0.01
    weight_decay: float = 2.5e-5
    validation_split: float = 0.2
    precompute_term_stats: bool = False


r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Preliminaries
"""

Transition = namedtuple(
    "Transition", "obs action reward next_obs next_action done next_action_valid",
    defaults=(True,),
)


class SingleDynamicsModel(nn.Module):
    obs_dim: int
    n_layers: int
    layer_size: int

    @nn.compact
    def __call__(self, delta_obs_action):
        x = delta_obs_action
        for _ in range(self.n_layers):
            x = nn.relu(nn.Dense(self.layer_size)(x))
        obs_reward_stats = nn.Dense(2 * (self.obs_dim + 1))(x)
        return obs_reward_stats


class EnsembleDynamicsModel(nn.Module):
    obs_dim: int
    action_dim: int
    num_ensemble: int
    n_layers: int
    layer_size: int
    max_logvar_init: float = 0.5
    min_logvar_init: float = -10.0

    @nn.compact
    def __call__(self, obs_action):
        # --- Compute ensemble predictions ---
        batched_model = nn.vmap(
            SingleDynamicsModel,
            variable_axes={"params": 0},  # Parameters not shared between critics
            split_rngs={"params": True},  # Different initializations
            in_axes=None,
            out_axes=0,
            axis_size=self.num_ensemble,
        )
        ensemble = batched_model(
            obs_dim=self.obs_dim,
            n_layers=self.n_layers,
            layer_size=self.layer_size,
            name="ensemble",
        )
        output = ensemble(obs_action)
        pred_mean, logvar = jnp.split(output, 2, axis=-1)

        # --- Soft clamp log-variance ---
        max_logvar = self.param(
            "max_logvar",
            init_fn=lambda key: jnp.full((self.obs_dim + 1,), self.max_logvar_init),
        )
        min_logvar = self.param(
            "min_logvar",
            init_fn=lambda key: jnp.full((self.obs_dim + 1,), self.min_logvar_init),
        )
        logvar = max_logvar - nn.softplus(max_logvar - logvar)
        logvar = min_logvar + nn.softplus(logvar - min_logvar)
        return pred_mean, logvar


class EnsembleDynamics:
    """Wrapper class for ensemble dynamics model that handles sampling and rollouts."""

    def __init__(
        self,
        dynamics_model,
        params,
        elite_idxs,
        termination_fn,
        discrepancy: Optional[float] = None,
        min_r: Optional[float] = None,
    ):
        self.dynamics_model = dynamics_model
        self.num_ensemble = len(elite_idxs)
        self.dynamics_model.num_ensemble = self.num_ensemble
        self.termination_fn = termination_fn
        self.discrepancy = discrepancy
        self.min_r = min_r
        self.dataset = None

        # Discard params of non-elite models
        params = frozen_dict.unfreeze(params)
        ensemble_params = params["params"]["ensemble"]
        ensemble_params = jax.tree.map(lambda p: p[elite_idxs], ensemble_params)
        params["params"]["ensemble"] = ensemble_params
        self.params = frozen_dict.freeze(params)

    def make_rollout_fn(
        self,
        batch_size,
        rollout_length,
        step_penalty_coef=0.0,
        term_penalty_offset=None,
        threshold_coef=1.0,
    ):
        """Make buffer update function."""

        def _rollout_fn(rng, policy, rollout_buffer):
            @jax.vmap
            def _sample_step(carry, _):
                obs, rng = carry
                rng_step, rng_reset, rng = jax.random.split(rng, 3)
                transition = self._sample_transition(
                    rng_step,
                    policy,
                    obs,
                    step_penalty_coef=step_penalty_coef,
                    term_penalty_offset=term_penalty_offset,
                    threshold_coef=threshold_coef,
                )
                # Begin a fresh model rollout after a terminal transition.
                # Continuing from its terminal state would generate invalid data.
                reset_idx = jax.random.randint(rng_reset, (), 0, len(self.dataset.obs))
                next_obs = jnp.where(transition.done, self.dataset.obs[reset_idx], transition.next_obs)
                return (next_obs, rng), transition

            # --- Rollout in dynamics ensemble ---
            rng, rng_sample, rng_rollout = jax.random.split(rng, 3)
            init_obs_idx = jax.random.choice(
                rng_sample, self.dataset.obs.shape[0], (batch_size,)
            )
            init_obs = self.dataset.obs[init_obs_idx]
            rng_rollout = jax.random.split(rng_rollout, init_obs.shape[0])
            _, rollouts = jax.lax.scan(
                _sample_step, (init_obs, rng_rollout), None, length=rollout_length
            )
            rollouts = jax.tree.map(
                lambda x: x.reshape(-1, *x.shape[2:]), rollouts
            )

            # --- Update rollout buffer ---
            n = min(rollouts.obs.shape[0], rollout_buffer.obs.shape[0])

            rollout_buffer = jax.tree.map(
                lambda x, y: jnp.concatenate([x[:-n], y[:n]]),
                rollout_buffer,
                rollouts,
            )
            return rollout_buffer

        return _rollout_fn

    def _sample_transition(
        self,
        rng,
        policy,
        obs,
        step_penalty_coef=0.0,
        term_penalty_offset=None,
        threshold_coef=1.0,
    ):
        """Sample transition from policy and dynamics model."""
        rng_action, rng_dynamics, rng_noise, rng_next_action = jax.random.split(rng, 4)

        # --- Sample action and model predictions ---
        action = policy(obs, rng_action)
        obs_action = jnp.concatenate([obs, action], axis=-1)
        ensemble_mean, ensemble_logvar = self.dynamics_model.apply(
            self.params, obs_action
        )
        ensemble_std = jnp.exp(0.5 * ensemble_logvar)

        # --- Sample transition from a model elite ---
        sample_idx = jax.random.randint(rng_dynamics, (), 0, self.num_ensemble)
        sample_mean, sample_std = ensemble_mean[sample_idx], ensemble_std[sample_idx]
        noise = jax.random.normal(key=rng_noise, shape=sample_mean.shape)
        samples = sample_mean + noise * sample_std
        delta_obs, reward = samples[..., :-1], samples[..., -1]
        next_obs = obs + delta_obs
        next_action = policy(next_obs, rng_next_action)
        done = self.termination_fn(obs, action, next_obs)

        # --- Apply uncertainty penalties ---
        step_penalty = jnp.max(jnp.linalg.norm(ensemble_std, axis=-1))
        reward -= step_penalty_coef * step_penalty
        if term_penalty_offset is not None:
            assert (
                self.discrepancy is not None and self.min_r is not None
            ), "Discrepancy and min_r must be precomputed for termination penalties."
            threshold = self.discrepancy * threshold_coef

            # Calculate uncertainty as maximum distance between elite model predictions
            dists = ensemble_mean[:, None, :] - ensemble_mean[None, :, :]
            dists = jnp.linalg.norm(dists, axis=-1)
            is_uncertain = jnp.max(dists) > threshold
            reward = jnp.where(is_uncertain, self.min_r + term_penalty_offset, reward)
            done |= is_uncertain
        return Transition(obs, action, reward, next_obs, next_action, done, jnp.array(True))


def create_train_state(args, rng, network, dummy_input):
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, dummy_input),
        tx=optax.adamw(args.lr, eps=1e-5, weight_decay=args.weight_decay),
    )


def compute_model_discrepancy(train_state, dataset, elite_idxs, rng):
    """
    Compute maximum distance between predictions of any two elite models.
    Compare elites at the same input on up to 500,000 shuffled transitions.
    """

    @jax.jit
    def get_ensemble_samples(batch_obs_action):
        """Get mean predictions from ensemble dynamics model."""
        # Keep the model's configured ensemble axis (num_ensemble), then
        # select the elite predictions from the model outputs. Slicing the
        # parameters first would leave a 5-member parameter tree passed to a
        # model configured for 7 members.
        mean_obs_rew, _ = train_state.apply_fn(
            train_state.params, batch_obs_action
        )
        return mean_obs_rew[elite_idxs]

    # --- Get elite model predictions across subset of dataset ---
    obs_action = jnp.concatenate((dataset.obs, dataset.action), axis=-1)
    if not len(obs_action):
        raise ValueError("Cannot compute discrepancy on an empty dataset")
    obs_action = jax.random.permutation(rng, obs_action, axis=0)
    obs_action = obs_action[:500_000]
    batch_size = min(500, len(obs_action))
    num_batches = (len(obs_action) + batch_size - 1) // batch_size
    # Repeat existing samples to pad the final batch; this cannot alter a maximum.
    idx = jnp.arange(num_batches * batch_size) % len(obs_action)
    batches = obs_action[idx].reshape(num_batches, batch_size, -1)

    def batch_discrepancy(batch):
        samples = get_ensemble_samples(batch)  # (elite, input, output)
        # Compare models at the SAME input, not predictions at different states.
        differences = samples[:, None, :, :] - samples[None, :, :, :]
        return jnp.linalg.norm(differences, axis=-1).max()

    return float(jax.lax.map(batch_discrepancy, batches).max())


def create_dataset_iter(rng, inputs, targets, batch_size):
    """Create a batched dataset iterator."""
    if not len(inputs) or batch_size < 1:
        raise ValueError("Dynamics partitions and batch_size must be nonempty")
    batch_size = min(batch_size, len(inputs))
    perm = jax.random.permutation(rng, inputs.shape[0])
    shuffled_inputs, shuffled_targets = inputs[perm], targets[perm]
    num_batches = inputs.shape[0] // batch_size
    iter_size = num_batches * batch_size
    dataset_iter = jax.tree.map(
        lambda x: x[:iter_size].reshape(num_batches, batch_size, *x.shape[1:]),
        (shuffled_inputs, shuffled_targets),
    )
    return dataset_iter


def save_dynamics_model(args, dynamics_model, source_dataset=None):
    """Save the EnsembleDynamics object."""
    dynamics_model.training_args = asdict(args) | {
        "dataset_sampling_strategy": DATASET_SAMPLING_STRATEGY,
    }
    dynamics_model.discrepancy_definition = "same_input_elite_l2_v1"
    if source_dataset is not None:
        dynamics_model.provenance_json = collect_run_provenance(args, source_dataset)
    path = dynamics_model_path(
        args.dataset, args.dataset_fraction, args.seed, args.model_path
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(dynamics_model, f)


def dynamics_model_path(dataset, dataset_fraction, seed, model_dir="dynamics_models"):
    dataset_name = dataset.replace("/", "_")
    fraction_name = format(dataset_fraction, "g").replace(".", "p")
    sampling_suffix = (
        f"_sampling_{DATASET_SAMPLING_STRATEGY}"
        if dataset_fraction < 1
        else ""
    )
    filename = (
        f"ensemble_dynamics_model_{dataset_name}_fraction_{fraction_name}"
        f"{sampling_suffix}_seed_{seed}.pkl"
    )
    return os.path.join(model_dir, filename)


def load_dynamics_model(path, *, expected_dataset=None, dataset_fraction=None, seed=None,
                        dataset=None, recompute_discrepancy=False):
    """Load the EnsembleDynamics object."""
    print(f"Loading dynamics model from {path}")
    with open(path, "rb") as f:
        dynamics_model = pickle.load(f)
    training_args = getattr(dynamics_model, "training_args", None)
    if training_args is None:
        if dataset_fraction is not None and dataset_fraction < 1:
            raise ValueError(
                "Cannot use a legacy dynamics checkpoint with a partial dataset "
                "fraction because its sampling strategy is unknown; retrain it."
            )
        warnings.warn("Legacy dynamics checkpoint has no dataset/fraction/seed provenance", UserWarning)
    else:
        expected = {"dataset": expected_dataset, "dataset_fraction": dataset_fraction, "seed": seed}
        for key, value in expected.items():
            if value is not None and training_args.get(key) != value:
                raise ValueError(f"Dynamics checkpoint {key} mismatch: {training_args.get(key)!r} != {value!r}")
        if dataset_fraction is not None and dataset_fraction < 1:
            saved_strategy = training_args.get("dataset_sampling_strategy")
            if saved_strategy != DATASET_SAMPLING_STRATEGY:
                raise ValueError(
                    "Dynamics checkpoint sampling strategy mismatch: "
                    f"{saved_strategy!r} != {DATASET_SAMPLING_STRATEGY!r}; retrain it."
                )
    if expected_dataset is not None:
        dynamics_model.termination_fn = get_termination_fn(expected_dataset)
    if recompute_discrepancy and (
        getattr(dynamics_model, "discrepancy_definition", None) != "same_input_elite_l2_v1"
        or getattr(dynamics_model, "discrepancy", None) is None
    ):
        if dataset is None:
            raise ValueError("Provide matching transitions to recompute legacy MOReL discrepancy")
        warnings.warn("Recomputing legacy MOReL discrepancy at matching state-action inputs; checkpoint is unchanged", UserWarning)
        state = SimpleNamespace(apply_fn=dynamics_model.dynamics_model.apply, params=dynamics_model.params)
        dynamics_model.discrepancy = compute_model_discrepancy(
            state, dataset, jnp.arange(dynamics_model.num_ensemble), jax.random.PRNGKey(seed or 0)
        )
        dynamics_model.min_r = float(jnp.min(dataset.reward))
        dynamics_model.discrepancy_definition = "same_input_elite_l2_v1"
    return dynamics_model


def log_info(info):
    """Log metrics to wandb."""
    info = {"dynamics/" + k: v for k, v in info.items()}
    jax.experimental.io_callback(wandb.log, None, info)


r"""
         _(_)_
        (_)@(_)
         /(_)
    |\  /
    \ ||
     \||
      \|
    ___|_____
    \       /
     \     /
      \___/     Dynamics Model
"""


def train_dynamics_model(train_state, args, dataset, rng):
    # --- Prepare dataset for training ---
    inputs = jnp.concatenate((dataset.obs, dataset.action), axis=-1)
    delta_obs = dataset.next_obs - dataset.obs
    targets = jnp.concatenate((delta_obs, dataset.reward.reshape(-1, 1)), axis=-1)
    if len(inputs) < 2 or not 0 < args.validation_split < 1:
        raise ValueError("Dynamics training requires at least two transitions and 0 < validation_split < 1")

    rng, split_rng = jax.random.split(rng)
    train_size = min(len(inputs) - 1, max(1, int((1 - args.validation_split) * inputs.shape[0])))
    shuffled_idxs = jax.random.permutation(split_rng, inputs.shape[0])
    train_idxs, val_idxs = shuffled_idxs[:train_size], shuffled_idxs[train_size:]
    train_inputs, train_targets = inputs[train_idxs], targets[train_idxs]
    val_inputs, val_targets = inputs[val_idxs], targets[val_idxs]
    print(f"Dataset size: {inputs.shape[0]}, train partition: {train_size}")

    # --- Make train functions ---
    def _train_step(train_state, batch):
        """Train dynamics model for one step."""
        inputs, targets = batch

        def _loss_fn(params):
            mean, logvar = train_state.apply_fn(params, inputs)
            mse_loss = ((mean - targets) ** 2) * jnp.exp(-logvar)
            mse_loss = mse_loss.sum(0).mean()
            var_loss = logvar.sum(0).mean()
            max_logvar = params["params"]["max_logvar"]
            min_logvar = params["params"]["min_logvar"]
            logvar_diff = (max_logvar - min_logvar).sum()
            loss = mse_loss + var_loss + args.logvar_diff_coef * logvar_diff
            info = {
                "step": train_state.step,
                "loss": loss,
                "mse_loss": mse_loss,
                "var_loss": var_loss,
                "max_logvar": max_logvar.sum(),
                "min_logvar": min_logvar.sum(),
                "logvar_diff": logvar_diff,
            }
            return loss, info

        grads, info = jax.grad(_loss_fn, has_aux=True)(train_state.params)
        train_state = train_state.apply_gradients(grads=grads)
        if args.log:
            do_log = train_state.step % args.eval_interval == 1
            jax.lax.cond(do_log, log_info, lambda x: None, info)
        return train_state, None

    def _eval_step(train_state, batch):
        """Evaluate the dynamics model on the holdout set."""
        inputs, targets = batch

        def _loss_fn(params):
            """Calculate evaluation loss as unweighted prediction MSE."""
            mean_predictions, _ = train_state.apply_fn(params, inputs)
            loss = jnp.mean(((mean_predictions - targets) ** 2), axis=(1, 2))
            elite_idxs = jnp.argsort(loss)[: args.num_elites]
            info = {
                "step": train_state.step,
                "validation_loss": loss.mean(),
                "elite_idxs": elite_idxs,
            }
            return loss, info

        loss, info = _loss_fn(train_state.params)
        return train_state, (loss, info)

    def train_epoch(_, carry):
        """Train dynamics model for one epoch."""
        rng, train_state, elite_idxs = carry
        rng, rng_train, rng_val = jax.random.split(rng, 3)
        size = args.batch_size
        train_iter = create_dataset_iter(rng_train, train_inputs, train_targets, size)
        train_state = jax.lax.scan(_train_step, train_state, train_iter)[0]
        val_iter = create_dataset_iter(rng_val, val_inputs, val_targets, size)
        val_loss, val_info = jax.lax.scan(_eval_step, train_state, val_iter)[1]
        elite_idxs = val_loss.mean(axis=0).argsort()[: args.num_elites]
        if args.log:
            val_info = jax.tree.map(lambda x: jnp.mean(x, axis=0), val_info)
            log_info({**val_info, "elite_idxs": elite_idxs})
        return rng, train_state, elite_idxs

    # --- Train model for N epochs ---
    dummy_elite_idxs = jnp.zeros((args.num_elites,), jnp.int32)
    init_carry = (rng, train_state, dummy_elite_idxs)
    _, train_state, elite_idxs = jax.lax.fori_loop(
        0, args.num_epochs, train_epoch, init_carry
    )
    return train_state, elite_idxs


if __name__ == "__main__":
    # --- Parse arguments ---
    args = tyro.cli(Args)
    rng = jax.random.PRNGKey(args.seed)

    # --- Initialize logger ---
    if args.log:
        wandb.init(
            config=args,
            project=args.wandb_project,
            entity=args.wandb_team,
            group=args.wandb_group,
            job_type="train_dynamics",
        )

    # --- Initialize environment and dataset ---
    minari_dataset = load_minari_dataset(args.dataset)
    transition_dict = transitions_from_minari(
        minari_dataset, args.dataset_fraction, args.seed, include_next_action=True
    )
    dataset = Transition(
        obs=transition_dict["obs"],
        action=transition_dict["action"],
        reward=transition_dict["reward"],
        next_obs=transition_dict["next_obs"],
        next_action=transition_dict["next_action"],
        next_action_valid=transition_dict["next_action_valid"],
        done=transition_dict["done"],
    )
    env = gym.make(minari_dataset.env_spec)
    num_actions = env.action_space.shape[0]
    dummy_delta_obs_action = jnp.zeros(env.observation_space.shape[0] + num_actions)
    dynamics_net = EnsembleDynamicsModel(
        obs_dim=env.observation_space.shape[0],
        action_dim=num_actions,
        num_ensemble=args.num_ensemble,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
    )
    rng, dynamics_rng = jax.random.split(rng)
    train_state = create_train_state(
        args, dynamics_rng, dynamics_net, [dummy_delta_obs_action]
    )

    # --- Train dynamics model ---
    print("Training dynamics model...")
    rng, trn_rng = jax.random.split(rng)
    train_state, elite_idxs = train_dynamics_model(train_state, args, dataset, trn_rng)

    # --- Precompute statistics for termination penalties (required for MOReL) ---
    discrepancy, min_r = None, None
    if args.precompute_term_stats:
        print("Precomputing dataset statistics for termination penalties...")
        rng, rng_discrepancy = jax.random.split(rng)
        discrepancy = compute_model_discrepancy(
            train_state, dataset, elite_idxs, rng_discrepancy
        )
        min_r = float(jnp.min(dataset.reward))
        print(f"Model discrepancy: {discrepancy}, Minimum dataset reward: {min_r}")

    # --- Save dynamics model ---
    termination_fn = get_termination_fn(args.dataset)
    dynamics_model = EnsembleDynamics(
        dynamics_net, train_state.params, elite_idxs, termination_fn, discrepancy, min_r
    )
    print("Saving dynamics model...")
    save_dynamics_model(args, dynamics_model, source_dataset=minari_dataset)

    env.close()
    if args.log:
        wandb.finish()
