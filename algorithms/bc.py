from collections import namedtuple
from dataclasses import dataclass, asdict
from datetime import datetime
import os
import warnings

import distrax
import flax.linen as nn
from flax.training.train_state import TrainState
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as onp
import optax
import tyro
import wandb
import minari

from utils import load_minari_dataset, transitions_from_minari, get_normalized_score, create_dummy_obs, flatten_observation

os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True"


@dataclass
class Args:
    # --- Experiment ---
    seed: int = 0
    dataset: str = "mujoco/halfcheetah/medium-v0"
    algorithm: str = "bc"
    num_updates: int = 1_000_000
    eval_interval: int = 2500
    eval_workers: int = 8
    eval_final_episodes: int = 1000
    # --- Logging ---
    log: bool = False
    wandb_project: str = "unifloral"
    wandb_team: str = "flair"
    wandb_group: str = "debug"
    # --- Generic optimization ---
    lr: float = 3e-4
    batch_size: int = 256


r"""
     |\  __
     \| /_/
      \|
    ___|_____
    \       /
     \     /
      \___/     Preliminaries
"""

AgentTrainState = namedtuple("AgentTrainState", "actor")
Transition = namedtuple("Transition", "obs action reward next_obs done")


class DeterministicTanhActor(nn.Module):
    num_actions: int
    obs_mean: jax.Array
    obs_std: jax.Array

    @nn.compact
    def __call__(self, x):
        x = (x - self.obs_mean) / (self.obs_std + 1e-3)
        for _ in range(2):
            x = nn.Dense(256)(x)
            x = nn.relu(x)
        action = nn.Dense(self.num_actions)(x)
        pi = distrax.Transformed(
            distrax.Deterministic(action),
            distrax.Tanh(),
        )
        return pi


def create_train_state(args, rng, network, dummy_input):
    return TrainState.create(
        apply_fn=network.apply,
        params=network.init(rng, *dummy_input),
        tx=optax.adam(args.lr, eps=1e-5),
    )


def eval_agent(args, rng, env, agent_state):
    # --- Reset environment ---
    step = 0
    returned = onp.zeros(args.eval_workers).astype(bool)
    cum_reward = onp.zeros(args.eval_workers)
    rng, rng_reset = jax.random.split(rng)
    rng_reset = jax.random.split(rng_reset, args.eval_workers)
    obs, info = env.reset()
    obs = flatten_observation(obs)

    # --- Rollout agent ---
    @jax.jit
    @jax.vmap
    def _policy_step(rng, obs):
        pi = agent_state.actor.apply_fn(agent_state.actor.params, obs)
        action = pi.sample(seed=rng)
        return jnp.nan_to_num(action)

    max_episode_steps = env.unwrapped.spec.max_episode_steps
    while step < max_episode_steps and not returned.all():
        # --- Take step in environment ---
        step += 1
        rng, rng_step = jax.random.split(rng)
        rng_step = jax.random.split(rng_step, args.eval_workers)
        action = _policy_step(rng_step, jnp.array(obs))
        obs, reward, terminated, truncated, info = env.step(onp.array(action))
        obs = flatten_observation(obs)
        done = terminated | truncated

        # --- Track cumulative reward ---
        cum_reward += reward * ~returned
        returned |= done

    if step >= max_episode_steps and not returned.all():
        warnings.warn("Maximum steps reached before all episodes terminated")
    return cum_reward


r"""
          __/)
       .-(__(=:
    |\ |    \)
    \ ||
     \||
      \|
    ___|_____
    \       /
     \     /
      \___/     Agent
"""


def make_train_step(args, actor_apply_fn, dataset):
    """Make JIT-compatible agent train step."""

    def _train_step(runner_state, _):
        rng, agent_state = runner_state

        # --- Sample batch ---
        rng, rng_batch = jax.random.split(rng)
        batch_indices = jax.random.randint(
            rng_batch, (args.batch_size,), 0, len(dataset.obs)
        )
        batch = jax.tree_util.tree_map(lambda x: x[batch_indices], dataset)

        # --- Update actor ---
        def _actor_loss_function(params):
            def _transition_loss(transition):
                pi = actor_apply_fn(params, transition.obs)
                pi_action = pi.sample(seed=None)
                bc_loss = jnp.square(pi_action - transition.action).mean()
                return bc_loss

            bc_loss = jax.vmap(_transition_loss)(batch)
            return bc_loss.mean()

        loss_fn = jax.value_and_grad(_actor_loss_function)
        actor_loss, actor_grad = loss_fn(agent_state.actor.params)
        agent_state = agent_state._replace(
            actor=agent_state.actor.apply_gradients(grads=actor_grad)
        )

        loss = {
            "actor_loss": actor_loss,
            "bc_loss": actor_loss,
        }
        return (rng, agent_state), loss

    return _train_step


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
            job_type="train_agent",
        )

    # --- Initialize environment and dataset ---
    minari_dataset = load_minari_dataset(args.dataset)
    env = gym.make_vec(minari_dataset.env_spec, num_envs=args.eval_workers)
    dataset = Transition(**transitions_from_minari(minari_dataset))

    # --- Initialize agent and value networks ---
    base_env = gym.make(minari_dataset.env_spec)
    num_actions = base_env.action_space.shape[0]
    base_env.close()
    obs_mean = dataset.obs.mean(axis=0)
    obs_std = jnp.nan_to_num(dataset.obs.std(axis=0), nan=1.0)
    dummy_obs = create_dummy_obs(base_env.observation_space)
    dummy_obs = flatten_observation(dummy_obs)  # Ensure flat for network init
    dummy_action = jnp.zeros(num_actions)
    actor_net = DeterministicTanhActor(num_actions, obs_mean, obs_std)

    rng, rng_actor = jax.random.split(rng)
    agent_state = AgentTrainState(
        actor=create_train_state(args, rng_actor, actor_net, [dummy_obs])
    )

    # --- Make train step ---
    _agent_train_step_fn = make_train_step(args, actor_net.apply, dataset)

    num_evals = args.num_updates // args.eval_interval
    for eval_idx in range(num_evals):
        # --- Execute train loop ---
        (rng, agent_state), loss = jax.lax.scan(
            _agent_train_step_fn,
            (rng, agent_state),
            None,
            args.eval_interval,
        )

        # --- Evaluate agent ---
        rng, rng_eval = jax.random.split(rng)
        returns = eval_agent(args, rng_eval, env, agent_state)
        scores = get_normalized_score(minari_dataset, returns)

        # --- Log metrics ---
        step = (eval_idx + 1) * args.eval_interval
        print("Step:", step, f"\t Score: {scores.mean():.2f}")
        if args.log:
            log_dict = {
                "return": returns.mean(),
                "score": onp.mean(scores),
                "score_std": onp.std(scores),
                "num_updates": step,
                **{k: loss[k][-1] for k in loss},
            }
            wandb.log(log_dict)

    # --- Evaluate final agent ---
    if args.eval_final_episodes > 0:
        final_iters = int(onp.ceil(args.eval_final_episodes / args.eval_workers))
        print(f"Evaluating final agent for {final_iters} iterations...")
        _rng = jax.random.split(rng, final_iters)
        rets = onp.array([eval_agent(args, _rng, env, agent_state) for _rng in _rng])
        scores = get_normalized_score(minari_dataset, rets)
        agg_fn = lambda x, k: {k: x, f"{k}_mean": x.mean(), f"{k}_std": x.std()}
        info = agg_fn(rets, "final_returns") | agg_fn(scores, "final_scores")

        # --- Write final returns to file ---
        os.makedirs("final_returns", exist_ok=True)
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        safe_dataset = args.dataset.replace("/", "-")
        filename = f"{args.algorithm}_{safe_dataset}_{time_str}.npz"
        with open(os.path.join("final_returns", filename), "wb") as f:
            onp.savez_compressed(f, **info, args=asdict(args))

        if args.log:
            wandb.save(os.path.join("final_returns", filename))

    env.close()
    if args.log:
        wandb.finish()
