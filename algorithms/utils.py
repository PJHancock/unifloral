"""Utility functions for offline RL experiments."""

import os
import re
import warnings
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

import gymnasium as gym
import minari
import numpy as np
import jax.numpy as jnp


DATASET_SAMPLING_STRATEGY = "trajectory_return_quantile_v1"
DATASET_SAMPLING_QUANTILE_BINS = 5


def final_returns_directory(wandb_group=None):
	"""Return the output directory for final evaluation archives.

	Sweep runs are grouped under ``final_returns/<wandb_group>`` so that a
	notebook can analyze one sweep without mixing it with other runs. Group
	names are sanitized before being used as directory names.
	"""
	group = wandb_group if wandb_group is not None else os.environ.get("WANDB_GROUP", "")
	group = str(group).strip()
	base_dir = os.environ.get("FINAL_RETURNS_ROOT", "final_returns")
	if not group:
		return base_dir

	safe_group = re.sub(r"[^A-Za-z0-9_.-]+", "_", group).strip("._")
	if not safe_group:
		raise ValueError(f"Invalid W&B group name for a directory: {group!r}")
	return os.path.join(base_dir, safe_group)


def load_minari_dataset(dataset_id: str):
	"""Load a Minari dataset by its native ID.

	Args:
		dataset_id: A Minari dataset ID, e.g. 'mujoco/halfcheetah/medium-v0' or 'D4RL/pen/human-v2'

	Returns:
		A MinariDataset object.
	"""
	return minari.load_dataset(dataset_id, download=True)


def _flatten_obs(obs):
	"""Flatten observation (handles both flat arrays and dict of arrays)."""
	if isinstance(obs, dict):
		# For goal-conditioned envs (e.g., kitchen), flatten dict recursively
		flat_parts = []
		for key in sorted(obs.keys()):
			val = obs[key]
			if isinstance(val, dict):
				flat_parts.append(_flatten_obs(val))
			elif isinstance(val, np.ndarray):
				flat_parts.append(val.reshape(val.shape[0], -1) if val.ndim > 1 else val)
		return np.concatenate(flat_parts, axis=-1)
	elif isinstance(obs, np.ndarray):
		# Flat array observation
		return obs.reshape(obs.shape[0], -1) if obs.ndim > 1 else obs
	else:
		raise ValueError(f"Unsupported observation type: {type(obs)}")


def _sample_trajectory_indices(scores, lengths, dataset_fraction, seed):
	"""Choose whole trajectories across return quantiles near a transition budget."""
	num_trajectories = len(scores)
	if dataset_fraction == 1:
		return np.arange(num_trajectories)

	num_bins = min(DATASET_SAMPLING_QUANTILE_BINS, num_trajectories)
	ordered = np.argsort(scores, kind="stable")
	bins = [np.asarray(items) for items in np.array_split(ordered, num_bins)]
	rng = np.random.default_rng(seed)
	for items in bins:
		rng.shuffle(items)
	# Rotate the bin order by a seeded permutation so small budgets do not always
	# favor the lowest-return quantiles.
	bin_order = rng.permutation(num_bins)
	interleaved = []
	for depth in range(max(map(len, bins))):
		for bin_idx in bin_order:
			if depth < len(bins[bin_idx]):
				interleaved.append(int(bins[bin_idx][depth]))

	target_transitions = sum(lengths) * dataset_fraction
	best_count = 1
	selected_transitions = int(lengths[interleaved[0]])
	best_distance = abs(selected_transitions - target_transitions)
	for count, trajectory_idx in enumerate(interleaved[1:], start=2):
		selected_transitions += int(lengths[trajectory_idx])
		distance = abs(selected_transitions - target_transitions)
		# Keep the shorter prefix on ties, avoiding unnecessary budget overshoot.
		if distance < best_distance:
			best_count = count
			best_distance = distance
		elif selected_transitions > target_transitions:
			break
	return np.asarray(interleaved[:best_count], dtype=int)


def transitions_from_minari(minari_dataset, dataset_fraction=1.0, seed=0, *, include_next_action=False):
	"""Flatten selected complete Minari episodes into transition-level arrays.

	Args:
		minari_dataset: A MinariDataset object.
		dataset_fraction: Fraction of all transitions used as a target budget. For
			fractions below 1, complete episodes are selected across return quantiles
			to get as close as possible to that budget.
		seed: Random seed used to select episodes within return quantiles.
		include_next_action: Include episode-aligned next actions and a validity
			mask. The last action of an episode has no recorded successor.

	Returns:
		A dict with keys: obs, action, reward, next_obs, done (all as JAX arrays).
	"""
	if not 0 < dataset_fraction <= 1:
		raise ValueError("dataset_fraction must be in the interval (0, 1]")

	episodes = []
	for episode in minari_dataset.iterate_episodes():
		# Flatten full observations first, then slice
		flat_observations = _flatten_obs(episode.observations)
		ep_obs = flat_observations[:-1]
		ep_next_obs = flat_observations[1:]
		n = len(episode.actions)
		if n == 0:
			continue
		if not all(len(x) == n for x in (ep_obs, ep_next_obs, episode.rewards, episode.terminations)):
			raise ValueError(f"Inconsistent transition lengths in episode {episode.id}")

		episode_data = {
			"obs": ep_obs,
			"action": episode.actions,
			"reward": episode.rewards,
			"next_obs": ep_next_obs,
			"done": episode.terminations,
			"score": float(np.sum(episode.rewards)),
			"length": n,
		}
		if not np.isfinite(episode_data["score"]):
			raise ValueError(f"Non-finite reward sum in episode {episode.id}")
		if include_next_action:
			# Construct before subsampling; never cross an episode boundary.
			episode_data["next_action"] = np.concatenate((episode.actions[1:], np.zeros_like(episode.actions[:1])))
			episode_data["next_action_valid"] = np.arange(n) < n - 1
		episodes.append(episode_data)
	if not episodes:
		raise ValueError("Dataset contains no transitions")

	indices = _sample_trajectory_indices(
		[sample["score"] for sample in episodes],
		[sample["length"] for sample in episodes],
		dataset_fraction,
		seed,
	)
	keys = ("obs", "action", "reward", "next_obs", "done")
	if include_next_action:
		keys += ("next_action", "next_action_valid")
	return {
		key: jnp.array(np.concatenate([episodes[idx][key] for idx in indices]))
		for key in keys
	}


def evaluation_reset(env, rng_keys):
    """Seed every worker from the evaluation RNG, independently of training."""
    seeds = [int(np.asarray(key, dtype=np.uint32)[0]) for key in np.asarray(rng_keys)]
    return env.reset(seed=seeds)


def validate_evaluation_actions(actions):
    """Fail visibly if a policy is nonfinite instead of hiding it with zeros."""
    if not np.isfinite(np.asarray(actions)).all():
        raise FloatingPointError("Policy produced nonfinite evaluation actions")


def collect_run_provenance(args, dataset):
    """Optional JSON metadata for new archives; old archives remain readable."""
    code_dir = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(code_dir.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    metadata = dataset.storage.read_raw_metadata(dataset.storage.data_path)
    metadata_hash = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(code_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    versions = {}
    for package in ("jax", "flax", "optax", "gymnasium", "gymnasium-robotics", "minari", "mujoco", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return json.dumps({
        "schema_version": 1, "algorithm_commit": commit,
        "algorithm_sources_sha256": digest.hexdigest(),
        "dataset_metadata_sha256": metadata_hash, "dataset_id": args.dataset,
        "dataset_fraction": args.dataset_fraction, "subset_seed": args.seed,
        "dataset_sampling_strategy": DATASET_SAMPLING_STRATEGY,
        "dataset_sampling_quantile_bins": DATASET_SAMPLING_QUANTILE_BINS,
        "environment_spec": metadata.get("env_spec"),
        "evaluation_environment_spec": metadata.get("eval_env_spec", metadata.get("env_spec")),
        "versions": versions, "evaluation_reset_seeded": True,
        "next_action_episode_aligned": True,
    }, sort_keys=True)


def flatten_observation(obs):
	"""Flatten an observation (handles both flat arrays and dict of arrays, preserves batch dims)."""
	if isinstance(obs, dict):
		# For goal-conditioned envs (e.g., kitchen), flatten dict recursively
		# Check if this is a batch of dicts (list of dicts) or single dict
		flat_parts = []
		for key in sorted(obs.keys()):
			val = obs[key]
			if isinstance(val, dict):
				flattened = flatten_observation(val)
				if isinstance(flattened, (np.ndarray, jnp.ndarray)) and flattened.size > 0:
					val_arr = np.asarray(flattened)
					# Reshape to (-1,) if single obs, or keep batch dim if batched
					if val_arr.ndim == 1:
						flat_parts.append(val_arr.reshape(-1))
					else:
						flat_parts.append(val_arr.reshape(val_arr.shape[0], -1))
			elif isinstance(val, (np.ndarray, jnp.ndarray)) and val.size > 0:
				val_arr = np.asarray(val)
				# Preserve batch dimension if it exists
				if val_arr.ndim == 1:
					flat_parts.append(val_arr.reshape(-1))
				else:
					flat_parts.append(val_arr.reshape(val_arr.shape[0], -1))

		if not flat_parts:
			return np.array([])

		# Concatenate along last axis (preserves batch dim if present)
		return np.concatenate(flat_parts, axis=-1)
	elif isinstance(obs, (np.ndarray, jnp.ndarray)):
		# Flat array observation (numpy or JAX)
		obs = np.asarray(obs)
		# If 3D+ (batch + spatial), reshape to (batch, -1)
		if obs.ndim > 2:
			return obs.reshape(obs.shape[0], -1)
		# If 2D, could be (batch, features) - keep as is
		elif obs.ndim == 2:
			return obs
		# If 1D, single obs
		else:
			return obs
	else:
		return obs


def create_dummy_obs(observation_space):
	"""Create a dummy observation matching the observation space shape/structure."""
	# Check if it's a Dict space
	if hasattr(observation_space, 'spaces'):
		# Dict observation space - recursively create for each sub-space
		dummy = {}
		for key, space in observation_space.spaces.items():
			if hasattr(space, 'spaces'):
				# Nested dict
				dummy[key] = create_dummy_obs(space)
			elif hasattr(space, 'shape') and space.shape and None not in space.shape:
				# Box space with valid shape
				dummy[key] = jnp.zeros(space.shape)
			else:
				# Can't create from shape, use space.sample() instead
				dummy[key] = jnp.array(space.sample())
		return dummy
	elif hasattr(observation_space, 'shape'):
		# Flat Box observation space
		if observation_space.shape is None or (isinstance(observation_space.shape, tuple) and None in observation_space.shape):
			raise ValueError(f"Cannot create dummy obs for shape: {observation_space.shape}")
		return jnp.zeros(observation_space.shape)
	else:
		raise ValueError(f"Unsupported observation space type: {type(observation_space)}")


# D4RL reference scores for environments in Minari's mujoco/ namespace, which (unlike
# D4RL/* Minari datasets) do not carry ref_min_score/ref_max_score metadata. Source:
# d4rl/infos.py (https://github.com/Farama-Foundation/D4RL) 
# One (min, max) pair per environment, shared across
# all dataset splits (medium, medium-expert, expert, random, medium-replay, ...), per
# D4RL's own convention of one ref_min/ref_max per environment regardless of split.
_D4RL_REFERENCE_SCORES = {
	"halfcheetah": (-280.178953, 12135.0),
	"hopper": (-20.272305, 3234.3),
	"walker2d": (1.629008, 4592.3),
}


def _lookup_fallback_reference_score(minari_dataset):
	"""Find (ref_min, ref_max) for a dataset by matching its ID against known envs.

	Returns None if no match is found.
	"""
	dataset_id = getattr(minari_dataset, "id", None) or getattr(
		minari_dataset.spec, "dataset_id", ""
	)
	dataset_id = dataset_id.lower()
	for env_name, bounds in _D4RL_REFERENCE_SCORES.items():
		if env_name in dataset_id:
			return bounds
	return None


def get_normalized_score(minari_dataset, returns):
	"""Normalize returns using reference scores as 0 and 100 anchors.

	Tries Minari's own stored ref_min_score/ref_max_score metadata first (valid for
	D4RL/pen, D4RL/pointmaze, D4RL/antmaze, etc.). Most non-D4RL-derived Minari
	datasets (notably the mujoco/ namespace: mujoco/halfcheetah, mujoco/hopper,
	mujoco/walker2d) do not carry this metadata, so this function falls back to a
	small locally-maintained table of D4RL reference scores for those environments.

	If a dataset has neither Minari metadata nor a local fallback entry, this
	function emits a UserWarning and returns the raw, unnormalized returns rather
	than raising -- raising would crash long-running training loops over what is
	likely just a newly-added dataset that hasn't been added to the fallback table
	yet. Callers should watch for this warning; raw returns are not comparable
	across datasets/methods.

	Known limitation (out of scope for this fix): D4RL/kitchen/mixed-v2's Minari
	reward is a persistent/monotonic completion indicator rather than D4RL's
	original one-time sparse reward, so its summed episode returns (roughly 154-428)
	do not correspond to its stored ref_max_score of 4.0, and normalized scores for
	Kitchen should not be trusted even though this function will not warn about it
	(it does have valid metadata, so the Minari path succeeds "silently"). Fixing
	this would require changing return computation in each algorithm's eval loop
	and is deliberately not addressed here.

	Args:
		minari_dataset: A MinariDataset object.
		returns: Episode returns to normalize (scalar or array).

	Returns:
		Normalized score(s), possibly below 0 or above 100, if bounds are available
		(via Minari metadata or the local fallback table), otherwise raw returns.
	"""
	try:
		return minari.get_normalized_score(minari_dataset, np.asarray(returns)) * 100.0
	except (ValueError, AttributeError):
		pass

	fallback = _lookup_fallback_reference_score(minari_dataset)
	if fallback is not None:
		ref_min, ref_max = fallback
		return (np.asarray(returns) - ref_min) / (ref_max - ref_min) * 100.0

	dataset_id = getattr(minari_dataset, "id", "<unknown dataset>")
	warnings.warn(
		f"No normalization reference scores available for dataset '{dataset_id}' "
		"(neither Minari metadata nor local fallback table). Returning raw, "
		"unnormalized returns -- these are NOT comparable across datasets/methods. "
		"Add an entry to _D4RL_REFERENCE_SCORES in utils.py if a D4RL reference "
		"score exists for this environment.",
		UserWarning,
		stacklevel=2,
	)
	return returns
