"""Utility functions for offline RL experiments."""

import warnings

import gymnasium as gym
import minari
import numpy as np
import jax.numpy as jnp


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


def transitions_from_minari(minari_dataset, dataset_fraction=1.0, seed=0):
	"""Flatten a MinariDataset's episodes into transition-level arrays.

	Args:
		minari_dataset: A MinariDataset object.
		dataset_fraction: Fraction of transitions to sample uniformly without replacement.
		seed: Random seed used to select transitions.

	Returns:
		A dict with keys: obs, action, reward, next_obs, done (all as JAX arrays).
	"""
	if not 0 < dataset_fraction <= 1:
		raise ValueError("dataset_fraction must be in the interval (0, 1]")

	obs, next_obs, actions, rewards, dones = [], [], [], [], []
	for episode in minari_dataset.iterate_episodes():
		# Flatten full observations first, then slice
		flat_observations = _flatten_obs(episode.observations)
		ep_obs = flat_observations[:-1]
		ep_next_obs = flat_observations[1:]

		obs.append(ep_obs)
		next_obs.append(ep_next_obs)
		actions.append(episode.actions)
		rewards.append(episode.rewards)
		dones.append(episode.terminations)

	transitions = {
		"obs": jnp.array(np.concatenate(obs)),
		"action": jnp.array(np.concatenate(actions)),
		"reward": jnp.array(np.concatenate(rewards)),
		"next_obs": jnp.array(np.concatenate(next_obs)),
		"done": jnp.array(np.concatenate(dones)),
	}
	if dataset_fraction < 1:
		num_transitions = len(transitions["obs"])
		sample_size = max(1, int(num_transitions * dataset_fraction))
		indices = np.random.default_rng(seed).permutation(num_transitions)[:sample_size]
		transitions = {key: values[indices] for key, values in transitions.items()}
	return transitions


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
	"""Normalize returns to a 0-100 scale using reference min/max scores.

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
		Normalized score(s) in range [0, 100] if reference bounds are available
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
