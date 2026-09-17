"""Utility functions for offline RL experiments."""

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


def transitions_from_minari(minari_dataset):
	"""Flatten a MinariDataset's episodes into transition-level arrays.

	Args:
		minari_dataset: A MinariDataset object.

	Returns:
		A dict with keys: obs, action, reward, next_obs, done (all as JAX arrays).
	"""
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

	return {
		"obs": jnp.array(np.concatenate(obs)),
		"action": jnp.array(np.concatenate(actions)),
		"reward": jnp.array(np.concatenate(rewards)),
		"next_obs": jnp.array(np.concatenate(next_obs)),
		"done": jnp.array(np.concatenate(dones)),
	}


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


def get_normalized_score(minari_dataset, returns):
	"""Normalize returns to a 0-100 scale using Minari's stored reference scores.

	Falls back to raw returns if the dataset has no ref_min_score/ref_max_score
	(true for most non-D4RL-derived Minari datasets, e.g. the mujoco/ namespace).

	Args:
		minari_dataset: A MinariDataset object.
		returns: Episode returns to normalize (scalar or array).

	Returns:
		Normalized score(s) in range [0, 100], or raw returns if normalization unavailable.
	"""
	try:
		return minari.get_normalized_score(minari_dataset, np.asarray(returns)) * 100.0
	except ValueError:
		return returns
