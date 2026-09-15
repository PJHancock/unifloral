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
