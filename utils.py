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


def transitions_from_minari(minari_dataset):
	"""Flatten a MinariDataset's episodes into transition-level arrays.

	Args:
		minari_dataset: A MinariDataset object.

	Returns:
		A dict with keys: obs, action, reward, next_obs, done (all as JAX arrays).
	"""
	obs, next_obs, actions, rewards, dones = [], [], [], [], []
	for episode in minari_dataset.iterate_episodes():
		obs.append(episode.observations[:-1])
		next_obs.append(episode.observations[1:])
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
