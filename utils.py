"""Utility functions for offline RL experiments."""

import gymnasium as gym
import minari


def load_d4rl_dataset(dataset_name: str):
    """
    Load a D4RL dataset using Minari.

    Maps old D4RL v2 names to Minari dataset IDs and loads the data.
    Returns dict with keys: observations, actions, rewards, next_observations, terminals
    """
    # Map old D4RL names to Minari dataset IDs
    d4rl_to_minari = {
        "halfcheetah-medium-v2": "D4RL/halfcheetah/medium-v0",
        "halfcheetah-medium-expert-v2": "D4RL/halfcheetah/medium-expert-v0",
        "halfcheetah-expert-v2": "D4RL/halfcheetah/expert-v0",
        "halfcheetah-random-v2": "D4RL/halfcheetah/random-v0",
        "hopper-medium-v2": "D4RL/hopper/medium-v0",
        "hopper-medium-expert-v2": "D4RL/hopper/medium-expert-v0",
        "hopper-expert-v2": "D4RL/hopper/expert-v0",
        "hopper-random-v2": "D4RL/hopper/random-v0",
        "walker2d-medium-v2": "D4RL/walker2d/medium-v0",
        "walker2d-medium-expert-v2": "D4RL/walker2d/medium-expert-v0",
        "walker2d-expert-v2": "D4RL/walker2d/expert-v0",
        "walker2d-random-v2": "D4RL/walker2d/random-v0",
        "pen-human-v1": "D4RL/pen/human-v2",
        "pen-cloned-v1": "D4RL/pen/cloned-v2",
        "pen-expert-v1": "D4RL/pen/expert-v2",
        "kitchen-mixed-v0": "D4RL/kitchen/mixed-v0",
        "maze2d-large-v1": "D4RL/maze2d/large-v0",
        "antmaze-large-diverse-v2": "D4RL/antmaze/large-diverse-v0",
    }

    minari_id = d4rl_to_minari.get(dataset_name, dataset_name)

    # Load dataset from Minari
    dataset = minari.load_dataset(minari_id)

    # Extract episodes and flatten to transition format
    observations = []
    actions = []
    rewards = []
    next_observations = []
    terminals = []

    for episode in dataset.iterate_episodes():
        for i in range(len(episode.observations) - 1):
            observations.append(episode.observations[i])
            actions.append(episode.actions[i])
            rewards.append(episode.rewards[i])
            next_observations.append(episode.observations[i + 1])
            terminals.append(episode.truncations[i] or episode.terminations[i])

    import numpy as np
    return {
        "observations": np.array(observations),
        "actions": np.array(actions),
        "rewards": np.array(rewards),
        "next_observations": np.array(next_observations),
        "terminals": np.array(terminals),
    }


def get_normalized_score(dataset_name: str, returns):
    """
    Get normalized score for D4RL datasets (0-100 scale).

    Normalized by (return - random_return) / (expert_return - random_return)
    """
    # D4RL normalized score bounds for each task
    d4rl_scores = {
        "halfcheetah-medium-v2": {"random": -280.178, "expert": 12135.0},
        "halfcheetah-medium-expert-v2": {"random": -280.178, "expert": 12135.0},
        "halfcheetah-expert-v2": {"random": -280.178, "expert": 12135.0},
        "halfcheetah-random-v2": {"random": -280.178, "expert": 12135.0},
        "hopper-medium-v2": {"random": -20.272, "expert": 3234.3},
        "hopper-medium-expert-v2": {"random": -20.272, "expert": 3234.3},
        "hopper-expert-v2": {"random": -20.272, "expert": 3234.3},
        "hopper-random-v2": {"random": -20.272, "expert": 3234.3},
        "walker2d-medium-v2": {"random": 1.629, "expert": 4592.3},
        "walker2d-medium-expert-v2": {"random": 1.629, "expert": 4592.3},
        "walker2d-expert-v2": {"random": 1.629, "expert": 4592.3},
        "walker2d-random-v2": {"random": 1.629, "expert": 4592.3},
        "pen-human-v1": {"random": -1, "expert": 3237.0},
        "pen-cloned-v1": {"random": -1, "expert": 3237.0},
        "pen-expert-v1": {"random": -1, "expert": 3237.0},
        "kitchen-mixed-v0": {"random": 0.0, "expert": 1.0},
        "maze2d-large-v1": {"random": 0.0, "expert": 1.0},
        "antmaze-large-diverse-v2": {"random": 0.0, "expert": 1.0},
    }

    if dataset_name not in d4rl_scores:
        # Default normalization if not found
        return returns

    bounds = d4rl_scores[dataset_name]
    random_score = bounds["random"]
    expert_score = bounds["expert"]

    import numpy as np
    normalized = (returns - random_score) / (expert_score - random_score)
    return np.clip(normalized, 0, 1) * 100
