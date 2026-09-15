<h1 align="center">🌹 Unifloral: Unified Offline Reinforcement Learning</h1>

<p align="center">
    <a href= "https://arxiv.org/abs/2504.11453">
        <img src="https://img.shields.io/badge/arXiv-2504.11453-b31b1b.svg" /></a>
</p>

Unified implementations and rigorous evaluation for offline reinforcement learning - built by [Matthew Jackson](https://github.com/EmptyJackson), [Uljad Berdica](https://github.com/uljad), and [Jarek Liesen](https://github.com/keraJLi).

## 💡 Code Philosophy

- ⚛️ **Single-file**: We implement algorithms as standalone Python files.
- 🤏 **Minimal**: We only edit what is necessary between algorithms, making comparisons straightforward.
- ⚡️ **GPU-accelerated**: We use JAX and end-to-end compile all training code, enabling lightning-fast training.

Inspired by [CORL](https://github.com/tinkoff-ai/CORL) and [CleanRL](https://github.com/vwxyzjn/cleanrl) - check them out!

## 🤖 Algorithms

We provide two types of algorithm implementation:

1. **Standalone**: Each algorithm is implemented as a [single file](algorithms) with minimal dependencies, making it easy to understand and modify.
2. **Unified**: Most algorithms are available as configs for our unified implementation [`unifloral.py`](algorithms/unifloral.py).

After training, final evaluation results are saved to `.npz` files in [`final_returns/`](final_returns) for analysis using our evaluation protocol.

All scripts support [D4RL](https://github.com/Farama-Foundation/D4RL) and use [Weights & Biases](https://wandb.ai) for logging, with configs provided as WandB sweep files.

### Model-free

| Algorithm | Standalone | Unified | Extras |
| --- | --- | --- | --- |
| BC | [`bc.py`](algorithms/bc.py) | [`unifloral/bc.yaml`](configs/unifloral/bc.yaml) | - |
| SAC-N | [`sac_n.py`](algorithms/sac_n.py) | [`unifloral/sac_n.yaml`](configs/unifloral/sac_n.yaml) | [[ArXiv]](https://arxiv.org/abs/2110.01548) |
| EDAC | [`edac.py`](algorithms/edac.py) | [`unifloral/edac.yaml`](configs/unifloral/edac.yaml) | [[ArXiv]](https://arxiv.org/abs/2110.01548) |
| CQL | [`cql.py`](algorithms/cql.py) | - | [[ArXiv]](https://arxiv.org/abs/2006.04779) |
| IQL | [`iql.py`](algorithms/iql.py) | [`unifloral/iql.yaml`](configs/unifloral/iql.yaml) | [[ArXiv]](https://arxiv.org/abs/2110.06169) |
| TD3-BC | [`td3_bc.py`](algorithms/td3_bc.py) | [`unifloral/td3_bc.yaml`](configs/unifloral/td3_bc.yaml) | [[ArXiv]](https://arxiv.org/abs/2106.06860) |
| ReBRAC | [`rebrac.py`](algorithms/rebrac.py) | [`unifloral/rebrac.yaml`](configs/unifloral/rebrac.yaml) | [[ArXiv]](https://arxiv.org/abs/2305.09836) |
| TD3-AWR | - | [`unifloral/td3_awr.yaml`](configs/unifloral/td3_awr.yaml) | [[ArXiv]](https://arxiv.org/abs/2504.11453) |

### Model-based

We implement a single script for dynamics model training: [`dynamics.py`](algorithms/dynamics.py), with config [`dynamics.yaml`](configs/dynamics.yaml).

| Algorithm | Standalone | Unified | Extras |
| --- | --- | --- | --- |
| MOPO | [`mopo.py`](algorithms/mopo.py) | - | [[ArXiv]](https://arxiv.org/abs/2005.13239) |
| MOReL | [`morel.py`](algorithms/morel.py) | - | [[ArXiv]](https://arxiv.org/abs/2005.05951) |
| COMBO | [`combo.py`](algorithms/combo.py) | - | [[ArXiv]](https://arxiv.org/abs/2102.08363) |
| MoBRAC | - | [`unifloral/mobrac.yaml`](configs/unifloral/mobrac.yaml) | [[ArXiv]](https://arxiv.org/abs/2504.11453) |

New ones coming soon 👀

## 📚 Dataset Paths

This repository uses **native Minari dataset IDs** (no legacy D4RL package). Use these IDs with `--dataset` or in config files:

### MuJoCo Locomotion
Updated Minari datasets include only `expert-v0`, `medium-v0`, and `simple-v0` variants per environment. Splits like `medium-expert` and `medium-replay` are no longer available.

| Old D4RL | New Minari | Notes |
| --- | --- | --- |
| `hopper-medium-v2` | `mujoco/hopper/medium-v0` | ✓ Available |
| `halfcheetah-medium-v2` | `mujoco/halfcheetah/medium-v0` | ✓ Available |
| `halfcheetah-medium-expert-v2` | — | ✗ Use `mujoco/halfcheetah/expert-v0` or `medium-v0` |
| `walker2d-medium-v2` | `mujoco/walker2d/medium-v0` | ✓ Available |
| `walker2d-medium-replay-v2` | — | ✗ Replay splits not available in Minari |

### Adroit Hand Tasks
| Old D4RL | New Minari |
| --- | --- |
| `pen-human-v1` | `D4RL/pen/human-v2` |
| `pen-cloned-v1` | `D4RL/pen/cloned-v2` |
| `pen-expert-v1` | `D4RL/pen/expert-v2` |

### Navigation & Kitchen Tasks
| Old D4RL | New Minari |
| --- | --- |
| `kitchen-mixed-v0` | `D4RL/kitchen/mixed-v2` |
| `pointmaze-large-v0` | `D4RL/pointmaze/large-v2` |
| `antmaze-large-diverse-v2` | `D4RL/antmaze/large-diverse-v2` |

**Example usage:**
```bash
python3 algorithms/bc.py --dataset D4RL/pen/human-v2
python3 algorithms/iql.py --dataset mujoco/hopper/medium-v0
python3 algorithms/cql.py --dataset mujoco/halfcheetah/expert-v0
```

**Normalized Score Normalization:**
Minari's `get_normalized_score()` normalizes episode returns to a [0, 100] scale using reference bounds stored in each dataset:
- **D4RL-derived datasets** (`D4RL/*`, e.g., `D4RL/pen/human-v2`, `D4RL/antmaze/large-play-v2`) include `ref_min_score` and `ref_max_score` metadata, computed from random and expert policy returns. The normalized score formula is: `(return - ref_min) / (ref_max - ref_min) × 100`
- **MuJoCo datasets** (`mujoco/*`, e.g., `mujoco/halfcheetah/medium-v0`) do not include these reference bounds. When reference scores are unavailable, `get_normalized_score()` falls back gracefully and returns raw (unnormalized) returns.

**Practical impact:** When reporting results, D4RL-trained policies will show normalized scores (0–100 scale, where 50 = random, 100 = expert); MuJoCo-trained policies will show raw episode returns. Normalize manually if needed by dividing by typical expert return for the task.

## 📊 Evaluation

Our evaluation script ([`evaluation.py`](evaluation.py)) implements the protocol described in our paper, analysing the performance of a UCB bandit over a range of policy evaluations.

```python
from evaluation import load_results_dataframe, bootstrap_bandit_trials
import jax.numpy as jnp

# Load all results from the final_returns directory
df = load_results_dataframe("final_returns")

# Run bandit trials with bootstrapped confidence intervals
results = bootstrap_bandit_trials(
    returns_array=jnp.array(policy_returns),  # Shape: (num_policies, num_rollouts)
    num_subsample=8,     # Number of policies to subsample
    num_repeats=1000,    # Number of bandit trials
    max_pulls=200,       # Maximum pulls per trial
    ucb_alpha=2.0,       # UCB exploration coefficient
    n_bootstraps=1000,   # Bootstrap samples for confidence intervals
    confidence=0.95      # Confidence level
)

# Access results
pulls = results["pulls"]                      # Number of pulls at each step
means = results["estimated_bests_mean"]       # Mean score of estimated best policy
ci_low = results["estimated_bests_ci_low"]    # Lower confidence bound
ci_high = results["estimated_bests_ci_high"]  # Upper confidence bound
```

## 📝 Cite us!
```bibtex
@misc{jackson2025clean,
      title={A Clean Slate for Offline Reinforcement Learning},
      author={Matthew Thomas Jackson and Uljad Berdica and Jarek Liesen and Shimon Whiteson and Jakob Nicolaus Foerster},
      year={2025},
      eprint={2504.11453},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.11453},
}
```
