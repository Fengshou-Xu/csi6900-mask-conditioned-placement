"""Train PPO on PandaPlaceCube (placement-only task).

Usage:
    .venv/bin/python phase1/train_place.py
"""
import functools
import pickle
import time

import jax
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

from place_1 import PandaPlaceCube


def main() -> None:
    env = PandaPlaceCube()

    # PPO hyperparameters, taken from MuJoCo Playground's PandaPickCube config
    ppo_params = dict(
        num_timesteps=20_000_000,
        num_envs=2048,
        batch_size=512,
        num_minibatches=32,
        num_updates_per_batch=8,
        unroll_length=10,
        num_evals=10,
        num_resets_per_eval=10,
        episode_length=150,
        action_repeat=1,
        discounting=0.97,
        learning_rate=1e-3,
        entropy_cost=0.02,
        reward_scaling=1.0,
        normalize_observations=True,
    )

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(32, 32, 32, 32),
        value_hidden_layer_sizes=(256, 256, 256, 256, 256),
        policy_obs_key="state",
        value_obs_key="state",
    )

    print("Training PPO on PandaPlaceCube")
    print(f"  num_envs    = {ppo_params['num_envs']}")
    print(f"  num_steps   = {ppo_params['num_timesteps']:,}")
    print(f"  num_evals   = {ppo_params['num_evals']}")
    print()

    t_start = time.time()

    def progress(num_steps: int, metrics: dict) -> None:
        elapsed = time.time() - t_start
        reward = float(metrics.get("eval/episode_reward", 0))
        sps = num_steps / elapsed if elapsed > 0 else 0
        print(
            f"  [t={elapsed:6.1f}s] step {num_steps:>12,}  "
            f"reward = {reward:8.2f}  ({sps:,.0f} steps/sec)"
        )

    make_inference_fn, params, metrics = ppo.train(
        environment=env,
        progress_fn=progress,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        network_factory=network_factory,
        seed=0,
        **ppo_params,
    )

    print()
    print(f"Done in {time.time() - t_start:.1f}s")
    print(f"Final eval reward: {float(metrics.get('eval/episode_reward', 0)):.2f}")

    # Save trained params to disk (GPU -> CPU via jax.device_get for pickle)
    params_cpu = jax.device_get(params)
    with open("place_params.pkl", "wb") as f:
        pickle.dump(params_cpu, f)
    print("Saved params to place_params.pkl")


if __name__ == "__main__":
    main()
