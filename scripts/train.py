"""Train PPO on PandaPickCubeOrientation with Brax + MuJoCo Playground.

Usage:
    uv run scripts/train.py
"""
import functools
import time

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import registry, wrapper
from mujoco_playground.config import manipulation_params

ENV_NAME = "PandaPickCubeOrientation"


def main() -> None:
    env = registry.load(ENV_NAME)
    ppo_params = manipulation_params.brax_ppo_config(ENV_NAME)

    network_kwargs = dict(ppo_params.network_factory)
    del ppo_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **network_kwargs
    )

    print(f"Training PPO on {ENV_NAME}")
    print(f"  num_envs    = {ppo_params.num_envs}")
    print(f"  num_steps   = {ppo_params.num_timesteps:,}")
    print(f"  num_evals   = {ppo_params.num_evals}")
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


if __name__ == "__main__":
    main()
