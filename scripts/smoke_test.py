"""Verify JAX + MuJoCo Playground + GPU pipeline is healthy.

Usage: uv run scripts/smoke_test.py
"""
import time
import jax
import jax.numpy as jnp
from mujoco_playground import registry

ENV_NAME = "PandaPickCubeOrientation"
NUM_ENVS = 4096


def main() -> None:
    print(f"jax.devices(): {jax.devices()}")
    assert jax.default_backend() == "gpu", "JAX is not using GPU"

    env = registry.load(ENV_NAME)
    print(f"Loaded {ENV_NAME}: obs={env.observation_size}, act={env.action_size}")

    keys = jax.random.split(jax.random.PRNGKey(0), NUM_ENVS)
    reset_fn = jax.jit(jax.vmap(env.reset))
    step_fn = jax.jit(jax.vmap(env.step))

    t0 = time.time()
    state = reset_fn(keys)
    jax.block_until_ready(state.reward)
    print(f"reset compile+run: {time.time() - t0:.2f}s")

    act = jnp.zeros((NUM_ENVS, env.action_size))
    t0 = time.time()
    state = step_fn(state, act)
    jax.block_until_ready(state.reward)
    print(f"first step compile+run: {time.time() - t0:.2f}s")

    t0 = time.time()
    for _ in range(100):
        state = step_fn(state, act)
    jax.block_until_ready(state.reward)
    elapsed = time.time() - t0
    print(f"100 steps × {NUM_ENVS} envs = {100 * NUM_ENVS:,} steps in {elapsed:.2f}s")
    print(f"throughput: {100 * NUM_ENVS / elapsed:,.0f} env-steps/sec")


if __name__ == "__main__":
    main()
