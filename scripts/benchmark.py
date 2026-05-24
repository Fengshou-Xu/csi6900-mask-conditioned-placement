import time
from typing import final

import jax
import jax.numpy as jnp
from mujoco_playground import registry

env = registry.load("PandaPickCubeOrientation")
key = jax.random.PRNGKey(0x0d000721)
state = env.reset(key)
N_STEPS = 100

def run_python_loop(state, key):
    for i in range(N_STEPS):
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(subkey, (env.action_size,), minval=-1, maxval=1)
        state = env.step(state, action)
    return state

def run_jax_scan(state,key):
    keys = jax.random.split(key, N_STEPS)
    actions = jax.vmap(
        lambda k: jax.random.uniform(k, (env.action_size,), minval=-1, maxval=1)
    )(keys)
    def scan_step(state,action):
        state = env.step(state,action)
        return state, state.reward
    final_state, rewards = jax.lax.scan(scan_step, state, actions)
    return final_state

def main():
    # pre-heat
    _ = env.step(state, jnp.zeros(env.action_size))
    jax_fn = jax.jit(run_jax_scan)
    jax_fn(state, key)  # 触发编译

    start = time.time()
    run_python_loop(state, key)
    print(f"Python loop: {time.time() - start:.3f}s")

    start = time.time()
    jax_fn(state, key)
    print(f"Jax scan :{time.time() - start:.3f}s")

if __name__ == '__main__':
    main()



