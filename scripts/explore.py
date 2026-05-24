import jax
import jax.numpy as jnp
import mujoco_playground
from mujoco_playground import registry


env = registry.load("PandaPickCubeOrientation")

key = jax.random.PRNGKey(0x0d000721)
initial_state = env.reset(key)

state = initial_state
rewards = []
for i in range(150):
    key, subkey = jax.random.split(key)
    action = jax.random.uniform(subkey, (env.action_size,), minval=-1, maxval=1)
    state = env.step(state, action)
    rewards.append(float(state.reward))

print("First 10 rewards:", rewards[:10])
print(f"max = {max(rewards):.3f}, min = {min(rewards):.3f}")
print("Done after 150 steps:", float(state.done))