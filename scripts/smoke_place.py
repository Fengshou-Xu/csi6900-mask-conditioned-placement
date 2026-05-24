"""Run a small smoke test for the custom PandaPlaceCubeSimple environment.

Usage:
  .venv/bin/python scripts/smoke_place.py
"""

import jax
import jax.numpy as jnp

from place_env import PandaPlaceCubeSimple


def main() -> None:
  env = PandaPlaceCubeSimple()
  print(f"Loaded PandaPlaceCubeSimple: obs={env.observation_size}, act={env.action_size}")
  print(f"jax.devices(): {jax.devices()}")

  state = env.reset(jax.random.PRNGKey(0))
  gripper = state.data.site_xpos[env._gripper_site]
  box = state.data.xpos[env._obj_body]
  target = state.info["target_pos"]

  print("\nReset positions")
  print(f"  gripper: {gripper}")
  print(f"  box:     {box}")
  print(f"  target:  {target}")
  print(f"  box-target distance: {float(jnp.linalg.norm(box - target)):.4f} m")

  print("\n5 zero-action steps")
  for i in range(5):
    state = env.step(state, jnp.zeros(env.action_size))
    box = state.data.xpos[env._obj_body]
    target = state.info["target_pos"]
    print(
        f"  step {i + 1}: reward={float(state.reward):.4f}, "
        f"box-target={float(jnp.linalg.norm(box - target)):.4f} m, "
        f"done={float(state.done)}"
    )


if __name__ == "__main__":
  main()

