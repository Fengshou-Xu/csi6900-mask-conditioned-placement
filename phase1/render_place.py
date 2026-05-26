"""Load saved params and render a placement episode to video.

Usage:
    Run train_place.py first to save params -> place_params.pkl
    Then: .venv/bin/python phase1/render_place.py
"""
import functools
import pickle

import jax
import jax.numpy as jnp
import mediapy
import mujoco
from brax.training.acme import running_statistics
from brax.training.agents.ppo import networks as ppo_networks

from datetime import datetime

from place_1 import PandaPlaceCube

EPISODE_LENGTH = 150


def main():
    # --- 1. Create environment ---
    env = PandaPlaceCube()

    # --- 2. Rebuild network structure (must match training config exactly) ---
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(32, 32, 32, 32),
        value_hidden_layer_sizes=(256, 256, 256, 256, 256),
        policy_obs_key="state",
        value_obs_key="state",
    )
    # Training used normalize_observations=True, so we need the same preprocessor
    ppo_network = network_factory(
        env.observation_size,
        env.action_size,
        preprocess_observations_fn=running_statistics.normalize,
    )
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)

    # --- 3. Load trained params ---
    with open("place_params.pkl", "rb") as f:
        params = pickle.load(f)
    print("Loaded params from place_params.pkl")

    # --- 4. Create policy ---
    policy = make_inference_fn(params)

    # --- 5. Run one episode ---
    key = jax.random.PRNGKey(0x0d000721)
    state = env.reset(key)

    # --- 6. Initialize renderer ---
    mj_model = env.mj_model
    # Make target marker visually distinct: green, semi-transparent, flat disc
    target_body_id = mj_model.body("mocap_target").id
    for i in range(mj_model.ngeom):
        if mj_model.geom_bodyid[i] == target_body_id:
            mj_model.geom_rgba[i] = [0, 1, 0, 0.5]  # green, 50% transparent
            mj_model.geom_size[i] = [0.05, 0.05, 0.005]  # flat disc shape
            break

    renderer = mujoco.Renderer(mj_model, width=640, height=480)
    mj_data = mujoco.MjData(mj_model)
    frames = []
    prev_box_pos = None
    stable_count = 0

    for i in range(EPISODE_LENGTH):
        key, subkey = jax.random.split(key)
        action, _ = policy(state.obs, subkey)
        state = env.step(state, action)

        # Transfer GPU state to CPU renderer
        mj_data.qpos[:] = jax.device_get(state.data.qpos)
        mj_data.mocap_pos[:] = jax.device_get(state.data.mocap_pos)
        mj_data.mocap_quat[:] = jax.device_get(state.data.mocap_quat)
        mujoco.mj_forward(mj_model, mj_data)
        renderer.update_scene(mj_data)
        frames.append(renderer.render())

        box_pos = jax.device_get(state.data.xpos[env._obj_body])
        target_pos = state.info["target_pos"]
        dist = float(jnp.linalg.norm(box_pos - target_pos))

        # Detect when box has settled (position change < 1mm)
        if prev_box_pos is not None:
            moved = float(jnp.linalg.norm(box_pos - prev_box_pos))
            if moved < 0.001:
                stable_count += 1
            else:
                stable_count = 0
        prev_box_pos = box_pos

        if i % 10 == 0:
            print(f"step {i:3d}: dist={dist:.4f}, stable={stable_count}")

        # Stop recording after box is stable for 20 consecutive frames
        if stable_count >= 20:
            print(f"Box settled at step {i+1}, dist={dist:.4f}")
            break

    # --- 7. Write video ---
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = "place_demo_" + now + ".mp4"
    mediapy.write_video(file_name, frames, fps=50)
    print(f"Saved video: {file_name} ({len(frames)} frames, {len(frames)/50:.1f}s)")


if __name__ == "__main__":
    main()
