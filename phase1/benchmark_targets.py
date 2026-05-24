"""Benchmark different fixed target positions.

Each config runs in a separate subprocess to avoid GPU OOM.

Usage:
    .venv/bin/python phase1/benchmark_targets.py
"""
import subprocess
import sys
import json

CONFIGS = [
    ("A: directly below",       [0.637, 0.006, 0.03], 0.0),
    ("B: below + range 0.15",   [0.637, 0.006, 0.03], 0.15),
    ("C: x=0.55 (closer)",      [0.55,  0.0,   0.03], 0.0),
    ("D: x=0.55 + range 0.1",   [0.55,  0.0,   0.03], 0.1),
    ("E: offset y=0.15",        [0.637, 0.15,  0.03], 0.0),
    ("F: x=0.5 y=0.1",         [0.5,   0.1,   0.03], 0.0),
]

# Worker script that trains one config and prints JSON result on last line
WORKER = r'''
import functools, time, json, sys
import jax, jax.numpy as jnp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import pick, panda
from mujoco_playground._src.mjx_env import State
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

target_pos = json.loads(sys.argv[1])
target_range = float(sys.argv[2])
label = sys.argv[3]

def default_config():
    config = pick.default_config()
    config.reward_config.scales = config_dict.create(
        box_target=8.0, no_floor_collision=0.25, robot_target_qpos=0.3)
    return config

class Env(pick.PandaPickCube):
    def __init__(self):
        config = default_config()
        xml_path = (mjx_env.ROOT_PATH / "manipulation" / "franka_emika_panda"
                    / "xmls" / "mjx_single_cube_camera.xml")
        panda.PandaBase.__init__(self, xml_path, config, None)
        self._post_init(obj_name="box", keyframe="picked")
        self._sample_orientation = False
        self._floor_hand_found_sensor = [
            self._mj_model.sensor(f"{geom}_floor_found").id
            for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]]

    def reset(self, rng):
        rng, rng_t = jax.random.split(rng)
        center = jnp.array(target_pos)
        if target_range > 0:
            tp = jax.random.uniform(rng_t, (3,),
                minval=jnp.array([-target_range, -target_range, 0.0]),
                maxval=jnp.array([target_range, target_range, 0.0])) + center
        else:
            tp = center
        init_q = jnp.array(self._init_q)
        data = mjx_env.make_data(self._mj_model, qpos=init_q,
            qvel=jnp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=self._init_ctrl, impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax, naccdmax=self._config.naccdmax,
            njmax=self._config.njmax)
        tq = jnp.array([1.0, 0.0, 0.0, 0.0])
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(tp),
            mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(tq))
        data = mjx.forward(self._mjx_model, data)
        metrics = {"out_of_bounds": jnp.array(0.0, dtype=float),
                   **{k: 0.0 for k in self._config.reward_config.scales.keys()}}
        info = {"rng": rng, "target_pos": tp}
        obs = self._get_obs(data, info)
        reward, done = jnp.zeros(2)
        return State(data, obs, reward, done, metrics, info)

    def _get_reward(self, data, info):
        tp = info["target_pos"]
        bp = data.xpos[self._obj_body]
        dist = jnp.linalg.norm(tp - bp)
        bt = 1 - jnp.tanh(5.0 * dist)
        rq = 1 - jnp.tanh(jnp.linalg.norm(
            data.qpos[self._robot_arm_qposadr] - self._init_q[self._robot_arm_qposadr]))
        hfc = [data.sensordata[self._mj_model.sensor_adr[s]] > 0
               for s in self._floor_hand_found_sensor]
        nfc = (1 - (sum(hfc) > 0)).astype(float)
        return {"box_target": bt, "no_floor_collision": nfc, "robot_target_qpos": rq}

env = Env()
t0 = time.time()
def prog(n, m):
    r = float(m.get("eval/episode_reward", 0))
    elapsed = time.time() - t0
    print(f"  [{label}] step {n:>12,}  reward={r:.1f}  ({elapsed:.0f}s)", flush=True)

_, _, m = ppo.train(
    environment=env, progress_fn=prog,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    network_factory=functools.partial(ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=(32,32,32,32),
        value_hidden_layer_sizes=(256,256,256,256,256),
        policy_obs_key="state", value_obs_key="state"),
    seed=0, num_timesteps=20_000_000, num_envs=2048, batch_size=512,
    num_minibatches=32, num_updates_per_batch=8, unroll_length=10,
    num_evals=4, num_resets_per_eval=10, episode_length=150,
    action_repeat=1, discounting=0.97, learning_rate=1e-3,
    entropy_cost=0.02, reward_scaling=1.0, normalize_observations=True)

final = float(m.get("eval/episode_reward", 0))
elapsed = time.time() - t0
print(f"RESULT_JSON:{json.dumps({'label': label, 'reward': final, 'time': elapsed})}")
'''


def main():
    results = []
    for label, target, rng_range in CONFIGS:
        print(f"\n{'='*60}")
        print(f"Testing: {label}  target={target}  range=+-{rng_range}")
        print(f"{'='*60}", flush=True)

        proc = subprocess.run(
            [sys.executable, "-c", WORKER,
             json.dumps(target), str(rng_range), label],
            capture_output=True, text=True,
        )
        # Print training progress
        print(proc.stdout)
        if proc.stderr:
            # Filter out warp warnings, only show errors
            for line in proc.stderr.split("\n"):
                if "error" in line.lower() and "opengl" not in line.lower():
                    print(f"  STDERR: {line}")

        # Extract result
        for line in proc.stdout.split("\n"):
            if line.startswith("RESULT_JSON:"):
                data = json.loads(line[len("RESULT_JSON:"):])
                results.append(data)
                break

    print(f"\n\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'Reward':>8} {'Time':>6}")
    print("-" * 48)
    for r in results:
        print(f"{r['label']:<30} {r['reward']:>8.1f} {r['time']:>5.0f}s")


if __name__ == "__main__":
    main()
