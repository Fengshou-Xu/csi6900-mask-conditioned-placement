"""A tiny placement variant of MuJoCo Playground's PandaPickCube.

Mental model:
  pick  = start with cube on table, learn "go grab it, then move it to target"
  place = start with cube already near/in the gripper, learn "move it to target"

This is intentionally a simple state-based baseline.  It does not use masks yet.
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.manipulation.franka_emika_panda import pick
from mujoco_playground._src.mjx_env import State


def default_config() -> config_dict.ConfigDict:
  """Config for the first placement baseline."""
  config = pick.default_config()
  config.episode_length = 100

  # Placement is mostly about moving the already-held cube to the target.
  config.reward_config.scales.gripper_box = 2.0
  config.reward_config.scales.box_target = 10.0
  config.reward_config.scales.robot_target_qpos = 0.1
  return config


class PandaPlaceCubeSimple(pick.PandaPickCube):
  """Start from a picked-up cube pose and place it at a fixed target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    xml_path = (
        mjx_env.ROOT_PATH
        / "manipulation"
        / "franka_emika_panda"
        / "xmls"
        / "mjx_single_cube_camera.xml"
    )
    panda.PandaBase.__init__(self, xml_path, config, config_overrides)

    # The camera XML has a useful keyframe called "picked": the cube starts
    # already elevated and close to the gripper.
    self._post_init(obj_name="box", keyframe="picked")
    self._sample_orientation = False

    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    ]

  def reset(self, rng: jax.Array) -> State:
    rng, rng_target = jax.random.split(rng)

    # First baseline: a nearly fixed tray/drop location on the table.
    # The cube is 0.06m tall, so z=0.03 places its center on the table.
    target_pos = jp.array([0.55, -0.12, 0.03], dtype=float)

    # Small random target jitter keeps the task from being only one exact point.
    target_pos += jax.random.uniform(
        rng_target,
        (3,),
        minval=jp.array([-0.03, -0.03, 0.0]),
        maxval=jp.array([0.03, 0.03, 0.0]),
    )

    target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    init_q = jp.array(self._init_q)
    data = mjx_env.make_data(
        self._mj_model,
        qpos=init_q,
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        ctrl=self._init_ctrl,
        impl=self._mjx_model.impl.value,
        naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax,
        njmax=self._config.njmax,
    )
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
    )
    data = mjx.forward(self._mjx_model, data)

    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},
    }

    # In pick.py this flag means "the gripper has reached the box at least once".
    # For placement we start after that phase, so it is already true.
    info = {"rng": rng, "target_pos": target_pos, "reached_box": 1.0}
    obs = self._get_obs(data, info)
    reward, done = jp.zeros(2)
    return State(data, obs, reward, done, metrics, info)

  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    rewards = super()._get_reward(data, info)

    # Make the beginner baseline pure placement: the box-target reward is always
    # active, instead of waiting for a "reached_box" event as in pick.py.
    target_pos = info["target_pos"]
    box_pos = data.xpos[self._obj_body]
    box_mat = data.xmat[self._obj_body]
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    pos_err = jp.linalg.norm(target_pos - box_pos)
    rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])
    rewards["box_target"] = 1 - jp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
    return rewards
