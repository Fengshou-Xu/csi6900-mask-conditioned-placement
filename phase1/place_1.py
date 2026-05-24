import jax
import jax.numpy as jnp
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import pick, panda
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member



def default_config() -> config_dict.ConfigDict:
    """Based on pick.default_config(), with gripper_box reward removed."""
    config = pick.default_config()
    config.reward_config.scales = config_dict.create(
        box_target = 8.0,
        no_floor_collision = 0.25,
        robot_target_qpos=0.3,
    )
    return config

class PandaPlaceCube(pick.PandaPickCube):
    def __init__(self,
                 config: config_dict.ConfigDict = default_config(),
                 config_overrides = None,
                 ):
        xml_path = (
            mjx_env.ROOT_PATH
            / "manipulation"
            / "franka_emika_panda"
            / "xmls"
            / "mjx_single_cube_camera.xml"
        )

        # Skip PandaPickCube.__init__ (it hardcodes the wrong XML and keyframe),
        # call PandaBase directly to load model.
        panda.PandaBase.__init__(self, xml_path, config, config_overrides)

        # "picked" keyframe: box already in gripper (see mjx_single_cube_camera.xml)
        self._post_init(obj_name="box", keyframe="picked")

        self._sample_orientation = False

        self._floor_hand_found_sensor = [
            self._mj_model.sensor(f"{geom}_floor_found").id
            for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
        ]

    def reset(self, rng: jax.Array) -> State:
        rng, rng_target = jax.random.split(rng)
        # Only 1 rng needed (no box randomization; keyframe handles initial box pose)

        # Target on table surface, offset from box start position.
        # z=0.03 = half box height, so box center sits exactly on surface.
        table_z = 0.03
        target_pos = (
            jax.random.uniform(
                rng_target,
                (3,),
                minval=jnp.array([-0.1, -0.1, 0.0]),
                maxval=jnp.array([0.1, 0.1, 0.0]),
            )
            + jnp.array([0.55, 0.0, table_z])
        )

        target_quat = jnp.array([1.0, 0.0, 0.0, 0.0])  # no rotation

        # Use picked keyframe qpos as-is (box already in gripper)
        init_q = jnp.array(self._init_q)

        data = mjx_env.make_data(
            self._mj_model,
            qpos=init_q,
            qvel=jnp.zeros(self._mjx_model.nv, dtype=float),
            ctrl=self._init_ctrl,
            impl=self._mjx_model.impl.value,
            naconmax=self._config.naconmax,
            naccdmax=self._config.naccdmax,
            njmax=self._config.njmax,
        )

        # Set target mocap body position
        data = data.replace(
            mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
            mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
        )

        # Compute forward kinematics so xpos/site_xpos are valid immediately
        data = mjx.forward(self._mjx_model, data)

        # Keys must match default_config().reward_config.scales
        metrics = {
            "out_of_bounds": jnp.array(0.0, dtype=float),
            **{k: 0.0 for k in self._config.reward_config.scales.keys()},
        }

        # No "reached_box" — box is already grasped from step 0
        info = {"rng": rng, "target_pos": target_pos}

        obs = self._get_obs(data, info)
        reward, done = jnp.zeros(2)
        return State(data, obs, reward, done, metrics, info)


    def _get_reward(self, data: mjx.Data, info: dict) -> dict:
        """Distance-based placement reward. No gripper_box or reached_box."""
        target_pos = info["target_pos"]
        box_pos = data.xpos[self._obj_body]

        # Primary reward: box-to-target distance (same formula as pick.py)
        dist = jnp.linalg.norm(target_pos - box_pos)
        box_target = 1 - jnp.tanh(5.0 * dist)

        # Regularization: penalize arm deviation from home pose
        robot_target_qpos = 1 - jnp.tanh(
            jnp.linalg.norm(
                data.qpos[self._robot_arm_qposadr]
                - self._init_q[self._robot_arm_qposadr]
            )
        )

        # Safety: detect hand-floor collisions
        hand_floor_collision = [
            data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
            for sensor_id in self._floor_hand_found_sensor
        ]
        floor_collision = sum(hand_floor_collision) > 0
        no_floor_collision = (1 - floor_collision).astype(float)

        # Keys must match default_config().reward_config.scales
        return {
            "box_target": box_target,
            "no_floor_collision": no_floor_collision,
            "robot_target_qpos": robot_target_qpos,
        }
