# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Bring a box to a target and orientation."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.franka_emika_panda import panda
from mujoco_playground._src.mjx_env import State  # pylint: disable=g-importing-member
import numpy as np


def default_config() -> config_dict.ConfigDict:
  """Returns the default config for bring_to_target tasks."""
  config = config_dict.create(
      sim_dt=0.005,  # physics engine takes 0.005s to simulate each step(200Hz)
      ctrl_dt=0.02,    # agent takes 0.02s to make a decision (50Hz), i.e., for each action, the physics engine will pass 0.02/0.005 = 4 steps
      episode_length=150,  # one episode lasts for 150 steps maximum, which takes 150*0.02 = 3s to simulate
      action_repeat=1, # each action executes once (no repeat) | this value co-work with ctrl_dt.
      # E.g. if action_repeat = 1 & ctrl_dt = 0.02, then the agent will make decisions per 0.02s, a total of 50 decisions per second
      # However, if action_repeat = 3 while ctrl_dt keeps the same (0.02), after each agent decision, it will repeat that action for 3 consecutive ctrl_dt steps, totaling 0.06s
      # During the repetition, agent will be "frozen" and do not make any decision.
      action_scale=0.04, # agent output will be scaled to [-out*action_scale, out*action_scale], where out is [-1.0, 1.0]
      # ● agent 不是在说"往上"或"往左"。它输出的是 8 个数字，每个对应一个关节：
      # 
      #   action[0] → joint1（底座旋转）
      #   action[1] → joint2（肩膀上下）
      #   action[2] → joint3（底座旋转）
      #   action[3] → joint4（肘部弯曲）
      #   action[4] → joint5（前臂旋转）
      #   action[5] → joint6（腕部弯曲）
      #   action[6] → joint7（腕部旋转）
      #   action[7] → 手指开合
      #
      #   每个关节只能沿一个固定轴转动（在 XML 里定义好了的）。正值 = 沿这个轴正方向转，负值 = 反方向转。
      #
      #   所以 agent 不需要理解"上下左右"。它学到的是："如果我想让 gripper 往右移，我应该同时让 joint1 转一点、joint2 转一点、joint4
      #   转一点……" 这种多个关节的组合，通过试错慢慢学会。
      #
      #   打个比方：你的手臂也是这样工作的。你不会想"手往右移 5
      #   厘米"——你的大脑发出的是"肩膀肌肉收缩一点、肘部肌肉放松一点"这样的信号。多个关节配合的结果才是手往右移了。agent
      #   也是一样，只不过它是通过几百万次尝试学会这种配合的。
      reward_config=config_dict.create(
          scales=config_dict.create(
              # Gripper goes to the box. (box is the target object)
              gripper_box=4.0,
              # Box goes to the target mocap.
              box_target=8.0,  # (Target is the destination of the "box". In this case, it will be a point floating in the air of the virtual environment)
              # Thus, the task is clear: the gripper needs to reach the box, then move the box to where the target is.
              # Do not collide the gripper with the floor.
              no_floor_collision=0.25,
              # Arm stays close to target pose.
              robot_target_qpos=0.3,
          )
      ),
      impl='warp',
      naconmax=24 * 2048,
      naccdmax=24 * 2048,
      njmax=128,
  )
  return config


class PandaPickCube(panda.PandaBase):
  """Bring a box to a target."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),   # load default config, we just saw it at line 30
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None, # it allows user to adjust specific parameters without rewrite the whole config,
      #for example, 比如你只想把 episode 长度从 150 改成 300: env = PandaPickCube(config_overrides={"episode_length": 300})
      sample_orientation: bool = False,
      # 是否给 target 一个随机旋转角度。
      #   - False：agent 只需要把 box 搬到正确位置
      #   - True：agent 还需要把 box 转到正确朝向（最多 45°）
  ):
    xml_path = ( # 拼出 XML 文件的路径。这个 XML 定义了整个物理场景——机械臂长什么样、桌子在哪、cube 有多大多重、重力多少。 The same as the xml in the live demo
        mjx_env.ROOT_PATH
        / "manipulation"
        / "franka_emika_panda"
        / "xmls"
        / "mjx_single_cube.xml"
    )
    super().__init__( #call super class PandaBase to:
        # 1. read XML;
        # 2. mujoco.MjModel.from_xml_string() parse XML to MuJoCo module object;
        # Note: It will generate 2 module objects:
        # 在 PandaBase.__init__ 里（panda.py:76-82）：
        # mj_model = mujoco.MjModel.from_xml_string(xml, ...)   # XML → MuJoCo 模型对象
        # self._mj_model = mj_model                              # 存 CPU 版
        # self._mjx_model = mjx.put_model(mj_model, impl="warp") # 转 GPU/JAX 版
        # 3. mjx.put_model(): 把 MuJoCo 模型转成 MJX 格式（纯 JAX 数组）
        #    impl="warp" 启用 NVIDIA Warp 后端，可在 GPU 上加速
        #    不传 impl 则默认在 CPU 上跑（仍比原生 MuJoCo 快，因为 JAX jit 编译）
        xml_path,
        config,
        config_overrides,
    )
    self._post_init(obj_name="box", keyframe="home")  # keyframe is a pre-defined snapshot gesture of the robot arm.
    # It is being used to describe that at the current moment the joint angle and the position of all objects
    # 你只需要记住 _post_init 之后，self 上挂了这些属性可以用：
    # self._obj_body: box 的 ID（查位置用）
    # self._gripper_site: gripper 的 ID（查位置用）
    # self._mocap_target: target 的 ID
    # self._init_q: 整个模拟世界的初始状态 所有关节角度 + 自由物体的 3D 位置 + 四元数朝向
    # self._init_obj_pos: box 的初始 xyz 位置
    # self._init_ctrl: 初始控制信号
    # self._lowers / self._uppers: 各关节控制范围
    # self._robot_arm_qposadr: 7 个手臂关节在 qpos 数组里的索引位置
    self._sample_orientation = sample_orientation  # whether to randomize the orientation of the target

    # Contact sensor IDs.
    # Contact sensors are used to detect collisions of the arm.
    # 这些传感器定义在 XML 里，作用是检测"手的某个部位是否碰到了地板"：
    # left_finger_pad_floor_found: 左手指垫是否碰到了地板
    # right_finger_pad_floor_found: 右手指垫是否碰到了地板
    # hand_capsule_floor_found: 手掌是否碰到了地板
    self._floor_hand_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in ["left_finger_pad", "right_finger_pad", "hand_capsule"]
    ]

  def reset(self, rng: jax.Array) -> State:    # start a new episode like new game
    rng, rng_box, rng_target = jax.random.split(rng, 3)
    #JAX's random number logic is different from normal random method like random.random() in general python
    #It needs to work with JIT and get the same result when repetition.
    #Thus, key = jax.random.PRNGKey("somenumber") is an array like [0, 0x0d000721]
    #Here, "key" is not a random number but a token for a random number generator
    #Then we can use it: x = jax.random.uniform(key, (3,))  # 用 key 生成 3 个随机数
    #但是！ 如果你用同一个 "key" 两次，你会得到完全相同的结果。这是故意的（可复现性）。所以你不能重复使用同一个 key。
    #split 就是"从一个 "key" 生成多个新的、互相独立的 key"：key1, key2, key3 = jax.random.split(key, 3)
    # 不是把一个随机数切三份。它是一个确定性的哈希函数——输入一个 "key" ，输出 3 个新 key，这 3 个 key 之间没有统计关联。原来的
    # "key" 用完就丢掉了。

    # intialize box position
    box_pos = (
        jax.random.uniform(
            rng_box,
            (3,),  # generate an offset, as an array (vector actually) like [x,y,z] where x=Front, back; y=left,right; z=height
            #min/max val is the range of the offset [x,y,z], i.e. [+-0.2, +-0.2, 0]
            #Note: no Z offset !
            minval=jp.array([-0.2, -0.2, 0.0]),
            maxval=jp.array([0.2, 0.2, 0.0]),
        )
        + self._init_obj_pos #The default position of the box in the XML
        #Thus, it will be something like [x_default + x_offset, y_default + y_offset, z_default + z_offset]
    )

    # initialize target position
    # Same logic, check the comment above.
    # But note, here do have Z offset! 0.2 ~ 0.4
    # Target is in the air, thus the agent must move the box upward
    target_pos = (
        jax.random.uniform(
            rng_target,
            (3,),
            minval=jp.array([-0.2, -0.2, 0.2]),
            maxval=jp.array([0.2, 0.2, 0.4]),
        )
        + self._init_obj_pos
    )

    # randomize the target orientation
    target_quat = jp.array([1.0, 0.0, 0.0, 0.0], dtype=float)  #[1.0, 0.0, 0.0, 0.0] mean no rotation; Quaternion (四元数)
    if self._sample_orientation:
      # sample a random direction
      rng, rng_axis, rng_theta = jax.random.split(rng, 3)
      perturb_axis = jax.random.uniform(rng_axis, (3,), minval=-1, maxval=1)
      perturb_axis = perturb_axis / math.norm(perturb_axis)
      perturb_theta = jax.random.uniform(rng_theta, maxval=np.deg2rad(45))
      target_quat = math.axis_angle_to_quat(perturb_axis, perturb_theta) #math.axis_angle_to_quat requires a quaternion or raise error

    # initialize data
    init_q = (
        jp.array(self._init_q)  # self._init_q: 整个模拟世界的初始状态 所有关节角度 + 自由物体的 3D 位置 + 四元数朝向 通过 self._post_init(obj_name="box", keyframe="home") 获得
        #把 keyframe "home" 的 qpos 转成 JAX 数组。_init_q 是长度为 16 的向量（7 个手臂关节 + 2 个手指关节 + 7 个 box free joint = 16）。
        .at[self._obj_qposadr : self._obj_qposadr + 3]  # _obj_qposadr 是 box 的 freejoint 在 qpos 数组中的起始索引，值为 9（前面有 7 个手臂关节 + 2 个手指关节）。
        # freejoint 在 qpos 里占 7 个数：
        # bx, by, bz,  qw, qx, qy, qz
        # ─ 位置(3) ─  ── 旋转四元数(4) ──
        .set(box_pos)   #把位置的 3 个值替换成前面生成的随机 box_pos。
        # 结果 init_q 是一个新的数组，只有 box 位置变了，其他所有关节值保持 home 姿态。
        #这里是jax的数组操作。和一般的python语法不同，不能直接使用slice语法。必须通过 array.at["slice"].set("value") 来操作目标
        # === 基础：普通 Python list 怎么改元素 ===
        # lst = [10, 20, 30, 40, 50]
        # lst[2] = 99           # 直接改第3个元素
        # # lst = [10, 20, 99, 40, 50]
        #
        # === JAX 不能这样做，因为数组是 immutable（不可变）===
        # arr = jp.array([10, 20, 30, 40, 50])
        # # arr[2] = 99   ← 报错！JAX 不允许
        #
        # === 所以用 .at[].set()，它返回一个新数组 ===
        # arr2 = arr.at[2].set(99)
        # # arr  还是 [10, 20, 30, 40, 50]  （没变）
        # # arr2 是   [10, 20, 99, 40, 50]  （新的）
        #
        # === 用切片改连续多个元素 ===
        # arr3 = arr.at[1:4].set(jp.array([77, 88, 99]))
        # # arr3 = [10, 77, 88, 99, 50]
        # #              ↑ index 1,2,3 被替换了
        #
        # === 用变量做索引 ===
        # start = 1
        # arr4 = arr.at[start : start + 3].set(jp.array([77, 88, 99]))
        # # 和上面完全一样：arr4 = [10, 77, 88, 99, 50]
    )
    data = mjx_env.make_data(
        self._mj_model,
        #Recall:
        #     super().__init__
        # Also in panda.py, we have:
        # mj_model = mujoco.MjModel.from_xml_string(xml, assets=self._model_assets)
        # self._mj_model = mj_model
        # Thus self._mj_model is a MuJoCo model object.
        qpos=init_q, # check the init_q above
        qvel=jp.zeros(self._mjx_model.nv, dtype=float),
        # nv: number of degrees of freedom
        # 初速度全零，nv=15（自由度，比qpos少1因为四元数4→角速度3）
        # qpos 和 qvel 的长度可以不一样。
        #   qpos (位置): [j1 j2 j3 j4 j5 j6 j7 f1 f2 bx by bz qw qx qy qz] = 16个
        #   qvel (速度): [j1 j2 j3 j4 j5 j6 j7 f1 f2 vx vy vz wx wy wz]     = 15个
        # 手臂和手指部分一一对应，都是 9 个。区别在 box：
        #   - 位置用 7 个数：xyz 坐标(3) + 四元数(4)
        #   - 速度用 6 个数：线速度(3) + 角速度(3)
        #   四元数用 4 个数表示旋转，但旋转的自由度只有 3 个（绕 x 转、绕 y 转、绕 z 转）。多出来的那 1
        #   个数是因为四元数的数学表示需要 4 个分量，但实际自由度还是 3。
        #   所以 nv = degrees of freedom = 速度向量的真实维度 = 15，而 nq（qpos 的长度）= 16。
        #   这就是为什么叫 "degrees of freedom" 而不是 "speed"——它强调的是"这个系统真正能独立变化的方向有几个"。
        ctrl=self._init_ctrl,  #"当前施加在关节上的控制信号"。用 home keyframe 里预设的值，让手臂保持在初始姿态不乱动。
        impl=self._mjx_model.impl.value,  # 用哪个计算后端。这里是 'warp'（NVIDIA GPU 加速）。
        #以下三行是碰撞检测的内部缓冲区大小上限。
        naconmax=self._config.naconmax,
        naccdmax=self._config.naccdmax,
        njmax=self._config.njmax,
    )

    # set target mocap position
    # mocap 是 MuJoCo 里一种特殊物体——不受物理影响，你直接设定它的位置。target 就是一个 mocap
    # body，它不会被撞飞、不会掉下来，只是静静地浮在空中标记"把 box 搬到这里"。
    # data.replace(...) 是 JAX 常用的 immutable 更新方式——不修改原始 data，返回一个新 data。
    data = data.replace(
        mocap_pos=data.mocap_pos.at[self._mocap_target, :].set(target_pos),
        # at[self._mocap_target, :]
        # 是二维数组的索引。mocap_pos 是一个二维数组，每一行是一个 mocap body 的 xyz 位置：
        # # mocap_pos 的形状比如是 (2, 3)，表示有 2 个 mocap body，每个有 xyz 3 个坐标
        #   mocap_pos = [[0.1, 0.2, 0.3],   # 第 0 个 mocap body
        #                [0.4, 0.5, 0.6]]   # 第 1 个 mocap body
        #   逗号分隔的是不同的维度：
        #   mocap_pos[1, :]     # 第 1 行，所有列 → [0.4, 0.5, 0.6]
        #   #         ↑  ↑
        #   #        行  列（: 表示"全部"）
        #   所以 .at[self._mocap_target, :].set(target_pos) 的意思是：选中第 self._mocap_target 行的全部 3 列，替换成 target_pos。
        #   和一维的 .at[9:12] 是同一个机制，只不过一维用切片选连续元素，二维用 [行, 列] 选一整行。
        mocap_quat=data.mocap_quat.at[self._mocap_target, :].set(target_quat),
        # 所以这里就是更新target坐标和四元数为本次初始化的随机值。
    )

    # initialize env state and info
    metrics = {
        "out_of_bounds": jp.array(0.0, dtype=float),
        **{k: 0.0 for k in self._config.reward_config.scales.keys()},  #就是开头定义的config里的reward_config字段，不要想多了
        # 先看 **{} 这个语法。** 是把一个 dict 展开合并进外层 dict。比如：
        #   d1 = {"a": 1}
        #   d2 = {**d1, "b": 2}
        #   # d2 = {"a": 1, "b": 2}

        # 所以展开后 metrics 实际上是：
        # metrics = {
        #     "out_of_bounds": 0.0,
        #     "gripper_box": 0.0,
        #     "box_target": 0.0,
        #     "no_floor_collision": 0.0,
        #     "robot_target_qpos": 0.0,
        # }
    }
    info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
    # info — 需要在 step 之间传递的额外信息：

    # info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}

    # - rng — 最新的随机 key，留着以后 step 里如果需要随机性可以用
    # - target_pos — target 的位置。存在 info 里是因为 _get_reward 需要它，但它不在 data 里（mocap_pos 在 data  里，但直接存一份方便取用）
    # - reached_box: 0.0 — 那个 latch："gripper 有没有碰到过 box"。初始是 0（没碰过），碰过之后变成 1，永远不会回到 0
    obs = self._get_obs(data, info) # 获得这个模拟环境的上帝视角（完整观测数据）
    reward, done = jp.zeros(2) # just start, no reward, not done either
    state = State(data, obs, reward, done, metrics, info) # pack all the data into a State object
    return state

  def step(self, state: State, action: jax.Array) -> State: #输入一个state, 和agent决定的8维动作 | 输出更新后的state
    delta = action * self._action_scale   # self._action_scale  见开头的default config，默认是 0.04 , 是一个 关节角度变化量
    # 关于action:
    # agent 内部有一个神经网络，叫做 actor 或者 policy network。它的输入输出非常直接：
    # 输入: obs (67维数字向量) ──→ [神经网络] ──→ 输出: action (9维向量)
    # 网络内部大致长这样：
    # obs (67维)
    #    ↓
    # 全连接层 1 (67→256) + ReLU激活
    #    ↓
    # 全连接层 2 (256→256) + ReLU激活
    #    ↓
    # 全连接层 3 (256→9)   + tanh激活   ← 输出层   #注意tanh和三角函数里的tan没有关系！ tanh(x) => [-1,1] 它是压缩函数！
    #    ↓
    # action (9维, 每个值在 [-1, 1] 之间)

    ctrl = state.data.ctrl + delta # 在当前控制信号基础上加上变化量
    # 其中 state.data.ctrl 就是来自 self._post_init(obj_name="box", keyframe="home") 初始化的值（如果当前state是reset刚返回的，即刚生成）
    # 这里控制模式是速度控制（velocity control），不是位置控制。
    # 先理解 MuJoCo 中 ctrl 的含义：ctrl 是发给执行器（actuator）的目标值。
    # 在这个 XML 场景中，每个关节的执行器类型是 velocity actuator——你给它的值直接被当作关节角速度来执行。
    # 上一帧的 ctrl = [0.1, 0.2, -0.05, ...]   ← 上一帧各关节正在以什么速度转
    # delta         = [0.02, -0.012, 0.032, ...] ← 神经网络想让速度加快还是减慢
    # 新的 ctrl     = [0.12, 0.188, -0.018, ...] ← 叠加后的新目标速度
    # 打个比方：你在开车。

    # ctrl = 油门踏板踩多深（当前车速）
    # delta = 你再多踩一点 or 松一点
    # 新 ctrl = 新的油门位置
    # 你不是直接说"开到 60 公里"（那叫位置控制），而是说"在现在的踏板深度基础上再踩深一点"（速度控制）。
    ctrl = jp.clip(ctrl, self._lowers, self._uppers) # 裁剪到合法范围内
    #_lowers 和 _uppers 是从 MuJoCo 模型里读出的每个执行器的物理限制（见panda.py:111）
    # clip 保证不管你神经网络发什么疯，执行器不会收到超出物理极限的指令。

    data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
    #call MuJoCo engine to move the world forward
    # recall self._mjx_model is the MJX version MuJoCo object
    # n_substeps 意思是：每次控制信号更新时，物理引擎内部更新多少步
    # n_substeps 在哪算的？
    # 在 MjxEnv 基类里定义的 property（mjx_env.py:271-274）：

    # @property
    # def n_substeps(self) -> int:
    #     """Number of sim steps per control step."""
    #     return int(round(self.dt / self.sim_dt))

    #这里 self.dt = ctrl_dt = 0.02; self.sim_dt = sim_dt = 0.005 (见开头config)
    #所以 n_substeps = 0.02/0.005 = 4

    # mjx_env.step 的详细解释如下：
    # def step(
    #     model: mjx.Model,
    #     data: mjx.Data,
    #     action: jax.Array,
    #     n_substeps: int = 1,
    # ) -> mjx.Data:
    #   def single_step(data, _):
    #     data = data.replace(ctrl=action)
    #     data = mjx.step(model, data) #这里不是递归啊！ 别看走眼了！ 这个step本身是mjx_env.step ，它要call的是mjx.step
    #     return data, None
    #   return jax.lax.scan(single_step, data, (), n_substeps)[0]

    # 1.
    # 函数里面定义另一个函数，在 Python 里是合法的。这叫闭包（closure）。
    # 内部函数可以用外部函数的变量。这里 single_step 用了外部的 model 和 action
    # 2.
    # 为什么要这样写？因为 jax.lax.scan 要求传入一个函数，而且这个函数的签名必须是 (carry, x) -> (carry, output)。
    # 其中 结果（carry）， 输入（x）， 输出 (output)
    # 但我们还需要 model 和 action，没办法通过 scan 的参数传进去。
    # 3.
    # 为什么single_step要return data, None?
    # 单纯是为了满足jax.lax.scan的要求，见上文所述
    # 4.
    # 为什么 return jax.lax.scan(single_step, data, (), n_substeps)[0] 这里的第三个参数是 () ?
    # 这是 每次循环的额外输入（这里不需要，传空元组）
    # 5.
    # jax.lax.scan 的工作方式：
    # scan 就是一个循环，每次迭代接收上一次的结果（carry）和一个输入（x），产出新的 carry 和一个输出（output）。
    # scan(函数, 初始carry, 输入序列, 循环次数)
    # 第1次: 函数(carry=data,  x=()) → 返回 (新data, None)
    # 第2次: 函数(carry=新data, x=()) → 返回 (更新data, None)
    # 第3次: 函数(carry=更新data, x=()) → 返回 (再更新data, None)
    # 第4次: 函数(carry=再更新data, x=()) → 返回 (最终data, None)
    # - carry = 在迭代之间传递的状态，这里就是 data。每次物理模拟后 data 更新，传给下一次
    #   - x = 每次迭代的输入。这里每次都不需要额外输入，所以传 ()（空 tuple），纯占位
    #   - output = 每次迭代的额外输出，收集成序列。这里不需要，所以返回 None
    #   最后 [0]：
    # scan 返回 (最终carry, 所有output的序列)，即 (最终data, [None, None, None, None])。我们只要最终的 data，所以取 [0]。


    raw_rewards = self._get_reward(data, state.info)
    # self._get_reward will return something (in range [0,1] like:
    # raw_rewards = {
    #     "gripper_box":       0.92,   # 夹爪离 box 有多近（0~1，1=碰到了）
    #     "box_target":        0.15,   # box 离目标有多近（0~1，1=到了）
    #     "no_floor_collision": 1.0,   # 有没有碰地（1=没碰）
    #     "robot_target_qpos": 0.88,   # 手臂是否还在初始姿态附近（0~1）
    # }
    rewards = {
        k: v * self._config.reward_config.scales[k]
        for k, v in raw_rewards.items()
    }
    # weighted sum of reward, see reward_config of the default config at top of this file
    reward = jp.clip(sum(rewards.values()), -1e4, 1e4)
    # safety measure: clip reward to [-1e4, 1e4], but not a critical one
    box_pos = data.xpos[self._obj_body]
    # get box position (x,y,z) after update
    out_of_bounds = jp.any(jp.abs(box_pos) > 1.0)
    # check if abs(x) > 1.0 or abs(y) > 1.0 or abs(z) > 1.0
    # jp.any return true if any of the elements are true
    out_of_bounds |= box_pos[2] < 0.0
    # |= is not bit operation but just bool_result = bool_result or other_bool_result
    # e.g. out_of_bounds = True or False = True
    done = out_of_bounds | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()
    # The episode ends if:
    # out_of_bounds
    # jp.isnan(data.qpos).any() — 关节位置出现 NaN（物理仿真炸了）
    # jp.isnan(data.qvel).any() — 速度出现 NaN
    done = done.astype(float)
    # .astype() 简单理解成类型强转就好
    state.metrics.update(
        #state.metrics的定义参见reset()方法里的metrics数组，它是一般字典，可以用python原生方法直接in place update
        **raw_rewards, out_of_bounds=out_of_bounds.astype(float)
    )

    obs = self._get_obs(data, state.info)
    state = State(data, obs, reward, done, state.metrics, state.info)

    return state

  def _get_reward(self, data: mjx.Data, info: Dict[str, Any]) -> Dict[str, Any]:
    target_pos = info["target_pos"]
    #recall reset()里的 info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
    # info — 需要在 step 之间传递的额外信息：
    # info = {"rng": rng, "target_pos": target_pos, "reached_box": 0.0}
    box_pos = data.xpos[self._obj_body]
    gripper_pos = data.site_xpos[self._gripper_site]
    # xpos = 所有 body 的全局坐标位置（相对于世界原点），site_xpos = site 的全局位置。
    # site 是附着在某个 body 上的一个"标记点"——这里 gripper site 定义在两根手指中间。
    pos_err = jp.linalg.norm(target_pos - box_pos)
    # pos_err 就是两点之间的距离 , norm 就是空间中的直线距离
    box_mat = data.xmat[self._obj_body] # box 的 3×3 旋转矩阵 ； mat = matrix ；但是这里的_mat都特指旋转矩阵 (rotation matrix）
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target]) # 目标的 3×3 旋转矩阵
    rot_err = jp.linalg.norm(target_mat.ravel()[:6] - box_mat.ravel()[:6])
    # box 朝向越接近 target 要求的朝向，rot_err 越小。

    # two core reward
    # if error = 0 => tanh(0) = 0 => reward = 1-0 =1 (max/perfect)
    # if error = large => tanh(large) ~= 1 => reward = 1 - (~1) ~= 0 (min/bad)
    box_target = 1 - jp.tanh(5 * (0.9 * pos_err + 0.1 * rot_err))
    gripper_box = 1 - jp.tanh(5 * jp.linalg.norm(box_pos - gripper_pos))
    # 手臂越接近 home 姿态，reward 越高。防止 agent 用奇怪的姿势完成任务。
    # 1. 现实中机械臂会坏。 极端姿势下关节承受的力矩很大，长期这样用会损坏硬件。
    # 2. 奇怪姿势容易卡死。 比如手臂绕了一圈从背面去够 box，下一个 episode box 换了位置就够不到了，泛化性很差。
    # 3. 正常姿势活动范围最大。 home 姿态是机械臂设计者选的——手臂在中间位置，朝各个方向都能灵活移动。偏离太远，某些方向就动不了了。
    robot_target_qpos = 1 - jp.tanh(
        jp.linalg.norm(
            data.qpos[self._robot_arm_qposadr]
            - self._init_q[self._robot_arm_qposadr]
        )
    )

    # Check for collisions with the floor
    hand_floor_collision = [
        data.sensordata[self._mj_model.sensor_adr[sensor_id]] > 0
        for sensor_id in self._floor_hand_found_sensor
    ]
    floor_collision = sum(hand_floor_collision) > 0
    no_floor_collision = (1 - floor_collision).astype(float)

    info["reached_box"] = 1.0 * jp.maximum(
        info["reached_box"],
        (jp.linalg.norm(box_pos - gripper_pos) < 0.012),   # result is T/F but jp.maximum treat T/F as 1/0 | 0.012 = 1.2cm
    )

    rewards = {
        "gripper_box": gripper_box,
        "box_target": box_target * info["reached_box"],
        "no_floor_collision": no_floor_collision,
        "robot_target_qpos": robot_target_qpos,
    }
    return rewards

  def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
    gripper_pos = data.site_xpos[self._gripper_site]
    gripper_mat = data.site_xmat[self._gripper_site].ravel()
    #recall :
    # mat = matrix ；但是这里的_mat都特指旋转矩阵 (rotation matrix）
    # xpos = 所有 body 的全局坐标位置（相对于世界原点），site_xpos = site 的全局位置。
    # site 是附着在某个 body 上的一个"标记点"——这里 gripper site 定义在两根手指中间。
    target_mat = math.quat_to_mat(data.mocap_quat[self._mocap_target])
    obs = jp.concatenate([
        data.qpos, # 16 dims: 所有关节角度 + box位姿
        data.qvel, # 15 dims: 所有速度
        gripper_pos, # 3 dims: gripper 在哪
        gripper_mat[3:], # 6 dims: gripper 的朝向 but ignore take first row
        data.xmat[self._obj_body].ravel()[3:], # 6 dims: box 的朝向 but only ignore first row
        data.xpos[self._obj_body] - data.site_xpos[self._gripper_site],  # 3 dims: box 相对 gripper 的偏移
        info["target_pos"] - data.xpos[self._obj_body], # 3 dims: target 相对 box 的偏移
        target_mat.ravel()[:6] - data.xmat[self._obj_body].ravel()[:6], # 6 dims: 旋转误差 but only consider first row
        data.ctrl - data.qpos[self._robot_qposadr[:-1]], # 8 dims: 控制信号和实际关节角度的差
        # 最后一项 ctrl - qpos： 控制信号和实际关节角度的差。如果差为 0，说明关节已经到达目标位置；差大说明关节还在移动中。让 agent
        #  知道"我的上一个指令执行到什么程度了"。
        # recall: in step():
        # data = mjx_env.step(self._mjx_model, state.data, ctrl, self.n_substeps)
        # 并不保证经过 self.n_substeps 之后，机械臂会完成给定的动作

        # [:-1] 是什么？ self._robot_qposadr 有 9 个元素（7 手臂 + 2 手指），但 ctrl 只有 8
        # 个（两个手指共用一个控制信号），所以去掉最后一个。
    ])

    return obs


class PandaPickCubeOrientation(PandaPickCube):
  """Bring a box to a target and orientation."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(config, config_overrides, sample_orientation=True)
