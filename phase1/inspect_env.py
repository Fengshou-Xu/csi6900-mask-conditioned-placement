"""
Run this FIRST before editing place.py.
It prints the key values you need to fill in the placeholders.

Usage:  python inspect_env.py
"""
import jax
import jax.numpy as jp
from mujoco_playground._src.manipulation.franka_emika_panda.pick import PandaPickCube

env = PandaPickCube()
state = env.reset(jax.random.PRNGKey(0))

print("=" * 60)
print("VALUES YOU NEED FOR place.py")
print("=" * 60)

# 1. Gripper home position — this is where you'll place the box
print(f"\n[1] Gripper site position (home pose):")
print(f"    {state.data.site_xpos[env._gripper_site]}")

# 2. Finger joint indices and current (open) values
finger_indices = env._robot_qposadr[-2:]
print(f"\n[2] Finger joint qpos indices: {finger_indices}")
print(f"    Current (open) values: {env._init_q[finger_indices[0]]}, {env._init_q[finger_indices[1]]}")

# 3. Actuator limits for fingers (last 1 actuator controls both fingers)
print(f"\n[3] Actuator limits (all joints):")
print(f"    Lowers: {env._lowers}")
print(f"    Uppers: {env._uppers}")

# 4. Object qpos address (where box xyz starts in qpos array)
print(f"\n[4] Object qpos address: {env._obj_qposadr}")

# 5. Default box position on table
print(f"\n[5] Default box position: {env._init_obj_pos}")
print(f"    (Z value = table surface height)")

# 6. Full init_q for reference
print(f"\n[6] Full init_q ({len(env._init_q)} values):")
print(f"    {env._init_q}")

# 7. Init ctrl
print(f"\n[7] Init ctrl ({len(env._init_ctrl)} values):")
print(f"    {env._init_ctrl}")

# 8. All robot qpos addresses (arm + fingers)
print(f"\n[8] Robot qpos addresses: {env._robot_qposadr}")
print(f"    Arm only:              {env._robot_arm_qposadr}")

print("\n" + "=" * 60)
print("NEXT STEP: Copy the gripper position from [1] into place.py")
print("           where it says PLACEHOLDER")
print("=" * 60)
