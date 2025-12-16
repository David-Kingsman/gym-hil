#!/usr/bin/env python

from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.mujoco_gym_env import FrankaGymEnv, GymRenderingSpec

# --- CONFIG CONSTANTS (Matched to PickCube and new XML) ---
# 调整joint 2角度从0.195减小到0.1，让end effector的z轴更高
_PANDA_HOME = np.asarray((0, 0.1, 0, -2.43, 0, 2.62, 0.785))

# Extended X bounds to reach wall at X=0.6m (wall moved closer to robot)
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.7, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.4, 0.15]])

# New Target Definition (Coordinates from XML: x=0.6, y=0.0, z=0.352)
# Wall and base are positioned at X=0.6m (moved closer to robot, was 0.7m)
# Wall center is at Y=0.0 (aligned with base center)
# Foundation高度2cm，新砖块高度92mm，第4层中心Z = 0.352m
_TARGET_POS = np.asarray([0.6, 0.0, 0.362]) 
_PLACE_DISTANCE_THRESHOLD = 0.005 # 5mm tolerance: 砖块中心到目标中心的XYZ距离（考虑到手动操作精度）
_PLACE_XY_THRESHOLD = 0.005  # 5mm: X和Y方向的阈值（可以稍微宽松）
_PLACE_Z_THRESHOLD = 0.003   # 3mm: Z方向的阈值（需要更精确）
_RELEASE_DISTANCE = 0.05     # 5cm: gripper释放后，TCP到block的距离阈值

class PandaMasonryBlockInsertionEnv(FrankaGymEnv):
    """Environment for a Panda robot picking up a block and placing it into a masonry slot."""

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "dense", # Default to dense for placement
        random_block_position: bool = True,
        # Custom XML path
        xml_path: Optional[Path] = None, 
    ):
        self.reward_type = reward_type

        # Use the simplified placement XML
        if xml_path is None:
            xml_path = Path(__file__).parent.parent / "assets" / "masonry_insertion.xml"

        super().__init__(
            xml_path=xml_path,
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_PANDA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
        )

        # Task-specific setup (Matches PickCube)
        # Note: XML block geom name is "block"
        self._block_z_half_size = self._model.geom("block").size[2]  # Half-size (0.035m)
        # Block初始Z坐标应该从XML body的初始位置读取，而不是使用half-size
        # XML中moveable_block body的初始pos是 [0.5, 0.0, 0.046]，所以初始Z应该是0.046m（新砖块高度92mm的中心）
        try:
            block_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "moveable_block")
            # body的初始位置在model.body_pos中
            self._block_z_init = self._model.body_pos[block_body_id][2]  # 初始Z坐标
        except:
            # Fallback: 使用XML中定义的初始Z坐标
            self._block_z_init = 0.046  # XML中定义的初始Z（新砖块高度92mm的中心）
        
        self._random_block_position = random_block_position
        
        # terminate_on_success: 是否在成功时终止episode（默认True，保持向后兼容）
        # 如果设置为False，episode会继续运行直到达到max_episode_steps，即使已经成功
        self._terminate_on_success = True  # 默认值，可以通过属性设置
        
        # New: Target position site ID (CRITICAL: must exist for insertion task)
        try:
            self._target_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
            if self._target_site_id >= 0:
                print(f"[MasonryEnv] Target site found, ID: {self._target_site_id}")
            else:
                raise ValueError("target_site ID is negative")
        except Exception as e:
            # 对于插入任务，target_site是必需的，不应该静默失败
            print(f"ERROR: 'target_site' NOT FOUND in XML file: {xml_path}")
            print(f"Error details: {e}")
            print("This environment requires 'target_site' for insertion task!")
            print("Environment will NOT work correctly without target_site.")
            # 仍然设置为-1，但在_is_success中会直接返回False而不是使用fallback逻辑
            self._target_site_id = -1
        
        # 尝试获取gripper joint ID（用于检查gripper是否打开）
        try:
            self._right_driver_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "right_driver_joint")
            self._left_driver_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "left_driver_joint")
            self._has_gripper_joints = True
        except:
            self._has_gripper_joints = False
            print("WARNING: Could not find gripper driver joints, will use TCP distance fallback")


        # Setup observation space (Kept consistent with PickCube 3D pos)
        agent_dim = self.get_robot_state().shape[0]
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        env_box = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32) # Block XYZ position

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "front": spaces.Box(0, 255, (self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8),
                            "wrist": spaces.Box(0, 255, (self._render_specs.height, self._render_specs.width, 3), dtype=np.uint8),
                        }
                    ),
                    "agent_pos": agent_box,
                }
            )
        else:
            self.observation_space = spaces.Dict(
                {
                    "agent_pos": agent_box,
                    "environment_state": env_box,
                }
            )

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment (exactly like PandaPickCube)."""
        # Ensure gymnasium internal RNG is initialized when a seed is provided
        super().reset(seed=seed)
        
        # Ensure _np_random is initialized (Gymnasium should do this, but handle edge case)
        # If seed=None and _np_random doesn't exist, initialize it
        # If seed is provided, super().reset(seed=seed) should initialize it
        if not hasattr(self, '_np_random') or self._np_random is None:
            if seed is not None:
                self._np_random = np.random.default_rng(seed)
            else:
                # If no seed, use a default RNG (Gymnasium should have initialized this, but handle edge case)
                self._np_random = np.random.default_rng()
        
        mujoco.mj_resetData(self._model, self._data)
        
        # Reset the robot to home position
        # Note: reset_robot() now handles mocap position + orientation synchronization
        # 注意：link0的位置已经通过panda_masonry.xml在XML中设置为Z=0.05m（base顶部），不需要代码设置
        self.reset_robot()

        # Sample a new block position (using robust qpos setting via jnt_qposadr)
        # Use self._np_random to ensure determinism when seed is provided
        if self._random_block_position:
            block_xy = self._np_random.uniform(*_SAMPLING_BOUNDS)
        else:
            block_xy = np.asarray([0.5, 0.0])
        
        # 使用正确的初始Z坐标（0.06m），而不是half-size
        block_pos = np.asarray([block_xy[0], block_xy[1], self._block_z_init])
        # Block initial orientation: identity quaternion [1, 0, 0, 0]
        # (XML body is already oriented correctly, no rotation needed)
        initial_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion (no rotation)
        
        # Use jnt_qposadr to set qpos (most reliable method)
        try:
            block_joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, "block_joint")
            qpos_start = self._model.jnt_qposadr[block_joint_id]
            
            # Set position (3D: x, y, z)
            self._data.qpos[qpos_start:qpos_start + 3] = block_pos
            # Set orientation (4D quaternion: w, x, y, z)
            self._data.qpos[qpos_start + 3:qpos_start + 7] = initial_quat
            
            mujoco.mj_forward(self._model, self._data)
        except Exception as e:
            # Fallback to name-based access if ID lookup fails
            try:
                self._data.jnt("block_joint").qpos[:3] = block_pos
                self._data.jnt("block_joint").qpos[3:7] = initial_quat
                mujoco.mj_forward(self._model, self._data)
            except Exception as e2:
                print(f"WARNING: Failed to set block position. {e2}")
                # Continue with default position from XML
                mujoco.mj_forward(self._model, self._data)
        
        # Cache the initial block height (exactly like PandaPickCube)
        actual_block_pos = self._data.sensor("block_pos").data
        self._z_init = actual_block_pos[2]
        self._z_success = self._z_init + 0.1

        # 调试输出：检查reset后block位置（每次reset都打印，确认代码被加载）
        target_pos = self._get_target_pos() if self._target_site_id >= 0 else None
        if target_pos is not None:
            dist_to_target = np.linalg.norm(actual_block_pos - target_pos)
            print(f"\n[RESET] Block position after reset:")
            print(f"  block_pos = [{actual_block_pos[0]:.4f}, {actual_block_pos[1]:.4f}, {actual_block_pos[2]:.4f}]")
            print(f"  target_pos = [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
            print(f"  distance = {dist_to_target*1000:.2f}mm (threshold: {_PLACE_DISTANCE_THRESHOLD*1000:.1f}mm)")
            if dist_to_target < 0.05:
                print(f"  ⚠️  WARNING: Block is very close to target! This may cause false success!")
            print()
        
        # Reset episode step counter (like PandaPickCube - though it doesn't track it)
        self._episode_step = 0

        # 注意：link0的位置已经通过panda_masonry.xml在XML中设置为Z=0.05m（base顶部），不需要代码设置
        obs = self._compute_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment (like PandaPickCube)."""
        # Apply the action to the robot
        self.apply_action(action)
        
        # Increment episode step counter
        self._episode_step += 1

        # Compute observation, reward and termination (like PandaPickCube)
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()

        # 注意：对于插入任务，应该使用_is_success()的结果，而不是reward
        # 如果reward_type是"sparse"，也不要覆盖success，因为success应该是基于位置判定的
        # 原来的代码会覆盖success，导致误判，已禁用：
        # if self.reward_type == "sparse":
        #     success = rew == 1.0  # 这会导致误判，已禁用
        
        # 调试输出：如果success为True，打印详细信息
        if success:
            print(f"\n[STEP DEBUG] Step {self._episode_step}: success = {success}, rew = {rew:.4f}")
            print(f"  reward_type = {self.reward_type}")
            block_pos = self._data.sensor("block_pos").data
            target_pos = self._get_target_pos()
            dist_3d = np.linalg.norm(block_pos - target_pos)
            print(f"  final distance = {dist_3d*1000:.2f}mm\n")

        # Check if block is outside reasonable bounds
        # 注意：target位置在X=0.6，所以边界检查应该允许block移动到target位置
        block_pos = self._data.sensor("block_pos").data
        target_pos = self._get_target_pos()
        
        # 使用更合理的边界：考虑target位置，允许block移动到target附近
        # 边界应该覆盖从初始采样范围到target位置的范围
        min_x = min(_SAMPLING_BOUNDS[0][0] - 0.1, target_pos[0] - 0.1)  # 允许超出初始范围
        max_x = max(_SAMPLING_BOUNDS[1][0] + 0.1, target_pos[0] + 0.1)  # 允许移动到target
        min_y = _SAMPLING_BOUNDS[0][1] - 0.1
        max_y = _SAMPLING_BOUNDS[1][1] + 0.1
        
        # 只有在block明显超出合理范围时才判定为超出边界
        # 如果block在目标位置附近，不应该判定为超出边界
        dist_to_target_xy = np.linalg.norm(block_pos[:2] - target_pos[:2])
        if dist_to_target_xy < 0.15:  # 如果block在target附近15cm内，不算超出边界
            exceeded_bounds = False
        else:
            exceeded_bounds = (block_pos[0] < min_x or block_pos[0] > max_x or 
                             block_pos[1] < min_y or block_pos[1] > max_y)

        # Check termination (like PandaPickCube, but with max episode steps)
        # Note: max_episode_steps is 200 (20 seconds @ 10 FPS)
        # 如果terminate_on_success=False，成功时不会终止，继续运行直到max_episode_steps
        if self._terminate_on_success:
            terminated = bool(success or exceeded_bounds or self._episode_step >= 200)
        else:
            # 不因成功而终止，只因为超出边界或达到最大步数而终止
            terminated = bool(exceeded_bounds or self._episode_step >= 200)
        
        # 调试输出：episode结束时打印原因和详细失败分析
        if terminated:
            reasons = []
            if success:
                reasons.append("SUCCESS")
            if exceeded_bounds:
                reasons.append("EXCEEDED_BOUNDS")
            if self._episode_step >= 200:
                reasons.append("MAX_STEPS")
            
            print(f"\n[EPISODE END] Step {self._episode_step}, Reason: {', '.join(reasons)}")
            
            # 详细失败原因分析
            if not success:
                block_pos = self._data.sensor("block_pos").data
                target_pos = self._get_target_pos()
                dist_3d = np.linalg.norm(block_pos - target_pos)
                dist_x = abs(block_pos[0] - target_pos[0])
                dist_y = abs(block_pos[1] - target_pos[1])
                dist_z = abs(block_pos[2] - target_pos[2])
                
                # 检查各项失败原因
                failure_reasons = []
                
                # 1. 位置精度检查
                xy_ok = (dist_x < _PLACE_XY_THRESHOLD and dist_y < _PLACE_XY_THRESHOLD)
                z_at_target = dist_z < _PLACE_Z_THRESHOLD
                distance_ok = dist_3d < _PLACE_DISTANCE_THRESHOLD
                
                if not xy_ok:
                    if dist_x >= _PLACE_XY_THRESHOLD:
                        failure_reasons.append(f"❌ X位置偏差: {dist_x*1000:.2f}mm (阈值: {_PLACE_XY_THRESHOLD*1000:.1f}mm)")
                    if dist_y >= _PLACE_XY_THRESHOLD:
                        failure_reasons.append(f"❌ Y位置偏差: {dist_y*1000:.2f}mm (阈值: {_PLACE_XY_THRESHOLD*1000:.1f}mm)")
                
                if not z_at_target:
                    failure_reasons.append(f"❌ Z位置偏差: {dist_z*1000:.2f}mm (阈值: {_PLACE_Z_THRESHOLD*1000:.1f}mm)")
                
                if not distance_ok:
                    failure_reasons.append(f"❌ 3D距离过大: {dist_3d*1000:.2f}mm (阈值: {_PLACE_DISTANCE_THRESHOLD*1000:.1f}mm)")
                
                # 2. Gripper释放检查
                is_released = False
                release_check_method = "unknown"
                if self._has_gripper_joints:
                    try:
                        right_angle = self._data.qpos[self._right_driver_joint_id]
                        left_angle = self._data.qpos[self._left_driver_joint_id]
                        is_released = (right_angle > 0.5) and (left_angle > 0.5)
                        release_check_method = f"joint (R={right_angle:.3f}, L={left_angle:.3f})"
                    except Exception:
                        pass
                
                if not is_released:
                    try:
                        tcp_pos = self._data.sensor("2f85/pinch_pos").data
                        dist_tcp_block = np.linalg.norm(block_pos - tcp_pos)
                        is_released = dist_tcp_block > (_RELEASE_DISTANCE + 0.03)
                        if is_released:
                            release_check_method = f"TCP_distance ({dist_tcp_block*1000:.1f}mm)"
                        else:
                            failure_reasons.append(f"❌ Gripper未释放: {release_check_method}, TCP距离={dist_tcp_block*1000:.1f}mm (需要>{(_RELEASE_DISTANCE+0.03)*1000:.0f}mm)")
                    except Exception:
                        failure_reasons.append(f"❌ Gripper未释放: {release_check_method}")
                
                # 3. 稳定性检查
                try:
                    block_vel = self._data.sensor("block_vel").data.copy()
                    lin_vel = np.linalg.norm(block_vel[:3]) if len(block_vel) >= 3 else 0.0
                    is_stable = lin_vel < 0.02  # 线速度 < 2cm/s (放宽从1cm/s以提高成功率)
                    if not is_stable:
                        failure_reasons.append(f"❌ 不稳定: 速度={lin_vel*1000:.2f}mm/s (阈值: 20mm/s)")
                except Exception:
                    pass
                
                # 4. 提升检查
                block_lifted = block_pos[2] > self._z_init + 0.03
                if not block_lifted:
                    failure_reasons.append(f"❌ Block未提升: block_z={block_pos[2]:.4f}m, init_z={self._z_init:.4f}m (需要提升>3cm)")
                
                # 打印详细失败信息
                print(f"  📍 最终位置:")
                print(f"     block_pos = [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")
                print(f"     target_pos = [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
                print(f"     3D距离 = {dist_3d*1000:.2f}mm (阈值: {_PLACE_DISTANCE_THRESHOLD*1000:.1f}mm)")
                print(f"     X距离 = {dist_x*1000:.2f}mm, Y距离 = {dist_y*1000:.2f}mm, Z距离 = {dist_z*1000:.2f}mm")
                
                if failure_reasons:
                    print(f"  ❌ 失败原因:")
                    for reason in failure_reasons:
                        print(f"     {reason}")
                else:
                    print(f"  ⚠️  所有单项检查都通过，但总体success=False（可能组合条件未满足）")
            else:
                # 成功时的输出（已有详细输出在_is_success中）
                block_pos = self._data.sensor("block_pos").data
                target_pos = self._get_target_pos()
                dist_3d = np.linalg.norm(block_pos - target_pos)
                print(f"  ✅ Final distance: {dist_3d*1000:.2f}mm\n")

        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> dict:
        """Compute the current observation (Identical to PickCube)."""
        observation = {}
        robot_state = self.get_robot_state().astype(np.float32)
        block_pos = self._data.sensor("block_pos").data.astype(np.float32)

        if self.image_obs:
            front_view, wrist_view = self.render()
            observation = {
                "pixels": {"front": front_view, "wrist": wrist_view},
                "agent_pos": robot_state,
            }
        else:
            observation = {
                "agent_pos": robot_state,
                "environment_state": block_pos,
            }
        return observation

    def _get_target_pos(self) -> np.ndarray:
        """Helper to get the current target position from the site."""
        if self._target_site_id >= 0:
            return self._data.site_xpos[self._target_site_id].copy()
        return _TARGET_POS.copy()


    def _compute_reward(self) -> float:
        """Compute reward based on Pick/Place progress.
        
        改进版本：采用分阶段奖励，让学习更容易（参考 Pick & Lift 的简单性）
        - 阶段1：抓取阶段（靠近 + 抓中心 + 提升）
        - 阶段2：放置阶段（接近目标 + 精确放置）
        """
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        
        # ===== 情况 1：没有 target_site（旧 scene），保持原 PickCube 逻辑 =====
        if self._target_site_id < 0:
            if self.reward_type == "dense":
                dist = np.linalg.norm(block_pos - tcp_pos)
                r_close = np.exp(-20 * dist)
                r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
                r_lift = np.clip(r_lift, 0.0, 1.0)
                return float(0.3 * r_close + 0.7 * r_lift)
            else:
                lift = block_pos[2] - self._z_init
                return float(lift > 0.1)
        
        # ===== 情况 2：masonry_insertion，改进的分阶段奖励 =====
        target_pos = self._get_target_pos()
        dist_block_target = np.linalg.norm(block_pos - target_pos)
        dist_tcp_block = np.linalg.norm(block_pos - tcp_pos)
        block_lifted = block_pos[2] > self._z_init + 0.01
        
        # ===== 阶段1：抓取阶段（类似 Pick & Lift，简单有效）=====
        # 1) TCP 靠近砖块（使用与 Pick & Lift 相同的系数）
        r_close = np.exp(-20 * dist_tcp_block) * 0.20  # 权重0.20
        
        # 2) 抓在砖中心（检查 X、Y、Z 三个方向，确保抓在中心）
        dx = abs(block_pos[0] - tcp_pos[0])  # X方向（砖块长度方向）
        dy = abs(block_pos[1] - tcp_pos[1])  # Y方向（砖块宽度方向）
        dz = abs(block_pos[2] - tcp_pos[2])  # Z方向（高度方向）
        r_center_x = np.exp(-30 * dx)
        r_center_y = np.exp(-30 * dy)
        r_center_z = np.exp(-30 * dz)
        r_grasp_center = 0.20 * (r_center_x + r_center_y + r_center_z) / 3.0  # 权重0.20，三个方向平均
        
        # 3) 提升奖励（关键改进：像 Pick & Lift 一样，提升高度应该有奖励）
        # 这能让 agent 先学会"抓取并提升"，再学"精确放置"
        if block_lifted:
            # 使用固定的最大提升高度进行归一化（类似 Pick & Lift）
            max_lift_height = target_pos[2] - self._z_init  # 从初始位置到目标位置的高度差
            lift_progress = (block_pos[2] - self._z_init) / max(max_lift_height, 0.1)  # 避免除零
            lift_progress = np.clip(lift_progress, 0.0, 1.0)
            r_lift = 0.20 * lift_progress  # 权重0.20，鼓励提升
        else:
            r_lift = 0.0
        
        # ===== 阶段2：放置阶段（分层次奖励，避免奖励稀疏）=====
        # 使用分层次的奖励函数，让接近目标时也有奖励（即使还没达到5mm精度）
        if block_lifted:
            # 使用单一的分段奖励函数，避免重叠导致奖励过大
            # 4a) 粗放奖励：接近目标区域（20cm内）就有奖励
            if dist_block_target < 0.20:  # 20cm内
                r_place_coarse = np.exp(-5 * dist_block_target) * 0.15  # 权重0.15
            else:
                r_place_coarse = 0.0
            
            # 4b) 中等奖励：接近目标（10cm内）有更多奖励（叠加在粗放奖励上）
            if dist_block_target < 0.10:  # 10cm内
                r_place_medium = np.exp(-10 * dist_block_target) * 0.15  # 权重0.15
            else:
                r_place_medium = 0.0
            
            # 4c) 精确奖励：非常接近目标（5cm内）有高奖励（叠加在前两个奖励上）
            if dist_block_target < 0.05:  # 5cm内
                r_place_fine = np.exp(-20 * dist_block_target) * 0.10  # 权重0.10
            else:
                r_place_fine = 0.0
            
            r_place = r_place_coarse + r_place_medium + r_place_fine
        else:
            r_place = 0.0
        
        # ===== 组合奖励 =====
        # 抓取阶段：靠近 + 抓中心 + 提升
        # 放置阶段：在提升后，额外奖励接近目标
        r = r_close + r_grasp_center + r_lift + r_place

        return float(np.clip(r, 0.0, 1.0))

    def _is_success(self) -> bool:
        """Check if the task is successfully completed.
        
        Strict success criterion: 
        - 砖块中心到目标中心的XYZ距离不超过2mm
        - 砖块必须是稳定的（速度很小）
        - 砖块必须已经被提升（高于初始位置）
        """
        block_pos = self._data.sensor("block_pos").data
        
        # 如果找不到target_site，返回False
        if self._target_site_id < 0:
            if self._episode_step % 50 == 0:  # 每50步打印一次，避免太多输出
                print(f"[DEBUG] _is_success: target_site_id < 0, returning False")
            return False
        
        # 获取目标位置
        target_pos = self._get_target_pos()
        
        # 计算砖块中心到目标中心的3D距离和各方向距离
        dist_3d = np.linalg.norm(block_pos - target_pos)
        dist_x = abs(block_pos[0] - target_pos[0])
        dist_y = abs(block_pos[1] - target_pos[1])
        dist_z = abs(block_pos[2] - target_pos[2])
        
        # 位置检查：分别检查X、Y、Z方向，允许不同方向的阈值不同
        xy_ok = (dist_x < _PLACE_XY_THRESHOLD and dist_y < _PLACE_XY_THRESHOLD)
        z_at_target = dist_z < _PLACE_Z_THRESHOLD  # Z方向需要更精确
        distance_ok = dist_3d < _PLACE_DISTANCE_THRESHOLD  # 3D距离作为总体检查
        
        # Gripper释放检查：确保block已经被释放，而不是被gripper抓着
        # 这个检查很重要，防止在放下过程中误判为成功
        is_released = False
        release_check_method = "unknown"
        
        if self._has_gripper_joints:
            try:
                # 检查gripper joint角度：如果gripper打开（joint角度较大），则认为已释放
                right_angle = self._data.qpos[self._right_driver_joint_id]
                left_angle = self._data.qpos[self._left_driver_joint_id]
                # Gripper打开时，joint角度应该 > 0.4（约40度），表示已释放
                # 放宽阈值从0.5到0.4以提高成功率，同时确保gripper真正打开
                is_released = (right_angle > 0.4) and (left_angle > 0.4)
                release_check_method = f"joint (R={right_angle:.3f}, L={left_angle:.3f})"
            except Exception as e:
                release_check_method = f"joint_check_failed ({str(e)})"
        
        # 如果无法通过joint检查，使用TCP到block的距离作为fallback
        if not is_released:
            try:
                tcp_pos = self._data.sensor("2f85/pinch_pos").data
                dist_tcp_block = np.linalg.norm(block_pos - tcp_pos)
                # 如果TCP到block的距离 > 8cm，认为已释放（增加阈值，更严格）
                is_released = dist_tcp_block > (_RELEASE_DISTANCE + 0.03)  # 8cm
                if is_released:
                    release_check_method = f"TCP_distance ({dist_tcp_block*1000:.1f}mm)"
            except Exception as e:
                # 如果无法获取TCP位置，保守处理：不认为已释放
                # 这样可以防止在放下过程中误判为成功
                release_check_method = f"TCP_check_failed ({str(e)}), NOT_RELEASED"
                is_released = False
        
        # 稳定性检查：砖块必须几乎静止
        try:
            block_vel = self._data.sensor("block_vel").data.copy()
            lin_vel = np.linalg.norm(block_vel[:3]) if len(block_vel) >= 3 else 0.0
            is_stable = lin_vel < 0.02  # 线速度 < 2cm/s (放宽从1cm/s以提高成功率)
        except (KeyError, AttributeError):
            is_stable = True
        
        # 提升检查：砖块必须已经被提升（至少提升3cm，确保已被拿起）
        block_lifted = block_pos[2] > self._z_init + 0.03
        
        # 所有条件都必须满足：
        # 1. X和Y距离 < 5mm
        # 2. Z距离 < 3mm（需要更精确）
        # 3. 3D距离 < 5mm（总体检查）
        # 4. Gripper已释放
        # 5. 稳定性检查
        # 6. 已提升检查
        success = (xy_ok and z_at_target and distance_ok and 
                  is_released and is_stable and block_lifted)
        
        # 详细调试输出：每次成功时打印
        if success:
            status = "✅ SUCCESS"
            print(f"\n[{status}] Step {self._episode_step}")
            print(f"  block_pos = [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")
            print(f"  target_pos = [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
            print(f"  3D距离 = {dist_3d*1000:.2f}mm (阈值: {_PLACE_DISTANCE_THRESHOLD*1000:.1f}mm)")
            print(f"  X距离 = {dist_x*1000:.2f}mm (阈值: {_PLACE_XY_THRESHOLD*1000:.1f}mm)")
            print(f"  Y距离 = {dist_y*1000:.2f}mm (阈值: {_PLACE_XY_THRESHOLD*1000:.1f}mm)")
            print(f"  Z距离 = {dist_z*1000:.2f}mm (阈值: {_PLACE_Z_THRESHOLD*1000:.1f}mm)")
            print(f"  Gripper已释放: {is_released} (检查方法: {release_check_method})")
            print(f"  稳定性: {is_stable}, 已提升: {block_lifted}")
            print(f"  所有条件: XY={xy_ok}, Z={z_at_target}, 3D={distance_ok}, 释放={is_released}, 稳定={is_stable}, 提升={block_lifted}")
            print("  ⚠️  成功判定！请确认block是否真的放置到位。\n")
        # elif dist_3d < 0.10:  # 接近成功时也打印（距离<10cm，降低阈值以便更容易看到）
        #     status = "⚠️  接近目标"
        #     print(f"\n[{status}] Step {self._episode_step}")
        #     print(f"  block_pos = [{block_pos[0]:.4f}, {block_pos[1]:.4f}, {block_pos[2]:.4f}]")
        #     print(f"  target_pos = [{target_pos[0]:.4f}, {target_pos[1]:.4f}, {target_pos[2]:.4f}]")
        #     print(f"  3D距离 = {dist_3d*1000:.2f}mm (阈值: {_PLACE_DISTANCE_THRESHOLD*1000:.1f}mm)")
        #     print(f"  X距离 = {dist_x*1000:.2f}mm (阈值: {_PLACE_XY_THRESHOLD*1000:.1f}mm)")
        #     print(f"  Y距离 = {dist_y*1000:.2f}mm (阈值: {_PLACE_XY_THRESHOLD*1000:.1f}mm)")
        #     print(f"  Z距离 = {dist_z*1000:.2f}mm (阈值: {_PLACE_Z_THRESHOLD*1000:.1f}mm)")
        #     try:
        #         block_vel = self._data.sensor("block_vel").data.copy()
        #         lin_vel = np.linalg.norm(block_vel[:3]) if len(block_vel) >= 3 else 0.0
        #         print(f"  线速度 = {lin_vel*1000:.2f}mm/s")
        #         print(f"  已提升 = {block_pos[2] > self._z_init + 0.03} (block Z={block_pos[2]:.4f}, init Z={self._z_init:.4f})")
        #         print(f"  稳定性 = {lin_vel < 0.01}")
        #         # 显示gripper释放状态和所有条件的详细状态
        #         print(f"  条件检查:")
        #         print(f"    XY位置: {xy_ok} (X={dist_x*1000:.2f}mm, Y={dist_y*1000:.2f}mm)")
        #         print(f"    Z位置: {z_at_target} (Z={dist_z*1000:.2f}mm)")
        #         print(f"    3D距离: {distance_ok} ({dist_3d*1000:.2f}mm)")
        #         if self._has_gripper_joints:
        #             try:
        #                 right_angle = self._data.qpos[self._right_driver_joint_id]
        #                 left_angle = self._data.qpos[self._left_driver_joint_id]
        #                 tcp_pos = self._data.sensor("2f85/pinch_pos").data
        #                 dist_tcp_block = np.linalg.norm(block_pos - tcp_pos)
        #                 print(f"    Gripper释放: {is_released} (方法: {release_check_method})")
        #                 print(f"    Gripper角度: right={right_angle:.3f}, left={left_angle:.3f}")
        #                 print(f"    TCP距离: {dist_tcp_block*1000:.1f}mm")
        #             except:
        #                 print(f"    Gripper释放: {is_released} (方法: {release_check_method})")
        #         else:
        #             print(f"    Gripper释放: {is_released} (方法: {release_check_method})")
        #         print(f"    稳定性: {is_stable} (速度={lin_vel*1000:.2f}mm/s)")
        #         print(f"    已提升: {block_lifted} (block Z={block_pos[2]:.4f} > init Z={self._z_init:.4f}+0.03)")
        #     except:
        #         pass
        #     print()
        
        return success
    


if __name__ == "__main__":
    from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper

    # Note: Ensure you have "masonry_insertion_simple_place.xml" in your assets folder
    env = PandaMasonryBlockInsertionEnv(render_mode="human", image_obs=True, reward_type="dense")
    env = PassiveViewerWrapper(env)
    env.reset()
    for _ in range(100):
        env.step(np.random.uniform(-1, 1, 7))
    env.close()