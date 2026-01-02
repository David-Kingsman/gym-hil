#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from pathlib import Path
from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.mujoco_gym_env import FrankaGymEnv, GymRenderingSpec
from scipy.spatial.transform import Rotation as R

# KUKA iiwa14 home position 复位起始位置  
_KUKA_HOME = np.asarray((0, 0.785398, 0, -1.5708, 0, 0.785398, 0))
# _CARTESIAN_BOUNDS = np.asarray([[0.0, -0.5, 0], [1.5, 0.5, 1.0]])  # bounding box for the robot, X: 0.0-1.0, Y: -0.5-0.5, Z: 0-0.8
_CARTESIAN_BOUNDS = np.asarray([[0.0, 0.0, 0], [0.9, 0.0, 0.41]])  # bounding box for the robot, X: 0.0-0.9, Y: 0.0-0.0, Z: 0-0.4
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.6, 0.15]])  # Window sampling area (same as plate)

# 窗口装配环境 
class KukaWindowAssemblyEnv(FrankaGymEnv):
    """Environment for a KUKA iiwa14 robot with vacuum gripper inserting a window into a wall slot.
    The task is to:
        1. Pick up the window from the ground using the vacuum gripper
        2. Insert the window into a rectangular slot in the facade wall
        3. Align the window properly (position and orientation) within the slot
    """
    # 初始化窗口装配环境
    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        random_window_position: bool = False,
        xml_path: Path | None = None,
    ):
        self.reward_type = reward_type

        # 使用KUKA窗口装配场景XML文件
        if xml_path is None:
            xml_path = Path(__file__).parent.parent / "assets" / "kuka_window_assembly_scene.xml"

        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_KUKA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
            xml_path=xml_path,
        )

        # 获取窗口插入任务的目标站点ID
        self._target_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "target_site")

        # 任务特定设置
        # 窗口是一个盒子: size[0]是半宽度, size[1]是半长度, size[2]是半厚度
        window_geom = self._model.geom("window_collision")
        self._window_thickness = window_geom.size[2]  # Half-thickness (window thickness / 2)
        # 窗口中心Z位置: window_body初始位置Z在XML文件中(0.005), 所以底部与地面接触在Z=0
        self._window_z = 0.005  # Window center at Z = 0.005 (half-thickness), so bottom touches ground at Z=0
        self._random_window_position = random_window_position

        # terminate_on_success: 是否在成功时终止episode（默认True，保持向后兼容）
        # 如果设置为False，episode会继续运行直到达到max_episode_steps，即使已经成功
        self._terminate_on_success = True  # 默认值，可以通过属性设置

        # Setup observation space properly to match what _compute_observation returns
        agent_dim = self.get_robot_state().shape[0]
        agent_box = spaces.Box(-np.inf, np.inf, (agent_dim,), dtype=np.float32)
        env_box = spaces.Box(-np.inf, np.inf, (3,), dtype=np.float32)

        if self.image_obs:
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(
                        {
                            "front": spaces.Box(
                                0,
                                255,
                                (self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
                            "wrist": spaces.Box(
                                0,
                                255,
                                (self._render_specs.height, self._render_specs.width, 3),
                                dtype=np.uint8,
                            ),
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

    # Override the reset method to set the window position to the center of the sampling area
    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        if not hasattr(self, '_np_random') or self._np_random is None:
            import gymnasium as gym
            if seed is not None:
                self._np_random = np.random.default_rng(seed)
            else:
                self._np_random = np.random.default_rng()

        mujoco.mj_resetData(self._model, self._data)

        # Reset the robot to home position
        self.reset_robot()

        # Sample a new window position
        if self._random_window_position:
            window_xy = self._np_random.uniform(*_SAMPLING_BOUNDS)
            self._data.jnt("window_joint").qpos[:3] = (*window_xy, self._window_z)
        else:
            window_xy = np.asarray([0.6, 0.0])  # Same position as plate
            self._data.jnt("window_joint").qpos[:3] = (*window_xy, self._window_z)
            
        # Reset window orientation to flat (no rotation, lying on ground)
        self._data.jnt("window_joint").qpos[3:7] = [1, 0, 0, 0]  # Identity quaternion
        
        # Reset window velocity
        self._data.jnt("window_joint").qvel[:] = 0.0
        
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial window height (center of window)
        self._z_init = self._data.sensor("window_pos").data[2]

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Apply the action to the robot (use default control parameters, same as pick plate)
        # Removed custom damping_ratio=10 which caused sluggish/rigid control when gripper is on
        self.apply_action(action)
        
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()

        if self.reward_type == "sparse":
            success = rew == 1.0

        # Check if window is outside reasonable bounds (fall prevention)
        # Only terminate if window falls significantly outside workspace
        # This prevents premature termination during normal manipulation
        window_pos = self._data.sensor("window_pos").data
        # Relaxed bounds to allow more flexible manipulation:
        # - X: Allow movement from robot base (0.0) to beyond wall (1.5) for insertion
        # - Y: Allow wider lateral movement for approach angles
        # - Z: Allow from below ground (fall) to high above workspace
        exceeded_bounds = (window_pos[0] < -0.5 or window_pos[0] > 1.5 or 
                          window_pos[1] < -1.0 or window_pos[1] > 1.0 or
                          window_pos[2] < -0.5 or window_pos[2] > 1.5)

        # 如果terminate_on_success=False，成功时不会终止，继续运行直到max_episode_steps
        if self._terminate_on_success:
            terminated = bool(success or exceeded_bounds)
        else:
            # 即使成功也不终止，只因为超出边界而终止
            terminated = bool(exceeded_bounds)
        
        # Add debug info for reward tracking
        window_pos, target_pos, pos_dist, alignment = self._get_window_state()
        window_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "window_body")
        velocity = np.linalg.norm(self._data.cvel[window_body_id][:3])
        vacuum_on = self.get_gripper_pose()[0] > 127
        
        info = {
            "succeed": success,
            "pos_dist": float(pos_dist),
            "alignment": float(alignment),
            "velocity": float(velocity),
            "vacuum_on": bool(vacuum_on),
            "window_pos": window_pos.tolist(),
            "target_pos": target_pos.tolist(),
        }
        
        # Add debug info if episode terminates early due to bounds
        if exceeded_bounds and not success:
            info["termination_reason"] = "exceeded_bounds"
            info["bounds_exceeded"] = {
                "x": window_pos[0] < -0.5 or window_pos[0] > 1.5,
                "y": window_pos[1] < -1.0 or window_pos[1] > 1.0,
                "z": window_pos[2] < -0.5 or window_pos[2] > 1.5
            }

        return obs, rew, terminated, False, info

    def _compute_observation(self) -> dict:
        """Compute the current observation."""
        observation = {}
        robot_state = self.get_robot_state().astype(np.float32)
        window_pos = self._data.sensor("window_pos").data.astype(np.float32)

        if self.image_obs:
            front_view, wrist_view = self.render()
            observation = {
                "pixels": {"front": front_view, "wrist": wrist_view},
                "agent_pos": robot_state,
            }
        else:
            observation = {
                "agent_pos": robot_state,
                "environment_state": window_pos,
            }
        return observation

    def _get_window_state(self) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """Retrieve window position, target position, position distance, and alignment.
        
        Returns:
            window_pos: Window center position (3D)
            target_pos: Target slot center position (3D)
            pos_dist: Distance between window and target centers
            alignment: Alignment score (0-1, 1.0 = perfectly aligned)
        """
        # Get positions
        window_pos = self._data.sensor("window_pos").data
        target_pos = self._data.site(self._target_site_id).xpos
        
        # Compute position distance
        # For insertion task, we care more about X-direction (insertion depth) than Y/Z alignment
        # X-direction: insertion depth (wall is at X=0.9, slot front surface is at X≈0.885)
        # Y/Z directions: lateral alignment (should be within slot bounds, but less critical)
        diff = window_pos - target_pos
        x_error = abs(diff[0])  # X-direction error (insertion depth)
        yz_error = np.sqrt(diff[1]**2 + diff[2]**2)  # Y/Z lateral error
        
        # For insertion task: window needs to be inserted into slot, not necessarily at slot center
        # If window is already past the slot front surface (X > 0.885), consider it inserted
        # Adjust X error: if window is at X > 0.885, reduce X error to account for insertion
        wall_x = 0.9  # Wall position in world X
        slot_front_x = wall_x - 0.015  # Slot front surface (frame is 0.015m thick)
        
        # If window is inserted past slot front surface, use only Y/Z error
        # Otherwise, use weighted combination
        if window_pos[0] >= slot_front_x:
            # Window is inserted, only check Y/Z alignment
            pos_dist = yz_error
        elif yz_error < 0.005:  # Y/Z error < 5mm
            # Y/Z is well aligned, use only X error (distance to slot front)
            pos_dist = max(0, slot_front_x - window_pos[0])  # Distance to slot front surface
        else:
            # Combine with reduced weight for Y/Z (Y/Z error weighted 0.3x)
            x_dist_to_front = max(0, slot_front_x - window_pos[0])
            pos_dist = np.sqrt(x_dist_to_front**2 + (yz_error * 0.3)**2)
        
        # Compute orientation alignment
        # Window should be vertical when inserted into slot
        # When window is flat on ground: quat = [1, 0, 0, 0] (identity), Z-axis points up (normal direction)
        # Wall is rotated 90° around Z-axis, so wall normal is -X direction (pointing towards robot)
        # When window is vertical in slot: Z-axis (normal) should point in -X direction (wall normal, pointing towards robot)
        window_quat = self._data.sensor("window_quat").data  # [w, x, y, z] MuJoCo format
        # Convert to rotation matrix (scipy expects [x, y, z, w])
        window_mat = R.from_quat([window_quat[1], window_quat[2], window_quat[3], window_quat[0]]).as_matrix()
        # Rotation matrix columns are the X, Y, Z axes in world frame
        window_x_axis = window_mat[:, 0]  # Window's X-axis (width direction)
        window_y_axis = window_mat[:, 1]  # Window's Y-axis (length direction)
        window_z_axis = window_mat[:, 2]  # Window's Z-axis (normal direction, thickness)
        
        # Target normal is -X direction (pointing out from wall towards robot, after 90° rotation)
        target_normal = np.array([-1, 0, 0])
        
        # Check alignment: when window is vertical in slot, Z-axis should point in -X direction
        # The alignment is simply how well the Z-axis (normal) aligns with the target normal (-X)
        z_alignment = abs(np.dot(window_z_axis, target_normal))
        
        # Calculate X-axis alignment with target normal
        # When window is vertical and inserted, X-axis (width) should point to target normal direction (-X)
        # This is the most reliable indicator that window is vertical and properly oriented
        x_alignment = abs(np.dot(window_x_axis, target_normal))
        
        # Window is properly vertical if X-axis is well-aligned with target normal (>0.95)
        # Note: When window is vertical, X-axis aligns with target normal, even if Z-axis still points vertically
        is_properly_vertical = x_alignment > 0.95
        
        # Tilt penalty: use X-axis alignment as tilt indicator (1.0 = perfect alignment, 0.0 = misaligned)
        tilt_penalty = x_alignment
        
        # For debug output: calculate additional metrics
        z_is_horizontal = abs(np.dot(window_z_axis, np.array([0, 0, 1]))) < 0.5
        x_axis_x_component = abs(window_x_axis[0])  # X-axis's X component
        y_axis_x_component = abs(window_y_axis[0])  # Y-axis's X component
        
        # Alignment: use X-axis alignment when window is properly vertical (X-axis aligned with target normal)
        # When window is vertical and inserted, X-axis should point to target normal direction
        if is_properly_vertical:
            # Window is properly vertical, use X-axis alignment (X-axis should align with target normal)
            alignment = x_alignment * tilt_penalty
        else:
            # Window is not properly vertical yet
            # Use X-axis alignment as primary indicator (even if window appears flat, X-axis alignment shows readiness)
            if pos_dist < 0.04:  # If position is very close (<4cm), use X-axis alignment
                alignment = x_alignment * 0.8  # Reduced alignment for not-yet-vertical window
            else:
                alignment = x_alignment * 0.5  # Further reduced alignment when position is not close
        
        # Debug: if position is very close but alignment is low, or if window is tilted, print warning
        if pos_dist < 0.05 and (alignment < 0.5 or not is_properly_vertical):
            print(f"[Alignment Warning] pos_dist={pos_dist*1000:.2f}mm is close, alignment={alignment:.4f}, is_properly_vertical={is_properly_vertical}")
            print(f"  window_x_axis={window_x_axis}, window_y_axis={window_y_axis}, window_z_axis={window_z_axis}")
            print(f"  z_is_horizontal={z_is_horizontal}, z_alignment={z_alignment:.4f}")
            print(f"  x_axis_x={x_axis_x_component:.4f}, y_axis_x={y_axis_x_component:.4f}, tilt_penalty={tilt_penalty:.4f}")
            print(f"  x_alignment={x_alignment:.4f} (X-axis dot with target_normal), z_dot_up={abs(np.dot(window_z_axis, np.array([0,0,1]))):.4f} (Z-axis dot with [0,0,1])")
        
        return window_pos, target_pos, pos_dist, alignment

    def _compute_reward(self) -> float:
        """Compute reward based on window insertion into wall slot.
        
        Uses staged reward design to guide agent through:
        1. Reach: Approach the window
        2. Pick: Grasp the window with vacuum gripper
        3. Lift: Raise window to reduce ground friction
        4. Align: Orient window correctly for insertion
        5. Insert: Push window into wall slot
        """
        # Glass breakage detection (if implemented)
        if hasattr(self, "is_broken") and self.is_broken:
            return -20.0
        
        # Get window state (position, target, distance, alignment)
        window_pos, target_pos, pos_dist, alignment = self._get_window_state()
        
        # Vacuum gripper state (should be ON when holding window)
        vacuum_on = self.get_gripper_pose()[0] > 127
        
        if self.reward_type == "dense":
            # Get TCP position for reach reward
            tcp_pos = self._data.sensor("2f85/pinch_pos").data
            
            # A. Reach reward: TCP close to window (using exp for better near-distance gradient)
            dist_tcp_window = np.linalg.norm(tcp_pos - window_pos)
            r_reach = np.exp(-20 * dist_tcp_window)
            
            # B. Lift reward: Window raised above ground (only when vacuum is on)
            # This helps reduce ground friction for X-axis pushing motion
            lift_height = window_pos[2] - self._z_init
            r_lift = np.clip(lift_height / 0.15, 0, 1.0) if vacuum_on else 0.0
            
            # B2. Rotation reward: Window rotated from flat to vertical (only when vacuum is on)
            # Encourage rotating the window 90 degrees from lying flat to standing vertical (parallel to wall)
            # When window is flat: Z-axis points up/down (dot product with [0,0,1] is close to 1)
            # When window is vertical: Z-axis points horizontally (dot product with [0,0,1] is close to 0)
            # Note: Rotation can be around ry (Y-axis), which rotates Z-axis from vertical to horizontal
            window_quat = self._data.sensor("window_quat").data  # [w, x, y, z] MuJoCo format
            window_mat = R.from_quat([window_quat[1], window_quat[2], window_quat[3], window_quat[0]]).as_matrix()
            window_z_axis = window_mat[:, 2]  # Window's Z-axis (normal direction)
            window_x_axis = window_mat[:, 0]  # Window's X-axis (width direction)
            window_y_axis = window_mat[:, 1]  # Window's Y-axis (length direction)
            
            # Calculate how much Z-axis points up/down
            z_vertical_component = abs(np.dot(window_z_axis, np.array([0, 0, 1])))  # How much Z-axis points up/down
            
            # Calculate X/Y axis vertical components (for rotation detection and debug)
            x_axis_vertical = abs(np.dot(window_x_axis, np.array([0, 0, 1])))  # X-axis vertical component
            y_axis_vertical = abs(np.dot(window_y_axis, np.array([0, 0, 1])))  # Y-axis vertical component
            
            # Rotation reward: 1.0 when window is vertical (z_vertical_component ≈ 0), 0.0 when flat (z_vertical_component ≈ 1)
            # Use linear scaling to give clear reward signal during rotation process
            # Also check if window is rotating: if Z-axis is changing from vertical, give reward
            if vacuum_on:
                # When vacuum is on, encourage rotation to vertical
                # z_vertical_component: 1.0 = flat (Z-axis pointing up), 0.0 = vertical (Z-axis pointing horizontally)
                # r_rotate: 0.0 = flat, 1.0 = vertical
                
                # Primary rotation metric: Z-axis vertical component
                # This directly measures how much the window has rotated from flat to vertical
                z_based_rotate = 1.0 - np.clip(z_vertical_component, 0, 1.0)
                
                # Secondary rotation metric: X-axis vertical component (for ry rotation detection)
                # When rotating around ry: X-axis rotates from horizontal (Z=0) to vertical (Z=±1)
                # This provides additional signal during rotation, especially when Z-axis changes are small
                x_based_rotate = x_axis_vertical  # X-axis vertical component directly indicates rotation progress
                
                # Combine both metrics for dense reward signal
                # Use the maximum to ensure we capture rotation progress from either axis
                # Weight X-axis more heavily when it's changing (indicating active rotation)
                if x_axis_vertical > 0.1:  # X-axis is actively rotating
                    # During active rotation, give more weight to X-axis (which changes more smoothly)
                    # Blend: 70% X-axis, 30% Z-axis for smoother signal during rotation
                    r_rotate = 0.7 * x_based_rotate + 0.3 * z_based_rotate
                else:
                    # When X-axis is not rotating much, use Z-axis as primary metric
                    r_rotate = z_based_rotate
                
                # Ensure rotation reward is always positive when there's any rotation
                # This provides dense feedback throughout the entire rotation process
                if z_vertical_component < 0.999 or x_axis_vertical > 0.01:
                    # Window is rotating (either Z-axis changed or X-axis is rotating)
                    # Use the maximum of both metrics to ensure we capture all rotation progress
                    r_rotate = max(r_rotate, x_based_rotate * 0.8, z_based_rotate)  # Blend both signals
                
                # Apply boost for intermediate rotations to encourage progress
                # When window is partially rotated (0.3 < z_vertical_component < 0.7), give extra reward
                if 0.3 < z_vertical_component < 0.7:
                    r_rotate = r_rotate * 1.2  # 20% boost during rotation process
                
                # Ensure rotation reward smoothly transitions during rotation
                # Add small bonus for any rotation progress to make signal more dense
                if r_rotate > 0.01:  # Any rotation detected
                    # Give slight boost to make rotation signal more prominent
                    r_rotate = min(r_rotate * 1.05, 1.0)  # 5% boost, capped at 1.0
                
                r_rotate = np.clip(r_rotate, 0, 1.0)  # Ensure reward stays in [0, 1]
            else:
                r_rotate = 0.0  # No rotation reward when not grasping
            
            # C. Alignment reward: Window orientation (using square to enhance precision signal)
            # Critical for avoiding singularities during pushing phase
            # Use linear scaling instead of square to allow higher reward when alignment is good
            # When alignment >= 0.85, use linear scaling; otherwise use square for precision signal
            if alignment >= 0.85:
                r_align = alignment  # Linear scaling for high alignment (allows reward >= 0.85)
            else:
                r_align = np.power(np.clip(alignment, 0, 1), 2)  # Square for lower alignment (precision signal)
            
            # D. Insert reward: Window close to target slot (using exp for better gradient)
            # Reduce decay rate to allow higher reward when position is very close
            # Use progressively gentler decay as position gets closer
            # Check if window is already inserted past slot front surface
            wall_x = 0.9
            slot_front_x = wall_x - 0.015
            is_inserted = window_pos[0] >= slot_front_x
            
            if is_inserted:
                # Window is inserted: pos_dist is mainly Y/Z error, should be very small
                # Use very gentle decay for inserted windows
                if pos_dist < 0.005:  # Y/Z error < 5mm (excellent alignment)
                    r_insert = 0.99  # Almost perfect
                elif pos_dist < 0.010:  # Y/Z error < 10mm (good alignment)
                    r_insert = 0.95
                elif pos_dist < 0.015:  # Y/Z error < 15mm (acceptable)
                    r_insert = np.exp(-20 * pos_dist)  # Very gentle decay
                else:
                    r_insert = np.exp(-25 * pos_dist)  # Still gentle, but penalize larger Y/Z errors
            elif pos_dist < 0.012:  # Very close (<12mm) but not yet inserted
                r_insert = np.exp(-8 * pos_dist)  # Very gentle decay for extremely close positions
            elif pos_dist < 0.02:  # Close (<20mm)
                r_insert = np.exp(-10 * pos_dist)  # Gentler decay for very close positions
            else:
                r_insert = np.exp(-15 * pos_dist)  # Original decay for farther positions
            
            # Weight allocation based on task phase
            if not vacuum_on:
                # Phase 1: Guide to grasp window (emphasize reach, slight hint of target)
                reward = 0.8 * r_reach + 0.2 * r_insert
            else:
                # Phase 2: Guide alignment and insertion (emphasize alignment and lift)
                # Higher lift weight forces agent to raise window, reducing ground friction
                # When window is already inserted (pos_dist very small), ignore reach reward
                # Check if window is inserted past slot front surface
                wall_x = 0.9
                slot_front_x = wall_x - 0.015
                is_inserted = window_pos[0] >= slot_front_x
                
                if is_inserted and pos_dist < 0.015 and alignment >= 0.85:
                    # Window is inserted and well-aligned: ignore reach and rotation, focus on insertion quality
                    # When inserted, window should already be vertical, so rotation reward is less important
                    if pos_dist < 0.005:  # Y/Z error < 5mm (excellent)
                        # Perfect insertion: reward should be very high
                        reward = 0.05 * r_lift + 0.35 * r_align + 0.60 * r_insert
                    else:
                        # Good insertion: still high reward
                        reward = 0.10 * r_lift + 0.40 * r_align + 0.50 * r_insert
                elif pos_dist < 0.015 and alignment >= 0.85:  # Very close (<15mm) and well aligned
                    # Extremely close to success: heavily emphasize insertion, minimize reach and rotation impact
                    reward = 0.02 * r_reach + 0.08 * r_lift + 0.05 * r_rotate + 0.30 * r_align + 0.55 * r_insert
                elif pos_dist < 0.02 and alignment >= 0.85:  # Close (<20mm) and well aligned
                    # Very close to success: emphasize insertion and alignment, some rotation
                    reward = 0.05 * r_reach + 0.12 * r_lift + 0.08 * r_rotate + 0.30 * r_align + 0.45 * r_insert
                else:
                    # Normal phase: balanced weights, emphasize rotation when window is not yet vertical
                    # Rotation is important early in the task (after grasping, before insertion)
                    # Dynamically adjust weights based on rotation state
                    # If window is still flat (r_rotate < 0.3), emphasize rotation more
                    if r_rotate < 0.3:  # Window is still mostly flat
                        # Early rotation phase: heavily emphasize rotation
                        reward = 0.05 * r_reach + 0.10 * r_lift + 0.35 * r_rotate + 0.30 * r_align + 0.20 * r_insert
                    elif r_rotate < 0.7:  # Window is partially rotated
                        # Mid rotation phase: still emphasize rotation
                        reward = 0.06 * r_reach + 0.12 * r_lift + 0.25 * r_rotate + 0.32 * r_align + 0.25 * r_insert
                    else:
                        # Window is mostly vertical: normal weights
                        reward = 0.08 * r_reach + 0.15 * r_lift + 0.15 * r_rotate + 0.35 * r_align + 0.27 * r_insert
            
            # Debug: Print actual values when close to success or during rotation (for dense reward)
            # Also print when rotation is in progress (z_vertical_component is changing, indicating rotation)
            # Check if window is rotating: z_vertical_component should decrease from ~1.0 to ~0.0 during rotation
            rotation_in_progress = (vacuum_on and z_vertical_component < 0.99)  # Window is being rotated (not perfectly flat)
            if pos_dist < 0.05 or alignment > 0.85 or rotation_in_progress:  # Print when close to success or rotating
                window_quat = self._data.sensor("window_quat").data
                window_mat = R.from_quat([window_quat[1], window_quat[2], window_quat[3], window_quat[0]]).as_matrix()
                window_normal = window_mat[:, 2]
                target_normal = np.array([-1, 0, 0])
                dot_product = np.dot(window_normal, target_normal)
                is_success_sparse = (pos_dist < 0.04 and alignment > 0.80)
                print(f"[Reward Debug] pos_dist={pos_dist*1000:.2f}mm (need <40mm), alignment={alignment:.4f} (need >0.80), sparse_success={is_success_sparse}")
                print(f"  window_pos={window_pos}, target_pos={target_pos}, diff={(window_pos-target_pos)*1000}")
                print(f"  window_normal={window_normal}, target_normal={target_normal}, dot={dot_product:.4f}")
                # Calculate X/Y axis vertical components for debug
                x_axis_vertical = abs(np.dot(window_mat[:, 0], np.array([0, 0, 1])))
                y_axis_vertical = abs(np.dot(window_mat[:, 1], np.array([0, 0, 1])))
                # Recalculate x/y axis vertical components for debug (in case window_mat was recalculated)
                debug_x_vertical = abs(np.dot(window_mat[:, 0], np.array([0, 0, 1])))
                debug_y_vertical = abs(np.dot(window_mat[:, 1], np.array([0, 0, 1])))
                print(f"  dense_reward={reward:.4f}, r_reach={r_reach:.4f}, r_lift={r_lift:.4f}, r_rotate={r_rotate:.4f} (z_vertical={z_vertical_component:.4f}, x_vertical={debug_x_vertical:.4f}, y_vertical={debug_y_vertical:.4f}, vacuum_on={vacuum_on}), r_align={r_align:.4f}, r_insert={r_insert:.4f}")
            
            return float(np.clip(reward, -20.0, 1.0))
        else:
            # Sparse reward: 1.0 if successfully inserted, 0.0 otherwise
            # Success criteria: close to target position (within 3cm) and well aligned (>90%)
            # Use same criteria as _is_success for consistency
            is_success = (pos_dist < 0.005 and alignment > 0.90)
            
            # Debug: Print actual values when close to success
            if pos_dist < 0.05 or alignment > 0.85:  # Print when close to success criteria
                window_quat = self._data.sensor("window_quat").data
                window_mat = R.from_quat([window_quat[1], window_quat[2], window_quat[3], window_quat[0]]).as_matrix()
                window_normal = window_mat[:, 2]
                target_normal = np.array([-1, 0, 0])
                dot_product = np.dot(window_normal, target_normal)
                print(f"[Reward Debug] pos_dist={pos_dist*1000:.2f}mm (need <30mm), alignment={alignment:.4f} (need >0.90), success={is_success}")
                print(f"  window_pos={window_pos}, target_pos={target_pos}, diff={(window_pos-target_pos)*1000}")
                print(f"  window_normal={window_normal}, target_normal={target_normal}, dot={dot_product:.4f}")
            
            return 1.0 if is_success else 0.0

    def _is_success(self) -> bool:
        """Check if the window is successfully inserted into the wall slot.
        
        Success condition:
        - Window center is close to target slot center (within 3cm)
        - Window is properly aligned (normal aligned with wall normal, >90% alignment)
        - Window is stable (velocity < 1cm/s to ensure truly inserted, not just touching)
        """
        # Get window state (reuse the same computation logic)
        window_pos, target_pos, pos_dist, alignment = self._get_window_state()
        
        # Stability check: ensure window is truly inserted, not just touching
        window_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "window_body")
        # Get 6D velocity (first 3 elements are linear velocity)
        velocity = np.linalg.norm(self._data.cvel[window_body_id][:3])
        is_stable = velocity < 0.01  # 1cm/s threshold
        
        # Position and alignment checks (relaxed thresholds: 3cm, 90%)
        is_inserted = pos_dist < 0.005
        is_aligned = alignment > 0.90
        
        return is_inserted and is_aligned and is_stable


if __name__ == "__main__":
    from gym_hil import PassiveViewerWrapper

    env = KukaWindowAssemblyEnv(render_mode="human")
    env = PassiveViewerWrapper(env)
    obs, info = env.reset()
    
    print("Environment reset successful!")
    print(f"Initial window position: {obs['environment_state']}")
    print(f"Initial robot state shape: {obs['agent_pos'].shape}")
    
    print("\nTesting vacuum gripper control...")
    for i in range(50):
        if i < 20:
            action = np.array([0, 0, -0.5, 0, 0, 0, 1.0])  # Move down, turn on vacuum
        elif i < 30:
            action = np.array([0, 0, 0, 0, 0, 0, 1.0])  # Keep vacuum on
        else:
            action = np.array([0, 0, 0.5, 0, 0, 0, 1.0])  # Lift up, keep vacuum on
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            window_pos = obs['environment_state']
            gripper_state = env.unwrapped.get_gripper_pose()
            print(f"Step {i}: Window Z={window_pos[2]:.3f}, Gripper={gripper_state[0]:.0f}, Reward={reward:.2f}")
        
        if terminated:
            print(f"Episode terminated: {info}")
            break
    
    env.close()

