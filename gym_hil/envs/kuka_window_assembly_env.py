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

_KUKA_HOME = np.asarray((0, 0.785398, 0, -1.5708, 0, 0.785398, 0))
_CARTESIAN_BOUNDS = np.asarray([[0.0, 0.0, 0], [0.9, 0.0, 0.439]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.6, 0.15]]) 
class KukaWindowAssemblyEnv(FrankaGymEnv):
    """Environment for a KUKA iiwa14 robot with vacuum gripper inserting a window into a wall slot.
    The task is to:
        1. Pick up the window from the ground using the vacuum gripper
        2. Insert the window into a rectangular slot in the facade wall
        3. Align the window properly (position and orientation) within the slot
    """
    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = True,
        reward_type: str = "sparse",
        random_window_position: bool = False,
        xml_path: Path | None = None,   
    ):
        self.reward_type = reward_type

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

        self._target_site_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_SITE, "target_site")
        window_geom = self._model.geom("window_collision")
        self._window_thickness = window_geom.size[2]
        self._window_z = 0.005
        self._random_window_position = random_window_position
        self._terminate_on_success = True
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

    def reset(self, seed=None, **kwargs) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        if not hasattr(self, '_np_random') or self._np_random is None:
            if seed is not None:
                self._np_random = np.random.default_rng(seed)
            else:
                self._np_random = np.random.default_rng()

        mujoco.mj_resetData(self._model, self._data)
        self.reset_robot()

        if self._random_window_position:
            window_xy = self._np_random.uniform(*_SAMPLING_BOUNDS)
            self._data.jnt("window_joint").qpos[:3] = (*window_xy, self._window_z)
        else:
            initial_window_pos = self._data.jnt("window_joint").qpos[:3]
            window_xy = initial_window_pos[:2]
            self._data.jnt("window_joint").qpos[:3] = (*window_xy, self._window_z)
            
        self._data.jnt("window_joint").qpos[3:7] = [1, 0, 0, 0]
        self._data.jnt("window_joint").qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._z_init = self._data.sensor("window_pos").data[2]

        obs = self._compute_observation()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        self.apply_action(action)
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()

        if self.reward_type == "sparse":
            success = rew == 1.0

        window_pos = self._data.sensor("window_pos").data
        exceeded_bounds = (window_pos[0] < -0.5 or window_pos[0] > 1.5 or 
                          window_pos[1] < -1.0 or window_pos[1] > 1.0 or
                          window_pos[2] < -0.5 or window_pos[2] > 1.5)

        if self._terminate_on_success:
            terminated = bool(success or exceeded_bounds)
        else:
            terminated = bool(exceeded_bounds)
        
        return obs, rew, terminated, False, {}

    def _compute_observation(self) -> dict:
        """Compute current observation."""
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
        """Get window position, target position, distance, and alignment."""
        window_pos = self._data.sensor("window_pos").data
        target_pos = self._data.site(self._target_site_id).xpos
        
        diff = window_pos - target_pos
        x_error = abs(diff[0])
        yz_error = np.sqrt(diff[1]**2 + diff[2]**2)
    
        wall_x = 0.9
        slot_front_x = wall_x - 0.015
        
        if window_pos[0] >= slot_front_x:
            pos_dist = yz_error
        elif yz_error < 0.005:
            pos_dist = max(0, slot_front_x - window_pos[0])
        else:
            x_dist_to_front = max(0, slot_front_x - window_pos[0])
            pos_dist = np.sqrt(x_dist_to_front**2 + (yz_error * 0.3)**2)
        
        window_quat = self._data.sensor("window_quat").data
        window_mat = R.from_quat([window_quat[1], window_quat[2], window_quat[3], window_quat[0]]).as_matrix()
        window_x_axis = window_mat[:, 0]
        window_y_axis = window_mat[:, 1]
        window_z_axis = window_mat[:, 2]
        
        target_normal = np.array([-1, 0, 0])
        z_alignment = abs(np.dot(window_z_axis, target_normal))
        z_is_horizontal = abs(np.dot(window_z_axis, np.array([0, 0, 1]))) < 0.5
        
        x_axis_x_component = abs(window_x_axis[0])
        y_axis_x_component = abs(window_y_axis[0])
        x_axis_z_component = abs(window_x_axis[2])
        y_axis_z_component = abs(window_y_axis[2])
        
        x_axis_in_yz_plane = x_axis_x_component < 0.15
        y_axis_in_yz_plane = y_axis_x_component < 0.15
        min_z_component = min(x_axis_z_component, y_axis_z_component)
        max_z_component = max(x_axis_z_component, y_axis_z_component)
        axes_properly_oriented = (min_z_component < 0.2 and max_z_component > 0.8)
        
        is_properly_vertical = (z_is_horizontal and 
                                x_axis_in_yz_plane and 
                                y_axis_in_yz_plane and
                                axes_properly_oriented)
        
        x_y_plane_deviation = max(x_axis_x_component, y_axis_x_component)
        orientation_deviation = min(min_z_component, 1.0 - max_z_component)
        max_tilt = max(x_y_plane_deviation * 3.0, orientation_deviation * 2.0)
        tilt_penalty = 1.0 - min(max_tilt, 1.0)
        
        if z_is_horizontal and is_properly_vertical:
            alignment = z_alignment * tilt_penalty
        elif z_is_horizontal:
            alignment = z_alignment * tilt_penalty * 0.5
        else:
            if z_alignment > 0.3:
                alignment = z_alignment * 0.7
            elif pos_dist < 0.005:
                alignment = 0.85
            elif pos_dist < 0.01:
                alignment = max(z_alignment * 0.6, 0.6)
            elif pos_dist < 0.03:
                if is_properly_vertical:
                    alignment = 0.85
                else:
                    alignment = z_alignment * 0.5
            elif pos_dist < 0.05:
                if is_properly_vertical:
                    alignment = max(z_alignment * 0.5, 0.7)
                else:
                    alignment = z_alignment * 0.3
            else:
                alignment = 0.0
        
        if pos_dist < 0.05 and (alignment < 0.5 or not is_properly_vertical):
            print(f"[Alignment Warning] pos_dist={pos_dist*1000:.2f}mm, alignment={alignment:.4f}, is_properly_vertical={is_properly_vertical}")
        
        return window_pos, target_pos, pos_dist, alignment

    def _compute_reward(self) -> float:
        """Compute reward based on window insertion into wall slot."""
        if hasattr(self, "is_broken") and self.is_broken:
            return -20.0
        
        window_pos, target_pos, pos_dist, alignment = self._get_window_state()
        vacuum_on = self.get_gripper_pose()[0] > 127
        
        if self.reward_type == "dense":
            tcp_pos = self._data.sensor("2f85/pinch_pos").data
            dist_tcp_window = np.linalg.norm(tcp_pos - window_pos)
            r_reach = np.exp(-20 * dist_tcp_window)
            
            lift_height = window_pos[2] - self._z_init
            r_lift = np.clip(lift_height / 0.15, 0, 1.0) if vacuum_on else 0.0
            
            window_quat = self._data.sensor("window_quat").data
            window_mat = R.from_quat([window_quat[1], window_quat[2], window_quat[3], window_quat[0]]).as_matrix()
            window_z_axis = window_mat[:, 2]
            window_x_axis = window_mat[:, 0]
            window_y_axis = window_mat[:, 1]
            
            z_vertical_component = abs(np.dot(window_z_axis, np.array([0, 0, 1])))
            x_axis_vertical = abs(np.dot(window_x_axis, np.array([0, 0, 1])))
            y_axis_vertical = abs(np.dot(window_y_axis, np.array([0, 0, 1])))
            
            if vacuum_on:
                z_based_rotate = 1.0 - np.clip(z_vertical_component, 0, 1.0)
                x_based_rotate = x_axis_vertical
                
                if x_axis_vertical > 0.1:
                    r_rotate = 0.7 * x_based_rotate + 0.3 * z_based_rotate
                else:
                    r_rotate = z_based_rotate
                
                if z_vertical_component < 0.999 or x_axis_vertical > 0.01:
                    r_rotate = max(r_rotate, x_based_rotate * 0.8, z_based_rotate)
                
                if 0.3 < z_vertical_component < 0.7:
                    r_rotate = r_rotate * 1.2
                
                if r_rotate > 0.01:
                    r_rotate = min(r_rotate * 1.05, 1.0)
                
                r_rotate = np.clip(r_rotate, 0, 1.0)
            else:
                r_rotate = 0.0
            
            if alignment >= 0.85:
                r_align = alignment
            else:
                r_align = np.power(np.clip(alignment, 0, 1), 2)
            
            wall_x = 0.9
            slot_front_x = wall_x - 0.015
            is_inserted = window_pos[0] >= slot_front_x
            
            if is_inserted:
                if pos_dist < 0.005:
                    r_insert = 0.99
                elif pos_dist < 0.010:
                    r_insert = 0.95
                elif pos_dist < 0.015:
                    r_insert = np.exp(-20 * pos_dist)
                else:
                    r_insert = np.exp(-25 * pos_dist)
            elif pos_dist < 0.012:
                r_insert = np.exp(-8 * pos_dist)
            elif pos_dist < 0.02:
                r_insert = np.exp(-10 * pos_dist)
            else:
                r_insert = np.exp(-15 * pos_dist)
            
            if not vacuum_on:
                reward = 0.8 * r_reach + 0.2 * r_insert
            else:
                wall_x = 0.9
                slot_front_x = wall_x - 0.015
                is_inserted = window_pos[0] >= slot_front_x
                
                if is_inserted and pos_dist < 0.015 and alignment >= 0.85:
                    if pos_dist < 0.005:
                        reward = 0.05 * r_lift + 0.35 * r_align + 0.60 * r_insert
                    else:
                        reward = 0.10 * r_lift + 0.40 * r_align + 0.50 * r_insert
                elif pos_dist < 0.015 and alignment >= 0.85:
                    reward = 0.02 * r_reach + 0.08 * r_lift + 0.05 * r_rotate + 0.30 * r_align + 0.55 * r_insert
                elif pos_dist < 0.02 and alignment >= 0.85:
                    reward = 0.05 * r_reach + 0.12 * r_lift + 0.08 * r_rotate + 0.30 * r_align + 0.45 * r_insert
                else:
                    if r_rotate < 0.3:
                        reward = 0.05 * r_reach + 0.10 * r_lift + 0.35 * r_rotate + 0.30 * r_align + 0.20 * r_insert
                    elif r_rotate < 0.7:
                        reward = 0.06 * r_reach + 0.12 * r_lift + 0.25 * r_rotate + 0.32 * r_align + 0.25 * r_insert
                    else:
                        reward = 0.08 * r_reach + 0.15 * r_lift + 0.15 * r_rotate + 0.35 * r_align + 0.27 * r_insert
            
            return float(np.clip(reward, -20.0, 1.0))
        else:
            is_success = (pos_dist < 0.002 and alignment >= 0.85)
            
            if pos_dist < 0.005 or alignment >= 0.85:
                window_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "window_body")
                velocity = np.linalg.norm(self._data.cvel[window_body_id][:3])
                print(f"[Success Check] pos_dist={pos_dist*1000:.2f}mm, alignment={alignment:.4f}, velocity={velocity*100:.2f}cm/s, success={is_success}")
            
            return 1.0 if is_success else 0.0

    def _is_success(self) -> bool:
        """Check if window is successfully inserted: pos_dist < 2mm, alignment >= 85%, velocity < 1cm/s."""
        window_pos, target_pos, pos_dist, alignment = self._get_window_state()
        window_body_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "window_body")
        velocity = np.linalg.norm(self._data.cvel[window_body_id][:3])
        is_stable = velocity < 0.01
        is_inserted = pos_dist < 0.002
        is_aligned = alignment >= 0.85
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
            action = np.array([0, 0, -0.5, 0, 0, 0, 1.0])
        elif i < 30:
            action = np.array([0, 0, 0, 0, 0, 0, 1.0])
        else:
            action = np.array([0, 0, 0.5, 0, 0, 0, 1.0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            window_pos = obs['environment_state']
            gripper_state = env.unwrapped.get_gripper_pose()
            print(f"Step {i}: Window Z={window_pos[2]:.3f}, Gripper={gripper_state[0]:.0f}, Reward={reward:.2f}")
        
        if terminated:
            print(f"Episode terminated: {info}")
            break
    
    env.close()

