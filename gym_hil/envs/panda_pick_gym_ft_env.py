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

from typing import Any, Dict, Literal, Tuple

import mujoco
import numpy as np
from gymnasium import spaces

from gym_hil.mujoco_gym_env import FrankaGymEnv, GymRenderingSpec

_PANDA_HOME = np.asarray((0, 0.195, 0, -2.43, 0, 2.62, 0.785))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])


class PandaPickCubeGymFtEnv(FrankaGymEnv):
    """Environment for a Panda robot picking up a cube with Force/Torque sensing.
    
    This environment extends PandaPickCubeGymEnv by including force and torque
    measurements from the robot's sensors. The force/torque information is included
    in the observation space and can be used for learning force-aware manipulation policies.
    """

    def __init__(
        self,
        seed: int = 0,
        control_dt: float = 0.1,
        physics_dt: float = 0.002,
        render_spec: GymRenderingSpec = GymRenderingSpec(),  # noqa: B008
        render_mode: Literal["rgb_array", "human"] = "rgb_array",
        image_obs: bool = False,
        reward_type: str = "sparse",
        random_block_position: bool = False,
        include_velocity: bool = True,
    ):
        self.reward_type = reward_type
        self.include_velocity = include_velocity

        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_PANDA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
        )

        # Task-specific setup
        self._block_z = self._model.geom("block").size[2]
        self._random_block_position = random_block_position

        # Setup observation space properly to match what _compute_observation returns
        # Observation space design:
        #   - "state":  agent (robot) configuration as a single Box
        #     Includes: joint positions, velocities (optional), gripper pose (raw 0-255), TCP position, force/torque
        #   - "environment_state": block position in the world as a single Box
        #   - "pixels": (optional) dict of camera views if image observations are enabled

        # Base robot state: joint positions (7D) + gripper (1D, raw 0-255) + TCP position (3D) = 11D
        base_robot_dim = 11
        
        # Add velocity if enabled: joint velocities (7D) = 7D
        if self.include_velocity:
            base_robot_dim += 7
        
        # Force/Torque dimensions: wrist_force (3D) + joint_torques (7 joints × 3D each = 21D) = 24D
        force_torque_dim = 24
        
        agent_dim = base_robot_dim + force_torque_dim
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
        # Ensure gymnasium internal RNG is initialized when a seed is provided
        super().reset(seed=seed)
        
        # Ensure _np_random is initialized (Gymnasium should do this, but handle edge case)
        if not hasattr(self, '_np_random') or self._np_random is None:
            import gymnasium as gym
            # Create a new Generator if not initialized
            if seed is not None:
                self._np_random = np.random.default_rng(seed)
            else:
                self._np_random = np.random.default_rng()

        mujoco.mj_resetData(self._model, self._data)

        # Reset the robot to home position
        self.reset_robot()

        # Sample a new block position
        # Use self._np_random to ensure determinism when seed is provided
        if self._random_block_position:
            block_xy = self._np_random.uniform(*_SAMPLING_BOUNDS)
            self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        else:
            block_xy = np.asarray([0.5, 0.0])
            self._data.jnt("block").qpos[:3] = (*block_xy, self._block_z)
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial block height
        self._z_init = self._data.sensor("block_pos").data[2]
        self._z_success = self._z_init + 0.1

        obs = self._compute_observation()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        # Apply the action to the robot
        self.apply_action(action)

        # Compute observation, reward and termination
        obs = self._compute_observation()
        rew = self._compute_reward()
        success = self._is_success()

        if self.reward_type == "sparse":
            success = rew == 1.0

        # Check if block is outside bounds
        block_pos = self._data.sensor("block_pos").data
        exceeded_bounds = np.any(block_pos[:2] < (_SAMPLING_BOUNDS[0] - 0.05)) or np.any(
            block_pos[:2] > (_SAMPLING_BOUNDS[1] + 0.05)
        )

        terminated = bool(success or exceeded_bounds)

        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> dict:
        """Compute the current observation."""
        # Create the dictionary structure that matches our observation space
        observation = {}

        # Get robot state components
        # Joint positions (7D)
        qpos = self._data.qpos[self._panda_dof_ids].astype(np.float32)
        
        # Gripper pose (1D) - return raw value (0-255) to match non-FT version
        # This ensures consistency between FT and non-FT versions
        # Both versions now return raw gripper values (0-255)
        gripper_pose = self.get_gripper_pose().astype(np.float32)
        
        # TCP position (3D)
        tcp_pos = self._data.sensor("2f85/pinch_pos").data.astype(np.float32)
        
        # Build base robot state
        robot_state_parts = [qpos, gripper_pose, tcp_pos]
        
        # Add velocities if enabled
        if self.include_velocity:
            qvel = self._data.qvel[self._panda_dof_ids].astype(np.float32)
            robot_state_parts.insert(1, qvel)  # Insert after qpos
        
        # Concatenate base robot state
        robot_state = np.concatenate(robot_state_parts)
        
        # Add force/torque information (always included in this environment)
        force_torque = self._get_force_torque().astype(np.float32)
        robot_state = np.concatenate([robot_state, force_torque])

        # Assemble observation respecting the newly defined observation_space
        block_pos = self._data.sensor("block_pos").data.astype(np.float32)

        if self.image_obs:
            # Image observations
            front_view, wrist_view = self.render()
            observation = {
                "pixels": {"front": front_view, "wrist": wrist_view},
                "agent_pos": robot_state,
            }
        else:
            # State-only observations
            observation = {
                "agent_pos": robot_state,
                "environment_state": block_pos,
            }

        return observation

    def _get_force_torque(self) -> np.ndarray:
        """Get force and torque measurements from sensors.
        
        Returns:
            np.ndarray: Concatenated array of [wrist_force (3D), joint_torques (21D)]
                       Total dimension: 24D
                       - wrist_force: [Fx, Fy, Fz] from wrist force sensor (contact forces at the wrist)
                       - joint_torques: [Tx1, Ty1, Tz1, ..., Tx7, Ty7, Tz7] from 7 joint torque sensors
                                       Each joint torque sensor returns 3D torque vector [Tx, Ty, Tz]
        """
        try:
            # Get wrist force (3D) - contact forces at the wrist
            wrist_force = self._data.sensor("panda/wrist_force").data.copy()
            
            # Get joint torques (7 joints × 3D each = 21D)
            # Each joint torque sensor returns 3D torque vector [Tx, Ty, Tz]
            joint_torques = []
            for i in range(1, 8):
                torque = self._data.sensor(f"panda/joint{i}_torque").data.copy()
                joint_torques.append(torque)
            
            joint_torques_array = np.concatenate(joint_torques)
            
            # Concatenate: [wrist_force (3D), joint_torques (21D)]
            return np.concatenate([wrist_force, joint_torques_array]).astype(np.float32)
        except (AttributeError, ValueError, KeyError) as e:
            # If sensors are not available, return zeros
            import warnings
            warnings.warn(f"Failed to read force/torque sensors: {e}, returning zeros")
            return np.zeros(24, dtype=np.float32)

    def _compute_reward(self) -> float:
        """Compute reward based on current state."""
        block_pos = self._data.sensor("block_pos").data

        if self.reward_type == "dense":
            tcp_pos = self._data.sensor("2f85/pinch_pos").data
            dist = np.linalg.norm(block_pos - tcp_pos)
            r_close = np.exp(-20 * dist)
            r_lift = (block_pos[2] - self._z_init) / (self._z_success - self._z_init)
            r_lift = np.clip(r_lift, 0.0, 1.0)
            return 0.3 * r_close + 0.7 * r_lift
        else:
            lift = block_pos[2] - self._z_init
            return float(lift > 0.1)

    def _is_success(self) -> bool:
        """Check if the task is successfully completed."""
        block_pos = self._data.sensor("block_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        dist = np.linalg.norm(block_pos - tcp_pos)
        lift = block_pos[2] - self._z_init
        return dist < 0.05 and lift > 0.1


if __name__ == "__main__":
    from gym_hil import PassiveViewerWrapper

    # Test environment with force/torque enabled
    env = PandaPickCubeGymFtEnv(render_mode="human", include_velocity=True)
    env = PassiveViewerWrapper(env)
    obs, info = env.reset()
    print(f"Observation space: {env.observation_space}")
    print(f"Observation keys: {obs.keys()}")
    print(f"Agent state shape: {obs['agent_pos'].shape}")
    print(f"Force/Torque included: {obs['agent_pos'].shape[0] >= 35}")  # Should be 35+ with FT
    
    for _ in range(100):
        obs, reward, terminated, truncated, info = env.step(np.random.uniform(-1, 1, 7))
        if terminated:
            break
    env.close()

