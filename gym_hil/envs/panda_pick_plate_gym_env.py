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

_PANDA_HOME = np.asarray((0, 0.195, 0, -2.43, 0, 2.62, 0.785))
_CARTESIAN_BOUNDS = np.asarray([[0.2, -0.3, 0], [0.6, 0.3, 0.5]])
_SAMPLING_BOUNDS = np.asarray([[0.3, -0.15], [0.5, 0.15]])


class PandaPickPlateGymEnv(FrankaGymEnv):
    """Environment for a Panda robot with vacuum gripper picking up a plate.
    
    This environment is similar to PandaPickCubeGymEnv but uses a vacuum gripper
    instead of the 2f85 gripper, and the target object is a plate (circular disk)
    instead of a cube. The task is to use the vacuum gripper to suck up the plate
    and lift it to a certain height.
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
        random_plate_position: bool = False,
        xml_path: Path | None = None,
    ):
        self.reward_type = reward_type

        # Use vacuum gripper XML by default
        if xml_path is None:
            xml_path = Path(__file__).parent.parent / "assets" / "panda_pick_plate_scene.xml"

        super().__init__(
            seed=seed,
            control_dt=control_dt,
            physics_dt=physics_dt,
            render_spec=render_spec,
            render_mode=render_mode,
            image_obs=image_obs,
            home_position=_PANDA_HOME,
            cartesian_bounds=_CARTESIAN_BOUNDS,
            xml_path=xml_path,
        )

        # Task-specific setup
        # Plate is a box: size[0] is half-length, size[1] is half-width, size[2] is half-height (thickness)
        plate_geom = self._model.geom("plate")
        self._plate_thickness = plate_geom.size[2]  # Half-height (plate thickness / 2)
        # Plate center Z position: plate_body initial pos Z in XML (0.005), so bottom touches ground at Z=0
        self._plate_z = self._plate_thickness  # Plate center at Z = half-height (0.005)
        self._random_plate_position = random_plate_position

        # Setup observation space properly to match what _compute_observation returns
        # Observation space design:
        #   - "agent_pos": agent (robot) configuration as a single Box
        #   - "environment_state": plate position in the world as a single Box
        #   - "pixels": (optional) dict of camera views if image observations are enabled

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

        # Sample a new plate position
        # Use self._np_random to ensure determinism when seed is provided
        if self._random_plate_position:
            plate_xy = self._np_random.uniform(*_SAMPLING_BOUNDS)
            # CHANGE HERE: "plate" -> "plate_joint"
            self._data.jnt("plate_joint").qpos[:3] = (*plate_xy, self._plate_z)
        else:
            plate_xy = np.asarray([0.5, 0.0])
            # CHANGE HERE: "plate" -> "plate_joint"
            self._data.jnt("plate_joint").qpos[:3] = (*plate_xy, self._plate_z)
            
        # Reset plate orientation to flat (no rotation)
        # CHANGE HERE: "plate" -> "plate_joint"
        self._data.jnt("plate_joint").qpos[3:7] = [1, 0, 0, 0]  # Identity quaternion
        
        # Reset plate velocity
        # CHANGE HERE: "plate" -> "plate_joint"
        self._data.jnt("plate_joint").qvel[:] = 0
        
        mujoco.mj_forward(self._model, self._data)

        # Cache the initial plate height (center of plate)
        self._z_init = self._data.sensor("plate_pos").data[2]
        self._z_success = self._z_init + 0.1  # Success: lift plate by 10cm

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

        # Check if plate is outside bounds
        plate_pos = self._data.sensor("plate_pos").data
        exceeded_bounds = np.any(plate_pos[:2] < (_SAMPLING_BOUNDS[0] - 0.05)) or np.any(
            plate_pos[:2] > (_SAMPLING_BOUNDS[1] + 0.05)
        )

        terminated = bool(success or exceeded_bounds)

        return obs, rew, terminated, False, {"succeed": success}

    def _compute_observation(self) -> dict:
        """Compute the current observation."""
        # Create the dictionary structure that matches our observation space
        observation = {}

        # Get robot state
        robot_state = self.get_robot_state().astype(np.float32)

        # Assemble observation respecting the newly defined observation_space
        plate_pos = self._data.sensor("plate_pos").data.astype(np.float32)

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
                "environment_state": plate_pos,
            }

        return observation

    def _compute_reward(self) -> float:
        """Compute reward based on current state."""
        plate_pos = self._data.sensor("plate_pos").data

        if self.reward_type == "dense":
            tcp_pos = self._data.sensor("2f85/pinch_pos").data
            dist = np.linalg.norm(plate_pos - tcp_pos)
            r_close = np.exp(-20 * dist)
            r_lift = (plate_pos[2] - self._z_init) / (self._z_success - self._z_init)
            r_lift = np.clip(r_lift, 0.0, 1.0)
            return 0.3 * r_close + 0.7 * r_lift
        else:
            # Sparse reward: 1.0 if plate is lifted by 10cm, 0.0 otherwise
            lift = plate_pos[2] - self._z_init
            return float(lift > 0.1)

    def _is_success(self) -> bool:
        """Check if the task is successfully completed.
        
        Success condition:
        - TCP is close to plate (within 5cm)
        - Plate is lifted by at least 10cm from initial height
        - Vacuum gripper is ON (to ensure plate is being held)
        """
        plate_pos = self._data.sensor("plate_pos").data
        tcp_pos = self._data.sensor("2f85/pinch_pos").data
        dist = np.linalg.norm(plate_pos - tcp_pos)
        lift = plate_pos[2] - self._z_init
        
        # Check if vacuum gripper is ON (value > 127, i.e., > 50% of max)
        vacuum_on = self.get_gripper_pose()[0] > 127
        
        return dist < 0.05 and lift > 0.1 and vacuum_on


if __name__ == "__main__":
    from gym_hil import PassiveViewerWrapper

    env = PandaPickPlateGymEnv(render_mode="human")
    env = PassiveViewerWrapper(env)
    obs, info = env.reset()
    
    print("Environment reset successful!")
    print(f"Initial plate position: {obs['environment_state']}")
    print(f"Initial robot state shape: {obs['agent_pos'].shape}")
    
    # Test vacuum gripper control
    print("\nTesting vacuum gripper control...")
    for i in range(50):
        # Move down and turn on vacuum
        if i < 20:
            action = np.array([0, 0, -0.5, 0, 0, 0, 1.0])  # Move down, turn on vacuum
        elif i < 30:
            action = np.array([0, 0, 0, 0, 0, 0, 1.0])  # Keep vacuum on
        else:
            action = np.array([0, 0, 0.5, 0, 0, 0, 1.0])  # Lift up, keep vacuum on
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 10 == 0:
            plate_pos = obs['environment_state']
            gripper_state = env.unwrapped.get_gripper_pose()
            print(f"Step {i}: Plate Z={plate_pos[2]:.3f}, Gripper={gripper_state[0]:.0f}, Reward={reward:.2f}")
        
        if terminated:
            print(f"Episode terminated: {info}")
            break
    
    env.close()

