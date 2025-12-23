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

import logging
import sys
import time

import gymnasium as gym
import numpy as np

from gym_hil.mujoco_gym_env import MAX_GRIPPER_COMMAND

DEFAULT_EE_STEP_SIZE = {"x": 0.025, "y": 0.025, "z": 0.025}


class GripperPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, penalty=-0.05):
        super().__init__(env)
        self.penalty = penalty
        self.last_gripper_pos = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_gripper_pos = self.unwrapped.get_gripper_pose() / MAX_GRIPPER_COMMAND
        return obs, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        info["discrete_penalty"] = 0.0
        if (action[-1] < -0.5 and self.last_gripper_pos > 0.9) or (
            action[-1] > 0.5 and self.last_gripper_pos < 0.1
        ):
            info["discrete_penalty"] = self.penalty

        self.last_gripper_pos = self.unwrapped.get_gripper_pose() / MAX_GRIPPER_COMMAND
        return observation, reward, terminated, truncated, info


class EEActionWrapper(gym.ActionWrapper):
    def __init__(self, env, ee_action_step_size, use_gripper=False, use_6dof=False):
        super().__init__(env)
        self.ee_action_step_size = ee_action_step_size
        self.use_gripper = use_gripper
        self.use_6dof = use_6dof

        self._ee_step_size = np.array(
            [
                ee_action_step_size["x"],
                ee_action_step_size["y"],
                ee_action_step_size["z"],
            ]
        )
        num_actions = 3

        # Add rotation dimensions if 6-DoF mode
        if self.use_6dof:
            num_actions = 6  # xyz + rx ry rz

        # Initialize action space bounds for the non-gripper case
        action_space_bounds_min = -np.ones(num_actions)
        action_space_bounds_max = np.ones(num_actions)

        if self.use_gripper:
            action_space_bounds_min = np.concatenate([action_space_bounds_min, [0.0]])
            action_space_bounds_max = np.concatenate([action_space_bounds_max, [2.0]])
            num_actions += 1

        ee_action_space = gym.spaces.Box(
            low=action_space_bounds_min,
            high=action_space_bounds_max,
            shape=(num_actions,),
            dtype=np.float32,
        )
        self.action_space = ee_action_space

    def action(self, action):
        """
        Mujoco env is expecting a 7D action space
        [x, y, z, rx, ry, rz, gripper_open]
        For 3-DoF mode: only control x, y, z, gripper (rotation is zero)
        For 6-DoF mode: control x, y, z, rx, ry, rz, gripper
        """

        if self.use_6dof:
            # 6-DoF mode: action is [x, y, z, rx, ry, rz, (gripper)]
            # Scale position by step_size, rotation is already in radians
            action_xyz = action[:3] * self._ee_step_size
            actions_orn = action[3:6]  # rx, ry, rz already in correct units
        else:
            # 3-DoF mode: action is [x, y, z, (gripper)]
            action_xyz = action[:3] * self._ee_step_size
            actions_orn = np.zeros(3)  # No rotation control

        gripper_open_command = [0.0]
        if self.use_gripper:
            # NOTE: Normalize gripper action from [0, 2] -> [-1, 1]
            gripper_idx = 6 if self.use_6dof else 3
            gripper_open_command = [action[gripper_idx] - 1.0]

        action = np.concatenate([action_xyz, actions_orn, gripper_open_command])
        return action


class InputsControlWrapper(gym.Wrapper):
    """
    Wrapper that allows controlling a gym environment with a gamepad.

    This wrapper intercepts the step method and allows human input via gamepad
    to override the agent's actions when desired.
    """

    def __init__(
        self,
        env,
        x_step_size=1.0,
        y_step_size=1.0,
        z_step_size=1.0,
        use_gripper=False,
        auto_reset=False,
        input_threshold=0.001,
        use_gamepad=True,
        use_meta_quest=False,
        controller_config_path=None,
        meta_quest_config=None,
        use_gamepad_6dof=False,
        roll_step_size=0.01,
        pitch_step_size=0.01,
        yaw_step_size=0.01,
    ):
        """
        Initialize the inputs controller wrapper.

        Args:
            env: The environment to wrap
            x_step_size: Base movement step size for X axis in meters
            y_step_size: Base movement step size for Y axis in meters
            z_step_size: Base movement step size for Z axis in meters
            use_gripper: Whether to use gripper control
            auto_reset: Whether to auto reset the environment when episode ends
            input_threshold: Minimum movement delta to consider as active input
            use_gamepad: Whether to use gamepad or keyboard control
            controller_config_path: Path to the controller configuration JSON file
        """
        super().__init__(env)
        from gym_hil.wrappers.intervention_utils import (
            GamepadController,
            GamepadController6DoF,
            GamepadControllerHID,
            KeyboardController,
            MetaQuestController,
        )

        self.use_meta_quest = use_meta_quest
        self.use_gamepad_6dof = use_gamepad_6dof

        # use HidApi for macos
        if use_meta_quest:
            # Meta Quest 6-DoF control
            meta_config = meta_quest_config or {}
            self.controller = MetaQuestController(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
                translation_scale=meta_config.get("translation_scale", 1.0),
                rotation_scale=meta_config.get("rotation_scale", 0.3),
                deadzone=meta_config.get("deadzone", 0.001),
                right_controller=meta_config.get("right_controller", True),
            )
        elif use_gamepad:
            if self.use_gamepad_6dof:
                # 6-DoF gamepad control (xyz + rx ry rz)
                # use HidApi for macos
                import platform
                if platform.system() == "Darwin":
                    # TODO: Implement GamepadControllerHID6DoF if needed
                    print("Warning: 6-DoF gamepad control on macOS not yet implemented with HID. Falling back to pygame.")
                    self.controller = GamepadController6DoF(
                        x_step_size=x_step_size,
                        y_step_size=y_step_size,
                        z_step_size=z_step_size,
                        roll_step_size=roll_step_size,
                        pitch_step_size=pitch_step_size,
                        yaw_step_size=yaw_step_size,
                        config_path=controller_config_path,
                    )
                else:
                    self.controller = GamepadController6DoF(
                        x_step_size=x_step_size,
                        y_step_size=y_step_size,
                        z_step_size=z_step_size,
                        roll_step_size=roll_step_size,
                        pitch_step_size=pitch_step_size,
                        yaw_step_size=yaw_step_size,
                        config_path=controller_config_path,
                    )
            else:
                # 3-DoF gamepad control (xyz only)
                # use HidApi for macos
                import platform
                if platform.system() == "Darwin":
                    self.controller = GamepadControllerHID(
                        x_step_size=x_step_size,
                        y_step_size=y_step_size,
                        z_step_size=z_step_size,
                    )
                else:
                    self.controller = GamepadController(
                        x_step_size=x_step_size,
                        y_step_size=y_step_size,
                        z_step_size=z_step_size,
                        config_path=controller_config_path,
                    )
        else:
            self.controller = KeyboardController(
                x_step_size=x_step_size,
                y_step_size=y_step_size,
                z_step_size=z_step_size,
            )

        self.auto_reset = auto_reset
        self.use_gripper = use_gripper
        self.input_threshold = input_threshold
        self.controller.start()
        
        # For Meta Quest absolute pose control
        self._meta_quest_absolute_pose = None
        self._meta_quest_gripper_cmd = 1.0
        
        # Track if SUCCESS button was pressed in this episode
        # When SUCCESS is pressed, reward should be 1.0 for all subsequent steps
        self._success_pressed = False

    def get_gamepad_action(self):
        """
        Get the current action from the gamepad/meta quest if any input is active.

        Returns:
            Tuple of (is_active, action, terminate_episode, success)
        """
        # Update the controller to get fresh inputs
        self.controller.update()

        intervention_is_active = self.controller.should_intervene()

        # Get action based on controller type
        if self.use_meta_quest and hasattr(self.controller, 'get_target_pose'):
            # Meta Quest 6-DoF control using absolute pose (similar to real robot)
            target_pose = self.controller.get_target_pose()
            if target_pose is not None:
                # Use absolute pose control for Meta Quest
                # Get gripper command
                gripper_cmd = 1.0  # Default: stay
                if self.use_gripper:
                    gripper_command = self.controller.gripper_command()
                    if gripper_command == "open":
                        gripper_cmd = 2.0
                    elif gripper_command == "close":
                        gripper_cmd = 0.0
                
                # Store target_pose for later use in step() where absolute pose will be applied
                # The actual action passed here is just a placeholder - real control happens in step()
                self._meta_quest_absolute_pose = target_pose
                self._meta_quest_gripper_cmd = gripper_cmd
                # Return a placeholder action (will be converted to delta in step())
                controller_action = np.zeros(7, dtype=np.float32)
                controller_action[6] = gripper_cmd  # Set gripper
                # Debug: print when using absolute pose control
                if not hasattr(self, '_last_abs_pose_print_time'):
                    self._last_abs_pose_print_time = 0.0
                import time
                current_time = time.time()
                if current_time - self._last_abs_pose_print_time > 1.0:  # Print every 1s
                    # Also print current robot pose for comparison
                    current_robot_pos = self.env.unwrapped._data.mocap_pos[0].copy()
                    print(f"[MetaQuest绝对位姿控制] target_pose position: ({target_pose[0,3]:.4f}, {target_pose[1,3]:.4f}, {target_pose[2,3]:.4f})")
                    print(f"[MetaQuest绝对位姿控制] current_robot_pos: ({current_robot_pos[0]:.4f}, {current_robot_pos[1]:.4f}, {current_robot_pos[2]:.4f})")
                    self._last_abs_pose_print_time = current_time
            else:
                # Debug: print when falling back to delta control
                if not hasattr(self, '_last_fallback_print_time'):
                    self._last_fallback_print_time = 0.0
                import time
                current_time = time.time()
                if current_time - self._last_fallback_print_time > 1.0:  # Print every 1s
                    print(f"[MetaQuest警告] get_target_pose()返回None，回退到delta控制")
                    self._last_fallback_print_time = current_time
                # Fallback to delta control if target_pose is not available
                deltas = self.controller.get_6dof_deltas()
                controller_action = np.array([
                    deltas["delta_x"],
                    deltas["delta_y"],
                    deltas["delta_z"],
                    deltas["delta_roll"],
                    deltas["delta_pitch"],
                    deltas["delta_yaw"],
                ], dtype=np.float32)
                
                if self.use_gripper:
                    gripper_command = self.controller.gripper_command()
                    if gripper_command == "open":
                        controller_action = np.concatenate([controller_action, [2.0]])
                    elif gripper_command == "close":
                        controller_action = np.concatenate([controller_action, [0.0]])
                    else:
                        controller_action = np.concatenate([controller_action, [1.0]])
                self._meta_quest_absolute_pose = None
        elif self.use_meta_quest and hasattr(self.controller, 'get_6dof_deltas'):
            # Meta Quest 6-DoF control using deltas (fallback)
            deltas = self.controller.get_6dof_deltas()
            controller_action = np.array([
                deltas["delta_x"],
                deltas["delta_y"],
                deltas["delta_z"],
                deltas["delta_roll"],
                deltas["delta_pitch"],
                deltas["delta_yaw"],
            ], dtype=np.float32)
            
            if self.use_gripper:
                gripper_command = self.controller.gripper_command()
                if gripper_command == "open":
                    controller_action = np.concatenate([controller_action, [2.0]])
                elif gripper_command == "close":
                    controller_action = np.concatenate([controller_action, [0.0]])
                else:
                    controller_action = np.concatenate([controller_action, [1.0]])
            self._meta_quest_absolute_pose = None
        elif self.use_gamepad_6dof and hasattr(self.controller, "get_6dof_deltas"):
            # Gamepad 6-DoF control (xyz + rx ry rz)
            deltas = self.controller.get_6dof_deltas()
            controller_action = np.array([
                deltas["delta_x"],
                deltas["delta_y"],
                deltas["delta_z"],
                deltas["delta_roll"],
                deltas["delta_pitch"],
                deltas["delta_yaw"],
            ], dtype=np.float32)

            if self.use_gripper:
                gripper_command = self.controller.gripper_command()
                if gripper_command == "open":
                    controller_action = np.concatenate([controller_action, [2.0]])
                elif gripper_command == "close":
                    controller_action = np.concatenate([controller_action, [0.0]])
                else:
                    controller_action = np.concatenate([controller_action, [1.0]])
        else:
            # Gamepad/Keyboard 3-DoF control
            delta_x, delta_y, delta_z = self.controller.get_deltas()
            controller_action = np.array([delta_x, delta_y, delta_z], dtype=np.float32)

            if self.use_gripper:
                gripper_command = self.controller.gripper_command()
                if gripper_command == "open":
                    controller_action = np.concatenate([controller_action, [2.0]])
                elif gripper_command == "close":
                    controller_action = np.concatenate([controller_action, [0.0]])
                else:
                    controller_action = np.concatenate([controller_action, [1.0]])

        # Check episode ending buttons
        episode_end_status = self.controller.get_episode_end_status()
        terminate_episode = episode_end_status is not None
        success = episode_end_status == "success"
        rerecord_episode = episode_end_status == "rerecord_episode"

        return (
            intervention_is_active,
            controller_action,
            terminate_episode,
            success,
            rerecord_episode,
        )

    def step(self, action):
        """
        Step the environment, using gamepad input to override actions when active.

        cfg.
            action: Original action from agent

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Get gamepad state and action
        (
            is_intervention,
            gamepad_action,
            terminate_episode,
            success,
            rerecord_episode,
        ) = self.get_gamepad_action()

        # Update episode ending state if requested
        if terminate_episode:
            logging.info(f"Episode manually ended: {'SUCCESS' if success else 'FAILURE'}")

        if is_intervention:
            action = gamepad_action

        # For Meta Quest absolute pose control, apply absolute pose directly
        if (is_intervention and self.use_meta_quest and 
            hasattr(self, '_meta_quest_absolute_pose') and self._meta_quest_absolute_pose is not None):
            # Apply absolute pose directly (similar to real robot control)
            # Get current pose before applying new pose (for delta calculation for data collection)
            from scipy.spatial.transform import Rotation
            current_pos = self.env.unwrapped._data.mocap_pos[0].copy()
            current_quat = self.env.unwrapped._data.mocap_quat[0].copy()  # [w, x, y, z]
            current_rot = Rotation.from_quat([current_quat[1], current_quat[2], current_quat[3], current_quat[0]])
            
            # Calculate delta action for data collection (compatible with QCFQL)
            # This ensures saved actions are in delta format, even though we use absolute pose control
            target_pos = self._meta_quest_absolute_pose[:3, 3]
            target_rot_matrix = self._meta_quest_absolute_pose[:3, :3]
            target_rot = Rotation.from_matrix(target_rot_matrix)
            
            # Calculate position delta
            delta_pos = target_pos - current_pos
            
            # Calculate rotation delta (Euler angles)
            delta_rot = target_rot * current_rot.inv()
            delta_euler = delta_rot.as_euler('xyz', degrees=False)
            
            # Create delta action for data collection (same format as gamepad/delta control)
            delta_action = np.concatenate([delta_pos, delta_euler, [self._meta_quest_gripper_cmd]])
            
            # Apply absolute pose directly (this only sets mocap, does not step physics)
            self.env.unwrapped.apply_absolute_pose(self._meta_quest_absolute_pose, self._meta_quest_gripper_cmd)
            
            # Now step the environment with delta_action to advance physics and get obs/reward/info
            # Pass skip_mocap_update=True to apply_action so it doesn't override the absolute pose we just set
            # The delta_action is used for data collection (saved to info), but mocap stays at absolute pose
            original_apply_action = self.env.unwrapped.apply_action
            self.env.unwrapped.apply_action = lambda action: original_apply_action(action, skip_mocap_update=True)
            try:
                obs, reward, terminated, truncated, info = self.env.step(delta_action)
            finally:
                # Restore original apply_action
                self.env.unwrapped.apply_action = original_apply_action
            
            # Override the action in info with delta_action for data collection
            # This ensures data collection saves delta actions (compatible with QCFQL), not absolute poses
            info["teleop_action"] = delta_action
        else:
            # Normal step (gamepad or Meta Quest delta fallback)
            # Step the environment
            obs, reward, terminated, truncated, info = self.env.step(action)

        # Check terminate_on_success setting from underlying environment
        # Unwrap to find the base environment that has _terminate_on_success attribute
        unwrapped_env = self.env
        while hasattr(unwrapped_env, 'env'):
            unwrapped_env = unwrapped_env.env
        terminate_on_success = getattr(unwrapped_env, '_terminate_on_success', True)
        
        # Add episode ending if requested via gamepad
        # If terminate_on_success is False, don't terminate even if SUCCESS button is pressed
        # (This allows collecting more positive examples for reward classifier training)
        if terminate_episode and terminate_on_success:
            terminated = True
        elif terminate_episode and not terminate_on_success:
            # Don't terminate, but still set reward and log
            logging.info(f"[SUCCESS] SUCCESS button pressed - Episode continues (terminate_on_success=false, reward=1.0)")
        else:
            # Normal termination from environment
            terminated = terminated or truncated

        # Track if SUCCESS button was pressed in this episode
        # When SUCCESS is pressed, reward should be 1.0 for all subsequent steps
        if success:
            self._success_pressed = True
            reward = 1.0
            if terminate_on_success:
                logging.info("Episode ended successfully with reward 1.0")
            else:
                logging.info("[SUCCESS] SUCCESS button pressed - reward=1.0 for this and all subsequent steps")
        
        # If SUCCESS was pressed earlier, keep reward=1.0 for all subsequent steps
        if hasattr(self, '_success_pressed') and self._success_pressed:
            reward = 1.0

        info["is_intervention"] = is_intervention
        action_intervention = action

        info["teleop_action"] = action_intervention
        info["rerecord_episode"] = rerecord_episode

        # If episode ended, reset the state
        if terminated or truncated:
            # Add success/failure information to info dict
            info["next.success"] = success

            # Auto reset if configured
            if self.auto_reset:
                obs, reset_info = self.reset()
                info.update(reset_info)

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset the environment."""
        self.controller.reset()
        # Reset success state when episode resets
        self._success_pressed = False
        obs, info = self.env.reset(**kwargs)
        
        # Initialize last pose for Meta Quest absolute pose control
        if self.use_meta_quest:
            # Get initial robot pose from environment and set it as robot_init_pose
            if hasattr(self.env.unwrapped, '_data'):
                # Get current mocap pose (initial end-effector pose)
                init_pos = self.env.unwrapped._data.mocap_pos[0].copy()  # [x, y, z] in meters
                init_quat = self.env.unwrapped._data.mocap_quat[0].copy()  # [w, x, y, z]
                
                # Convert to 4x4 transformation matrix for Meta Quest
                # Note: Meta_quest2 expects robot_init_ee in mm (for real robot compatibility)
                # But we're working in m, so we convert: m * 1000 = mm
                from scipy.spatial.transform import Rotation
                init_rot = Rotation.from_quat([init_quat[1], init_quat[2], init_quat[3], init_quat[0]])  # [x, y, z, w]
                init_pose_mat = np.eye(4)
                init_pose_mat[:3, 3] = init_pos * 1000.0  # Convert m to mm for Meta_quest2
                init_pose_mat[:3, :3] = init_rot.as_matrix()
                
                # Set robot initial pose in Meta Quest controller
                # This is important for absolute pose calculation
                # Meta_quest2 will convert it back to m in get_target_pose()
                self.controller.set_robot_init_pose(init_pose_mat)
                
                # Debug: print initial pose for verification
                if not hasattr(self, '_init_pose_printed'):
                    print(f"[MetaQuest Reset] Robot init pose (m): pos=({init_pos[0]:.4f}, {init_pos[1]:.4f}, {init_pos[2]:.4f})")
                    print(f"[MetaQuest Reset] Robot init pose (mm for Meta_quest2): pos=({init_pose_mat[0,3]:.2f}, {init_pose_mat[1,3]:.2f}, {init_pose_mat[2,3]:.2f})")
                    self._init_pose_printed = True
                
                # Initialize last pose for delta calculation (for data collection)
                self.env.unwrapped._last_mocap_pos = init_pos.copy()
                self.env.unwrapped._last_mocap_quat = init_quat.copy()
            self._meta_quest_absolute_pose = None
        
        return obs, info

    def close(self):
        """Clean up resources when environment closes."""
        # Stop the controller
        if hasattr(self, "controller"):
            self.controller.stop()

        # Call the parent close method
        return self.env.close()


class ResetDelayWrapper(gym.Wrapper):
    """
    Wrapper that adds a time delay when resetting the environment.

    This can be useful for adding a pause between episodes to allow for human observation.
    """

    def __init__(self, env, delay_seconds=1.0):
        """
        Initialize the time delay reset wrapper.

        Args:
            env: The environment to wrap
            delay_seconds: The number of seconds to delay during reset
        """
        super().__init__(env)
        self.delay_seconds = delay_seconds

    def reset(self, **kwargs):
        """Reset the environment with a time delay."""
        # Add the time delay
        logging.info(f"Reset delay of {self.delay_seconds} seconds")
        time.sleep(self.delay_seconds)

        # Call the parent reset method
        return self.env.reset(**kwargs)
