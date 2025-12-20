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

import gymnasium as gym

from gym_hil.mujoco_gym_env import FrankaGymEnv, GymRenderingSpec, MujocoGymEnv
from gym_hil.wrappers.factory import make_env, wrap_env
from gym_hil.wrappers.viewer_wrapper import PassiveViewerWrapper

__all__ = [
    "MujocoGymEnv",
    "FrankaGymEnv",
    "GymRenderingSpec",
    "PassiveViewerWrapper",
    "make_env",
    "wrap_env",
]

from gymnasium.envs.registration import register

# Register the base environment directly
register(
    id="gym_hil/PandaPickCubeBase-v0",  # This is the base environment
    entry_point="gym_hil.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
)

register(
    id="gym_hil/PandaPickCubeFtBase-v0",  # Force/Torque enabled base environment
    entry_point="gym_hil.envs:PandaPickCubeGymFtEnv",
    max_episode_steps=100,
)

register(
    id="gym_hil/PandaArrangeBoxesBase-v0",  # This is the base environment
    entry_point="gym_hil.envs:PandaArrangeBoxesGymEnv",
    max_episode_steps=100,
)

register(
    id="gym_hil/PandaPickPlateBase-v0",  # Plate pick with vacuum gripper base environment
    entry_point="gym_hil.envs:PandaPickPlateGymEnv",
    max_episode_steps=100,
)

# KUKA Pick Plate Environments
register(
    id="gym_hil/KukaPickPlateBase-v0",  # KUKA iiwa14 plate pick with vacuum gripper base environment
    entry_point="gym_hil.envs:KukaPickPlateGymEnv",
    max_episode_steps=100,
)

# Register the viewer wrapper
register(
    id="gym_hil/PandaPickCubeViewer-v0",
    entry_point=lambda **kwargs: PassiveViewerWrapper(gym.make("gym_hil/PandaPickCubeBase-v0", **kwargs)),
    max_episode_steps=100,
)

register(
    id="gym_hil/PandaArrangeBoxesViewer-v0",
    entry_point=lambda **kwargs: PassiveViewerWrapper(gym.make("gym_hil/PandaArrangeBoxesBase-v0", **kwargs)),
    max_episode_steps=100,
)

# Register wrapped versions using the factory - NOTE: these now use PandaPickCubeBase-v0
register(
    id="gym_hil/PandaPickCube-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/PandaPickCubeBase-v0",  # Use the base environment
        "use_inputs_control": False,
        "gripper_penalty": -0.05,
    },
)

register(
    id="gym_hil/PandaPickCubeGamepad-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/PandaPickCubeBase-v0",  # Use the base environment
        "use_viewer": True,
        "use_inputs_control": True,
        "use_gamepad": True,
    },
)

register(
    id="gym_hil/PandaPickCubeFtGamepad-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/PandaPickCubeFtBase-v0",  # Use the FT base environment
        "use_viewer": True,
        "use_inputs_control": True,
        "use_gamepad": True,
    },
)

register(
    id="gym_hil/PandaPickPlateGamepad-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/PandaPickPlateBase-v0",  # Use the plate pick base environment
        "use_viewer": True,
        "use_inputs_control": True,
        "use_gamepad": True,
        "use_gripper": True,  # Vacuum gripper control
    },
)

register(
    id="gym_hil/PandaPickPlateGamepad6DoF-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=300,  # Increased to allow control_time_s * fps (20s * 10fps = 200, with buffer)
    kwargs={
        "env_id": "gym_hil/PandaPickPlateBase-v0",  # Use the plate pick base environment
        "use_viewer": True,
        "use_inputs_control": True,
        "use_gamepad": True,
        "use_gamepad_6dof": True,  # Enable 6-DoF gamepad control (xyz + rx ry rz)
        "use_gripper": True,  # Vacuum gripper control
        "roll_step_size": 0.1,  # Step size for roll rotation (radians) - ~5.7 degrees for smoother control
        "pitch_step_size": 0.1,  # Step size for pitch rotation (radians) - ~5.7 degrees for smoother control
        "yaw_step_size": 0.1,  # Step size for yaw rotation (radians) - ~5.7 degrees for smoother control
    },
)

# KUKA Pick Plate with Gamepad
register(
    id="gym_hil/KukaPickPlateGamepad-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/KukaPickPlateBase-v0",  # Use the KUKA plate pick base environment
        "use_viewer": True,
        "use_inputs_control": True,
        "use_gamepad": True,
        "use_gripper": True,  # Vacuum gripper control
    },
)

register(
    id="gym_hil/KukaPickPlateGamepad6DoF-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=300,  # Increased to allow control_time_s * fps (20s * 10fps = 200, with buffer)
    kwargs={
        "env_id": "gym_hil/KukaPickPlateBase-v0",  # Use the KUKA plate pick base environment
        "use_viewer": True,
        "use_inputs_control": True,
        "use_gamepad": True,
        "use_gamepad_6dof": True,  # Enable 6-DoF gamepad control (xyz + rx ry rz)
        "use_gripper": True,  # Vacuum gripper control
        "roll_step_size": 0.1,  # Step size for roll rotation (radians) - ~5.7 degrees for smoother control
        "pitch_step_size": 0.1,  # Step size for pitch rotation (radians) - ~5.7 degrees for smoother control
        "yaw_step_size": 0.1,  # Step size for yaw rotation (radians) - ~5.7 degrees for smoother control
    },
)

register(
    id="gym_hil/PandaPickPlateMetaQuest-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=300,  # Increased to allow control_time_s * fps (20s * 10fps = 200, with buffer)
    kwargs={
        "env_id": "gym_hil/PandaPickPlateBase-v0",  # Use the plate pick base environment
        "use_viewer": True,
        "use_inputs_control": True,
        "use_meta_quest": True,
        "use_gripper": True,  # Vacuum gripper control
        "meta_quest_config": {
            "translation_scale": 5,  # Position scaling (reduced from 10 to prevent sudden jumps when normalized values approach 1.0)
            "rotation_scale": 0.1,       # Rotation scaling (similar to spacemouse's 0.08, adjusted for cubic scaling)
            "deadzone": 0.001,          # Deadzone (reduced from 0.1 to 0.001 to match spacemouse's sensitivity)
        },
    },
)

register(
    id="gym_hil/PandaPickCubeKeyboard-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/PandaPickCubeBase-v0",  # Use the base environment
        "use_viewer": True,
        "gripper_penalty": -0.05,
        "use_inputs_control": True,
    },
)

register(
    id="gym_hil/PandaArrangeBoxes-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/PandaArrangeBoxesBase-v0",  # Use the base environment
    },
)

register(
    id="gym_hil/PandaArrangeBoxesGamepad-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/PandaArrangeBoxesBase-v0",  # Use the base environment
        "use_viewer": True,
        "use_inputs_control": True,
        "use_gamepad": True,
    },
)

register(
    id="gym_hil/PandaArrangeBoxesKeyboard-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=100,
    kwargs={
        "env_id": "gym_hil/PandaArrangeBoxesBase-v0",
        "use_viewer": True,
        "gripper_penalty": -0.05,
        "use_inputs_control": True,
    },
)


# ============================================================================
# Masonry Block Insertion Environments
# ============================================================================

register(
    id="gym_hil/MasonryBlockInsertionBase-v0",
    entry_point="gym_hil.envs:MasonryBlockInsertionGymEnv",
    max_episode_steps=250,  # 25秒 @ 10 FPS - 增加到足够完成整个任务
)

register(
    id="gym_hil/MasonryBlockInsertionGamepad-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=200,  # 20秒 @ 10 FPS
    kwargs={
        "env_id": "gym_hil/MasonryBlockInsertionBase-v0",
        "use_viewer": True,
        "use_inputs_control": True,
        "use_gamepad": True,
    },
)

register(
    id="gym_hil/MasonryBlockInsertionKeyboard-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=150,
    kwargs={
        "env_id": "gym_hil/MasonryBlockInsertionBase-v0",
        "use_viewer": True,
        "gripper_penalty": -0.05,
        "use_inputs_control": True,
    },
)

register(
    id="gym_hil/MasonryBlockInsertionMetaQuest-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_hil/MasonryBlockInsertionBase-v0",
        "use_viewer": True,
        "use_inputs_control": True,
        "use_meta_quest": True,
        "meta_quest_config": {
            "translation_scale": 1.0,
            "rotation_scale": 0.3,
            "deadzone": 0.001,
            "right_controller": True,
        },
    },
)

# ============================================================================
# Meta Quest 6-DoF Control Environments
# ============================================================================

register(
    id="gym_hil/PandaPickCubeMetaQuest-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=200,
    kwargs={
        "env_id": "gym_hil/PandaPickCubeBase-v0",
        "use_viewer": True,
        "use_inputs_control": True,
        "use_meta_quest": True,
        "meta_quest_config": {
            "translation_scale": 1.0,  # Position scaling
            "rotation_scale": 0.3,     # Reduced from 1.0 for smoother rotation
            "deadzone": 0.001,
            "right_controller": True,
        },
    },
)

register(
    id="gym_hil/PandaArrangeBoxesMetaQuest-v0",
    entry_point="gym_hil.wrappers.factory:make_env",
    max_episode_steps=500,
    kwargs={
        "env_id": "gym_hil/PandaArrangeBoxesBase-v0",
        "use_viewer": True,
        "use_inputs_control": True,
        "use_meta_quest": True,
        "meta_quest_config": {
            "translation_scale": 2.0,  # Increased from 0.2
            "rotation_scale": 2.0,     # Increased from 0.2
            "deadzone": 0.001,
            "right_controller": True,
        },
    },
)