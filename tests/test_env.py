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
import pytest
from gymnasium.utils.env_checker import check_env

import gym_hil  # noqa: F401


@pytest.mark.parametrize(
    "env_task, image_obs",
    [
        ("PandaPickCubeBase-v0", False),
        ("PandaPickCubeBase-v0", True),
        ("PandaPickCubeFtBase-v0", False),
        ("PandaPickCubeFtBase-v0", True),
        ("MasonryBlockInsertionBase-v0", False),
        ("MasonryBlockInsertionBase-v0", True),

    ],
)
def test_hil(env_task, image_obs):
    env = gym.make(f"gym_hil/{env_task}", image_obs=image_obs)
    check_env(env.unwrapped)


def test_panda_pick_ft_env():
    """Test PandaPickCubeGymFtEnv with force/torque sensing."""
    # Test with velocity included
    env = gym.make("gym_hil/PandaPickCubeFtBase-v0", include_velocity=True)
    obs, info = env.reset(seed=42)
    
    # Check observation shape
    assert obs["agent_pos"].shape == (42,), f"Expected shape (42,), got {obs['agent_pos'].shape}"
    assert obs["environment_state"].shape == (3,)
    
    # Check that force/torque data is included (last 24 dimensions)
    ft_data = obs["agent_pos"][-24:]
    assert ft_data.shape == (24,), f"Force/torque data should be 24D, got {ft_data.shape}"
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert obs["agent_pos"].shape == (42,)
    
    env.close()
    
    # Test without velocity
    env = gym.make("gym_hil/PandaPickCubeFtBase-v0", include_velocity=False)
    obs, info = env.reset(seed=42)
    assert obs["agent_pos"].shape == (35,), f"Expected shape (35,), got {obs['agent_pos'].shape}"
    
    # Verify force/torque is still included
    ft_data = obs["agent_pos"][-24:]
    assert ft_data.shape == (24,)
    
    env.close()
