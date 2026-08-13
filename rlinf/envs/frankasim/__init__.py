# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import gym

from .frankasim_env import FrankaSimEnv
from .semi_taks_t1_pickcube_env import SemiTaksT1PickCubeEnv

try:
    gym.spec("SemiTaksT1PickCubeVision-v0")
except gym.error.Error:
    gym.register(
        id="SemiTaksT1PickCubeVision-v0",
        entry_point=(
            "rlinf.envs.frankasim.semi_taks_t1_pickcube_env:SemiTaksT1PickCubeEnv"
        ),
        max_episode_steps=100,
    )

try:
    from franka_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv
except ImportError:
    GymRenderingSpec = None
    MujocoGymEnv = None

__all__ = [
    "FrankaSimEnv",
    "GymRenderingSpec",
    "MujocoGymEnv",
    "SemiTaksT1PickCubeEnv",
]
