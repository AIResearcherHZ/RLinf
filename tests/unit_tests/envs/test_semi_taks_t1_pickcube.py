# Copyright 2025 The RLinf Authors.
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

import mujoco
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = REPO_ROOT / "assets/Semi_Taks_T1/pickcube.xml"


def test_model_locks_waist_and_left_arm() -> None:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    equality_names = {model.equality(i).name for i in range(model.neq)}
    expected = {
        "lock_waist_yaw",
        "lock_waist_roll",
        "lock_waist_pitch",
        "lock_left_shoulder_pitch",
        "lock_left_shoulder_roll",
        "lock_left_shoulder_yaw",
        "lock_left_elbow",
        "lock_left_wrist_roll",
        "lock_left_wrist_yaw",
        "lock_left_wrist_pitch",
        "lock_left_gripper",
    }
    assert expected <= equality_names
    assert model.site("right_ee_site").id >= 0
    assert model.joint("block_joint").type == mujoco.mjtJoint.mjJNT_FREE
    table_top = model.geom("table").pos[2] + model.geom("table").size[2]
    block_bottom = model.body("block").pos[2] - model.geom("block").size[2]
    assert table_top == pytest.approx(0.665)
    assert block_bottom == pytest.approx(table_top)
    right_arm_actuators = (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
    )
    for actuator_name in right_arm_actuators:
        actuator = model.actuator(actuator_name)
        assert model.actuator_biastype[actuator.id] == mujoco.mjtBias.mjBIAS_NONE
        assert model.actuator_gainprm[actuator.id, 0] == 1.0
    locked_joints = (
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_yaw_joint",
        "left_wrist_pitch_joint",
        "dm_left_gripper_drive_joint",
    )
    for joint_name in locked_joints[:-1]:
        data.ctrl[model.actuator(joint_name).id] = 0.4
    data.ctrl[model.actuator("left_gripper").id] = 0.4
    for _ in range(100):
        mujoco.mj_step(model, data)
    locked_qpos = [data.qpos[model.joint(name).qposadr[0]] for name in locked_joints]
    assert np.max(np.abs(locked_qpos)) < 1e-3


def test_environment_reset_and_step() -> None:
    pytest.importorskip("gym")
    from rlinf.envs.frankasim.semi_taks_t1_pickcube_env import (
        SemiTaksT1PickCubeEnv,
    )

    env = SemiTaksT1PickCubeEnv(image_obs=False, control_substeps=2)
    observation, info = env.reset(seed=7)
    torque = env._operational_space_torques()
    assert np.all(np.abs(torque) <= env._arm_force_limits)
    next_observation, reward, terminated, truncated, next_info = env.step(
        np.zeros(4, dtype=np.float32)
    )
    assert observation["states"].shape == (7,)
    assert next_observation["states"].shape == (7,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)
    assert truncated is False
    assert set(info) == {"success", "fail"}
    assert set(next_info) == {"success", "fail"}
    env.close()


def test_rlt_openpi_observation_contract() -> None:
    pytest.importorskip("gym")
    import torch

    from rlinf.envs.frankasim.frankasim_env import FrankaSimEnv

    env = FrankaSimEnv.__new__(FrankaSimEnv)
    env.obs_mode = "rgb"
    env.state_key = "states"
    env.wrap_obs_mode = "rlt_openpi"
    env.task_prompt = "Pick up the cube."
    env.use_wrist_as_extra_view = True
    env._device = torch.device("cpu")
    front = np.zeros((8, 8, 3), dtype=np.uint8)
    wrist = np.ones((8, 8, 3), dtype=np.uint8)
    env._pick_images = lambda raw_obs: (front, wrist)

    observation = env._wrap_obs({"states": np.arange(7, dtype=np.float32)})

    assert observation["states"].shape == (7,)
    assert observation["main_images"].shape == (8, 8, 3)
    assert observation["wrist_images"].shape == (8, 8, 3)
    assert observation["extra_view_images"].shape == (1, 8, 8, 3)
    assert observation["task_descriptions"] == "Pick up the cube."
    assert observation["rlt_switch_flags"].item() is True
