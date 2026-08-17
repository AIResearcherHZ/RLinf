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
from typing import Any, Optional

import gym
import mujoco
import numpy as np


class SemiTaksT1PickCubeEnv(gym.Env):
    """Semi-Taks-T1 PickCube task using Franka-style operational-space control."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    _RIGHT_ARM_JOINTS = (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
    )

    def __init__(
        self,
        image_obs: bool = False,
        render_mode: Optional[str] = None,
        model_path: Optional[str] = None,
        control_substeps: int = 20,
        position_gain: float = 150.0,
        orientation_gain: float = 25.0,
        damping_ratio: float = 1.0,
        nullspace_gain: float = 10.0,
    ) -> None:
        """Initialize the MuJoCo task."""
        super().__init__()
        repo_root = Path(__file__).resolve().parents[3]
        default_path = repo_root / "assets/Semi_Taks_T1/pickcube.xml"
        self.model_path = Path(model_path) if model_path else default_path
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.image_obs = image_obs
        self.render_mode = render_mode
        self.control_substeps = control_substeps
        self.position_gain = position_gain
        self.orientation_gain = orientation_gain
        self.damping_ratio = damping_ratio
        self.nullspace_gain = nullspace_gain
        self._renderer: Optional[mujoco.Renderer] = None
        self._rng = np.random.default_rng()

        self._arm_joint_ids = np.array(
            [self.model.joint(name).id for name in self._RIGHT_ARM_JOINTS]
        )
        self._arm_qpos_adr = self.model.jnt_qposadr[self._arm_joint_ids]
        self._arm_dof_adr = self.model.jnt_dofadr[self._arm_joint_ids]
        self._arm_actuator_ids = np.array(
            [self.model.actuator(name).id for name in self._RIGHT_ARM_JOINTS]
        )
        self._arm_force_limits = np.abs(
            self.model.actuator_forcerange[self._arm_actuator_ids]
        )[:, 1]
        self._gripper_actuator_id = self.model.actuator("right_gripper").id
        self._gripper_joint_id = self.model.joint("dm_right_gripper_drive_joint").id
        self._gripper_qpos_adr = self.model.jnt_qposadr[self._gripper_joint_id]
        self._ee_site_id = self.model.site("right_ee_site").id
        self._block_joint_id = self.model.joint("block_joint").id
        self._block_qpos_adr = self.model.jnt_qposadr[self._block_joint_id]
        self._block_body_id = self.model.body("block").id
        self._target_box_body_id = self.model.body("target_box").id
        self._tabletop_z = 0.665
        self._mass_matrix = np.zeros((self.model.nv, self.model.nv))
        self._jacobian_position = np.zeros((3, self.model.nv))
        self._jacobian_rotation = np.zeros((3, self.model.nv))
        self._target_position = np.zeros(3)
        self._target_orientation = np.eye(3)
        self._nullspace_qpos = np.zeros(len(self._arm_joint_ids))

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        state_space = gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        if image_obs:
            image_space = gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
            self.observation_space = gym.spaces.Dict(
                {
                    "states": state_space,
                    "images": gym.spaces.Dict(
                        {
                            "left_eye": image_space,
                            "right_eye": image_space,
                            "wrist": image_space,
                        }
                    ),
                }
            )
        else:
            self.observation_space = gym.spaces.Dict({"states": state_space})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset the arm and randomize the cube on the tabletop."""
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        block_qpos = self._block_qpos_adr
        box_x = self._rng.uniform(0.48, 0.60)
        box_y = self._rng.uniform(-0.24, -0.10)
        self.model.body_pos[self._target_box_body_id, :2] = (box_x, box_y)
        self.data.qpos[block_qpos : block_qpos + 3] = np.array(
            [
                self._rng.uniform(0.22, 0.30),
                self._rng.uniform(-0.34, 0.02),
                self._tabletop_z + 0.02,
            ]
        )
        yaw = self._rng.uniform(-np.pi, np.pi)
        self.data.qpos[block_qpos + 3 : block_qpos + 7] = (
            np.cos(yaw / 2),
            0.0,
            0.0,
            np.sin(yaw / 2),
        )
        self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._target_position[:] = self.data.site_xpos[self._ee_site_id]
        self._target_orientation[:] = self.data.site_xmat[self._ee_site_id].reshape(
            3, 3
        )
        self._nullspace_qpos[:] = self.data.qpos[self._arm_qpos_adr]
        self.data.ctrl[self._arm_actuator_ids] = self._operational_space_torques()
        self.data.ctrl[self._gripper_actuator_id] = 0.0
        return self._observation(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply Cartesian translation and gripper commands to the right arm."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._target_position += action[:3] * 0.025
        self._target_position[:] = np.clip(
            self._target_position,
            np.array([0.20, -0.45, 0.62]),
            np.array([0.70, 0.10, 0.95]),
        )
        self.data.ctrl[self._gripper_actuator_id] = 0.9 if action[3] > 0 else 0.0
        for _ in range(self.control_substeps):
            self.data.ctrl[self._arm_actuator_ids] = self._operational_space_torques()
            mujoco.mj_step(self.model, self.data)

        info = self._info()
        ee_pos = self.data.site_xpos[self._ee_site_id]
        block_pos = self.data.xpos[self._block_body_id]
        distance = float(np.linalg.norm(ee_pos - block_pos))
        lift = max(0.0, float(block_pos[2] - (self._tabletop_z + 0.02)))
        reward = float(1.0 - np.tanh(8.0 * distance) + 5.0 * lift)
        if info["success"]:
            reward += 10.0
        return self._observation(), reward, bool(info["success"]), False, info

    def render(self) -> np.ndarray:
        """Render the left eye camera."""
        return self._render_camera("left_eye_camera")

    def close(self) -> None:
        """Release the renderer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _operational_space_torques(self) -> np.ndarray:
        mujoco.mj_jacSite(
            self.model,
            self.data,
            self._jacobian_position,
            self._jacobian_rotation,
            self._ee_site_id,
        )
        jacobian = np.vstack(
            (
                self._jacobian_position[:, self._arm_dof_adr],
                self._jacobian_rotation[:, self._arm_dof_adr],
            )
        )
        velocity = jacobian @ self.data.qvel[self._arm_dof_adr]
        position_error = self._target_position - self.data.site_xpos[self._ee_site_id]
        current_orientation = self.data.site_xmat[self._ee_site_id].reshape(3, 3)
        rotation_error_matrix = self._target_orientation @ current_orientation.T
        orientation_error = 0.5 * np.array(
            [
                rotation_error_matrix[2, 1] - rotation_error_matrix[1, 2],
                rotation_error_matrix[0, 2] - rotation_error_matrix[2, 0],
                rotation_error_matrix[1, 0] - rotation_error_matrix[0, 1],
            ]
        )
        stiffness = np.array([self.position_gain] * 3 + [self.orientation_gain] * 3)
        damping = 2.0 * self.damping_ratio * np.sqrt(stiffness)
        desired_wrench = (
            stiffness * np.concatenate((position_error, orientation_error))
            - damping * velocity
        )

        try:
            mujoco.mj_fullM(self.model, self.data, self._mass_matrix)
        except TypeError:
            mujoco.mj_fullM(self.model, self._mass_matrix, self.data.qM)
        arm_mass = self._mass_matrix[np.ix_(self._arm_dof_adr, self._arm_dof_adr)]
        mass_inverse = np.linalg.inv(arm_mass)
        task_inertia_inverse = jacobian @ mass_inverse @ jacobian.T
        task_inertia = np.linalg.pinv(task_inertia_inverse, rcond=1e-4)
        task_torque = jacobian.T @ task_inertia @ desired_wrench

        dynamically_consistent_inverse = mass_inverse @ jacobian.T @ task_inertia
        nullspace_projector = np.eye(len(self._arm_dof_adr)) - (
            jacobian.T @ dynamically_consistent_inverse.T
        )
        q_error = self._nullspace_qpos - self.data.qpos[self._arm_qpos_adr]
        nullspace_torque = (
            self.nullspace_gain * q_error
            - 2.0 * np.sqrt(self.nullspace_gain) * self.data.qvel[self._arm_dof_adr]
        )
        bias_torque = self.data.qfrc_bias[self._arm_dof_adr]
        torque = task_torque + nullspace_projector @ nullspace_torque + bias_torque
        return np.clip(torque, -self._arm_force_limits, self._arm_force_limits)

    def _observation(self) -> dict[str, Any]:
        ee_pos = self.data.site_xpos[self._ee_site_id].astype(np.float32).copy()
        block_pos = self.data.xpos[self._block_body_id].astype(np.float32).copy()
        gripper = np.array(
            [self.data.qpos[self._gripper_qpos_adr] / 0.9], dtype=np.float32
        )
        observation: dict[str, Any] = {
            "states": np.concatenate((ee_pos, block_pos, gripper))
        }
        if self.image_obs:
            observation["images"] = {
                "left_eye": self._render_camera("left_eye_camera"),
                "right_eye": self._render_camera("right_eye_camera"),
                "wrist": self._render_camera("right_wrist_camera"),
            }
        return observation

    def _render_camera(self, camera: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=128, width=128)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def _info(self) -> dict[str, Any]:
        block_pos = self.data.xpos[self._block_body_id]
        box_pos = self.data.xpos[self._target_box_body_id]
        in_box_xy = bool(np.all(np.abs(block_pos[:2] - box_pos[:2]) < 0.04))
        in_box_z = bool(self._tabletop_z + 0.01 < block_pos[2] < 0.70)
        success = in_box_xy and in_box_z
        return {"success": success, "fail": bool(block_pos[2] < 0.63)}
