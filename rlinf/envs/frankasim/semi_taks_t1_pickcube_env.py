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
    """Semi-Taks-T1 right-arm PickCube task with a fixed waist and left arm."""

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
        self._gripper_actuator_id = self.model.actuator("right_gripper").id
        self._gripper_joint_id = self.model.joint("dm_right_gripper_drive_joint").id
        self._gripper_qpos_adr = self.model.jnt_qposadr[self._gripper_joint_id]
        self._ee_site_id = self.model.site("right_ee_site").id
        self._block_joint_id = self.model.joint("block_joint").id
        self._block_qpos_adr = self.model.jnt_qposadr[self._block_joint_id]
        self._block_body_id = self.model.body("block").id

        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(4,), dtype=np.float32)
        state_space = gym.spaces.Box(-np.inf, np.inf, shape=(7,), dtype=np.float32)
        if image_obs:
            image_space = gym.spaces.Box(0, 255, shape=(128, 128, 3), dtype=np.uint8)
            self.observation_space = gym.spaces.Dict(
                {
                    "states": state_space,
                    "images": gym.spaces.Dict(
                        {"front": image_space, "wrist": image_space}
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
        self.data.qpos[block_qpos : block_qpos + 3] = np.array(
            [
                self._rng.uniform(0.43, 0.50),
                self._rng.uniform(-0.24, -0.10),
                0.535,
            ]
        )
        self.data.qpos[block_qpos + 3 : block_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
        self.data.ctrl[:] = 0.0
        self.data.ctrl[self._arm_actuator_ids] = self.data.qpos[self._arm_qpos_adr]
        self.data.ctrl[self._gripper_actuator_id] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._observation(), self._info()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply Cartesian translation and gripper commands to the right arm."""
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target_position = self.data.site_xpos[self._ee_site_id] + action[:3] * 0.025
        q_target = self._solve_position_ik(target_position)
        self.data.ctrl[self._arm_actuator_ids] = q_target
        self.data.ctrl[self._gripper_actuator_id] = 0.9 if action[3] > 0 else 0.0
        for _ in range(self.control_substeps):
            mujoco.mj_step(self.model, self.data)

        info = self._info()
        ee_pos = self.data.site_xpos[self._ee_site_id]
        block_pos = self.data.xpos[self._block_body_id]
        distance = float(np.linalg.norm(ee_pos - block_pos))
        lift = max(0.0, float(block_pos[2] - 0.535))
        reward = float(1.0 - np.tanh(8.0 * distance) + 5.0 * lift)
        if info["success"]:
            reward += 10.0
        return self._observation(), reward, bool(info["success"]), False, info

    def render(self) -> np.ndarray:
        """Render the front camera."""
        return self._render_camera("front")

    def close(self) -> None:
        """Release the renderer."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _solve_position_ik(self, target_position: np.ndarray) -> np.ndarray:
        q = self.data.qpos[self._arm_qpos_adr].copy()
        work_data = mujoco.MjData(self.model)
        work_data.qpos[:] = self.data.qpos
        for _ in range(12):
            work_data.qpos[self._arm_qpos_adr] = q
            mujoco.mj_forward(self.model, work_data)
            error = target_position - work_data.site_xpos[self._ee_site_id]
            if np.linalg.norm(error) < 1e-4:
                break
            jacobian = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, work_data, jacobian, None, self._ee_site_id)
            arm_jacobian = jacobian[:, self._arm_dof_adr]
            damping = 1e-4 * np.eye(3)
            delta = arm_jacobian.T @ np.linalg.solve(
                arm_jacobian @ arm_jacobian.T + damping, error
            )
            q += np.clip(delta, -0.08, 0.08)
            q = np.clip(
                q,
                self.model.jnt_range[self._arm_joint_ids, 0],
                self.model.jnt_range[self._arm_joint_ids, 1],
            )
        return q

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
                "front": self._render_camera("front"),
                "wrist": self._render_camera("right_wrist_camera"),
            }
        return observation

    def _render_camera(self, camera: str) -> np.ndarray:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=128, width=128)
        self._renderer.update_scene(self.data, camera=camera)
        return self._renderer.render().copy()

    def _info(self) -> dict[str, Any]:
        block_height = float(self.data.xpos[self._block_body_id, 2])
        return {"success": block_height > 0.62, "fail": block_height < 0.48}
