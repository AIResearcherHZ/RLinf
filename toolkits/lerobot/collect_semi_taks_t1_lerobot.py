#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
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

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["XDG_SESSION_TYPE"] = "x11"

import mujoco
import mujoco.viewer
import numpy as np

_VR_POS_SCALE = None
_VR_GRIPPER_THRESHOLD = 0.5


class _PickPlaceStateMachine:
    _STATE_NAMES = (
        "orient_down",
        "approach_above_cube",
        "lower_to_cube",
        "close_gripper",
        "settle_grasp",
        "lift_cube",
        "move_above_box",
        "lower_to_box",
        "release",
        "retreat",
    )

    _TARGETS = {
        1: ("block", 0.08, -1.0),
        2: ("block", 0.0, -1.0),
        3: ("block", 0.0, 1.0),
        4: ("block", 0.0, 1.0),
        6: ("box", 0.08, 1.0),
        7: ("box", 0.03, 1.0),
        8: ("box", 0.03, -1.0),
        9: ("box", 0.12, -1.0),
    }

    _GRIPPER_LENGTH = 0.103
    _DOWN_QUAT = np.array(
        [np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0], dtype=np.float64
    )
    _POS_TOL = 0.03
    _ORIENT_TOL = 0.25
    _HOVER_Z = 0.05

    def __init__(self, env: Any) -> None:
        self._env = env
        self._wrist_bid = env.model.body("right_wrist_pitch_link").id
        self._block_geom = env.model.geom("block").id
        self._finger_geoms = {
            env.model.geom("dm_right_gripper_left_finger_collision").id,
            env.model.geom("dm_right_gripper_right_finger_collision").id,
        }
        self.state = 0
        self._settled = 0
        self._dwell = 0
        self._lift_target: np.ndarray | None = None
        self._home: np.ndarray | None = None

    @property
    def done(self) -> bool:
        return self.state >= len(self._STATE_NAMES)

    def state_name(self, state: int | None = None) -> str:
        index = self.state if state is None else state
        return self._STATE_NAMES[index] if index < len(self._STATE_NAMES) else "done"

    def target(self) -> tuple[np.ndarray, np.ndarray, float]:
        env = self._env
        gl = self._GRIPPER_LENGTH
        if self.state == 0:
            if self._home is None:
                self._home = env.data.xpos[self._wrist_bid].copy()
            return self._home + (0.0, 0.0, self._HOVER_Z), self._DOWN_QUAT, -1.0
        if self.state == 5:
            if self._lift_target is None:
                self._lift_target = env.data.xpos[env._block_body_id] + (0.0, 0.0, 0.08 + gl)
            return self._lift_target, self._DOWN_QUAT, 1.0
        ref, dz, grip = self._TARGETS[self.state]
        base = env.data.xpos[env._block_body_id if ref == "block" else env._target_box_body_id]
        return base + (0.0, 0.0, dz + gl), self._DOWN_QUAT, grip

    def _orient_error(self) -> float:
        quat = self._env.data.xquat[self._wrist_bid]
        dot = float(np.clip(abs(np.dot(quat, self._DOWN_QUAT)), 0.0, 1.0))
        return 2.0 * np.arccos(dot)

    def _grasped(self) -> bool:
        env = self._env
        contacted: set[int] = set()
        for contact in env.data.contact[: env.data.ncon]:
            if contact.geom1 == self._block_geom and contact.geom2 in self._finger_geoms:
                contacted.add(contact.geom2)
            elif contact.geom2 == self._block_geom and contact.geom1 in self._finger_geoms:
                contacted.add(contact.geom1)
        return contacted == self._finger_geoms

    def advance(self) -> None:
        if self.done:
            return
        env = self._env
        target, _quat, gripper = self.target()
        wrist = env.data.xpos[self._wrist_bid]
        pos_ok = float(np.linalg.norm(target - wrist)) < self._POS_TOL
        gripper_pos = float(env.data.qpos[env._gripper_qpos_adr])
        gripper_ok = abs(gripper_pos - (0.9 if gripper > 0 else 0.0)) < 0.08
        orient_ok = self._orient_error() < self._ORIENT_TOL

        reached = pos_ok and orient_ok
        if self.state == 3:
            reached = reached and self._grasped() and gripper_pos > 0.2
        elif self.state != 0:
            reached = reached and gripper_ok

        self._settled = self._settled + 1 if reached else 0

        if self.state == 3 and self._settled >= 3:
            self._transition(4)
            self._dwell = 8
        elif self.state == 4:
            self._dwell -= 1
            if self._dwell <= 0:
                self._transition(5)
        elif self.state == 8:
            if reached:
                self._dwell -= 1
            if self._dwell <= 0:
                self._transition(9)
        elif self._settled >= 3:
            self._transition(self.state + 1)
            if self.state == 8:
                self._dwell = 8

    def _transition(self, next_state: int) -> None:
        env = self._env
        print(
            f"[状态机] {self.state}:{self.state_name()} -> "
            f"{next_state}:{self.state_name(next_state)} "
            f"wrist_z={env.data.xpos[self._wrist_bid][2]:.3f} "
            f"cube_z={env.data.xpos[env._block_body_id][2]:.3f}"
        )
        self.state = next_state
        self._settled = 0


class _SdkRightArmIK:
    _JOINTS = (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
    )

    _STEPS = 10

    def __init__(self, sdk_path: Path) -> None:
        sdk_path = sdk_path.expanduser().resolve()
        if str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))
        from taks.ik import T1DualArmIK

        self._solver = T1DualArmIK()
        self._solver.dt = 0.05
        self._qpos_ids = np.array(
            [self._solver.model.jnt_qposadr[self._solver.model.joint(n).id] for n in self._JOINTS],
            dtype=np.intp,
        )

    def reset_from_env(self, env: Any) -> None:
        for joint_id in range(env.model.njnt):
            name = env.model.joint(joint_id).name
            solver_id = mujoco.mj_name2id(
                self._solver.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if solver_id >= 0:
                self._solver.data.qpos[self._solver.model.jnt_qposadr[solver_id]] = env.data.qpos[
                    env.model.jnt_qposadr[joint_id]
                ]
        self._solver.data.qvel[:] = 0.0
        self._solver.data.ctrl[:] = 0.0
        mujoco.mj_forward(self._solver.model, self._solver.data)
        self._solver.cfg.update(self._solver.data.qpos)
        self._solver.posture_task.set_target_from_configuration(self._solver.cfg)
        wrist_bid = env.model.body("right_wrist_pitch_link").id
        mid = self._solver.mocap_ids["right_hand"]
        self._solver.data.mocap_pos[mid] = env.data.xpos[wrist_bid]
        self._solver.data.mocap_quat[mid] = env.data.xquat[wrist_bid]

    def solve(self, position: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(quaternion, dtype=np.float64)
        for _ in range(self._STEPS):
            self._solver.step(right_pos=position, right_quat=quaternion)
        return self._solver.data.qpos[self._qpos_ids].copy()


def _apply_sdk_joint_target(
    env: Any, joint_target: np.ndarray, gripper: float
) -> np.ndarray:
    env.data.qpos[env._arm_qpos_adr] = joint_target
    env.data.qvel[env._arm_dof_adr] = 0.0
    env.data.ctrl[env._arm_actuator_ids] = joint_target
    env.data.ctrl[env._gripper_actuator_id] = 0.9 if gripper > 0 else 0.0
    for _ in range(env.control_substeps):
        mujoco.mj_step(env.model, env.data)
    return np.r_[joint_target, gripper].astype(np.float32)


def _light_observation(env: Any) -> dict[str, Any]:
    states = np.concatenate(
        (
            env.data.site_xpos[env._ee_site_id].astype(np.float32),
            env.data.xpos[env._block_body_id].astype(np.float32),
            np.array([env.data.qpos[env._gripper_qpos_adr] / 0.9], dtype=np.float32),
        )
    )
    return {
        "states": states,
        "images": {
            "left_eye": env._render_camera("left_eye_camera"),
            "wrist": env._render_camera("right_wrist_camera"),
        },
    }


class _VRHandReference:
    def __init__(self, env: Any, body_id: int) -> None:
        self._env = env
        self._body_id = body_id

    def mocap_pos(self) -> np.ndarray:
        return self._env.data.xpos[self._body_id].copy()

    def mocap_quat(self) -> np.ndarray:
        return self._env.data.xquat[self._body_id].copy()


class _VRRobotReference:
    def __init__(self, env: Any) -> None:
        left_id = env.model.body("left_wrist_pitch_link").id
        right_id = env.model.body("right_wrist_pitch_link").id
        self.left_hand = _VRHandReference(env, left_id)
        self.right_hand = _VRHandReference(env, right_id)
        shoulder = env.data.xanchor[env.model.joint("right_shoulder_pitch_joint").id]
        elbow = env.data.xanchor[env.model.joint("right_elbow_joint").id]
        hand = env.data.xpos[right_id]
        self.arm_length = float(
            np.linalg.norm(elbow - shoulder) + np.linalg.norm(hand - elbow)
        )


class _VRActionSource:
    def __init__(
        self,
        env: Any,
        sdk_path: Path,
        ip: str,
        port: int,
        operator_height: float,
        pos_scale: float | None,
        gripper_threshold: float,
    ) -> None:
        sdk_path = sdk_path.expanduser().resolve()
        if not sdk_path.is_dir():
            raise FileNotFoundError(f"找不到 Taks SDK: {sdk_path}")
        if str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))
        try:
            from taks.vr import VRController
        except ImportError as exc:
            raise RuntimeError(f"无法从 {sdk_path} 导入 taks.vr") from exc

        self._env = env
        self._robot = _VRRobotReference(env)
        self._gripper_threshold = gripper_threshold
        self._last_gripper_action = -1.0
        self._controller: Any = VRController(
            ip=ip,
            port=port,
            operator_height=operator_height,
            pos_scale=pos_scale,
        )

    def start(self) -> None:
        self._controller.start()
        while not self._controller.tracking_enabled:
            time.sleep(0.05)
        print("VR 追踪已连接")

    def reset(self) -> None:
        self._controller.reset_offset()
        self._controller.init_offset(self._robot)
        self._last_gripper_action = -1.0

    def target_pose(self) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        if not self._controller.tracking_enabled:
            return None, None, self._last_gripper_action
        targets = self._controller.step()
        if not targets:
            return None, None, self._last_gripper_action
        quaternion = np.asarray(
            targets.get("right_quat", self._robot.right_hand.mocap_quat()),
            dtype=np.float64,
        )
        if quaternion.shape == (3, 3):
            converted = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(converted, quaternion)
            quaternion = converted
        gripper = 1.0 if self._controller.gripper[1] >= self._gripper_threshold else -1.0
        return np.asarray(targets["right_pos"], dtype=np.float64), quaternion, gripper

    def close(self) -> None:
        self._controller.close()


def main() -> None:
    from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter
    from rlinf.envs.frankasim.semi_taks_t1_pickcube_env import SemiTaksT1PickCubeEnv

    parser = argparse.ArgumentParser(
        description="采集 Semi-Taks-T1 将 cube 放入桌面目标框的 LeRobot 数据。"
    )
    parser.add_argument("--repo-id", default="semi_taks_t1_put_cube")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--control-mode",
        choices=("heuristic", "vr"),
        default="heuristic",
        help="动作来源；vr 使用右手柄控制右臂末端和夹爪",
    )
    parser.add_argument(
        "--taks-sdk-path",
        type=Path,
        default=Path("/home/xhz/taks-controller-web/backend/libs/SDK"),
        help="包含 taks Python 包的 SDK 目录",
    )
    parser.add_argument("--vr-ip", default="0.0.0.0", help="VR UDP 监听地址")
    parser.add_argument("--vr-port", type=int, default=7000, help="VR UDP 监听端口")
    parser.add_argument(
        "--operator-height",
        type=float,
        default=1.75,
        help="操作者身高（米），用于自动缩放手柄位移",
    )
    parser.add_argument("--viewer", action="store_true", help="打开 MuJoCo 可视化窗口")
    args = parser.parse_args()
    if not args.viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")

    env = SemiTaksT1PickCubeEnv(image_obs=True, control_substeps=20)
    viewer = mujoco.viewer.launch_passive(env.model, env.data) if args.viewer else None
    vr_source = None
    sdk_ik = None
    writer = None
    writer_created = False
    successes = 0
    try:
        if args.control_mode == "vr":
            vr_source = _VRActionSource(
                env=env,
                sdk_path=args.taks_sdk_path,
                ip=args.vr_ip,
                port=args.vr_port,
                operator_height=args.operator_height,
                pos_scale=_VR_POS_SCALE,
                gripper_threshold=_VR_GRIPPER_THRESHOLD,
            )
            vr_source.start()
        sdk_ik = _SdkRightArmIK(args.taks_sdk_path)
        writer = LeRobotDatasetWriter()
        from lerobot.utils.constants import HF_LEROBOT_HOME

        dataset_root = HF_LEROBOT_HOME / args.repo_id
        if dataset_root.exists():
            shutil.rmtree(dataset_root)
            print(f"已删除旧数据集: {dataset_root}")
        writer.create(
            repo_id=args.repo_id,
            robot_type="semi_taks_t1",
            fps=args.fps,
            image_shape=(128, 128, 3),
            wrist_image_keys={"wrist_image": (128, 128, 3)},
            state_dim=7,
            action_dim=4,
        )
        writer_created = True

        wrist_bid = env.model.body("right_wrist_pitch_link").id
        mocap_slot = int(env.model.body_mocapid[env.model.body("right_hand_target").id])
        stop_requested = False
        for attempt in range(args.max_attempts):
            if successes >= args.num_episodes or stop_requested:
                break
            env.reset(seed=args.seed + attempt)
            obs = _light_observation(env)
            state_machine = _PickPlaceStateMachine(env)
            env.data.mocap_pos[mocap_slot] = env.data.xpos[wrist_bid]
            env.data.mocap_quat[mocap_slot] = env.data.xquat[wrist_bid]
            if vr_source is None:
                print(f"[状态机] 回合 {attempt + 1}，进入 0:{state_machine.state_name()}")
            else:
                vr_source.reset()
            frames = []
            for _step in range(args.max_steps):
                wrist_pos = env.data.xpos[wrist_bid]
                if vr_source is not None:
                    target, quaternion, gripper = vr_source.target_pose()
                    if target is None:
                        joint_target = env.data.qpos[env._arm_qpos_adr].copy()
                        gripper = -1.0
                        action = np.r_[np.zeros(3), gripper].astype(np.float32)
                    else:
                        joint_target = sdk_ik.solve(target, quaternion)
                        action = np.r_[
                            np.clip((target - wrist_pos) / 0.025, -1.0, 1.0), gripper
                        ].astype(np.float32)
                        env.data.mocap_pos[mocap_slot] = target
                        env.data.mocap_quat[mocap_slot] = quaternion
                else:
                    target, quaternion, gripper = state_machine.target()
                    env.data.mocap_pos[mocap_slot] = target
                    env.data.mocap_quat[mocap_slot] = quaternion
                    joint_target = sdk_ik.solve(target, quaternion)
                    action = np.r_[
                        np.clip((target - wrist_pos) / 0.025, -1.0, 1.0), gripper
                    ].astype(np.float32)
                _apply_sdk_joint_target(env, joint_target, gripper)
                info = env._info()
                next_obs = _light_observation(env)
                if vr_source is None:
                    state_machine.advance()
                frames.append(
                    {
                        "image": obs["images"]["left_eye"],
                        "wrist_image": obs["images"]["wrist"],
                        "state": obs["states"],
                        "actions": action,
                        "task": "Put the cube into the blue box.",
                        "is_success": np.asarray([bool(info["success"])], dtype=bool),
                    }
                )
                obs = next_obs
                if viewer is not None:
                    if not viewer.is_running():
                        stop_requested = True
                        break
                    viewer.sync()
                if bool(info["success"]) or (vr_source is None and state_machine.done):
                    break
            if frames and bool(info["success"]):
                writer.add_episode(frames)
                successes += 1
                print(f"成功回合 {successes}/{args.num_episodes}")
            elif attempt % 10 == 0:
                print(f"尝试 {attempt + 1}/{args.max_attempts}，成功 {successes}")
    finally:
        if writer is not None and writer_created:
            writer.finalize()
        if vr_source is not None:
            vr_source.close()
        if viewer is not None:
            viewer.close()
        env.close()
    if successes < args.num_episodes:
        raise SystemExit(
            f"仅采集到 {successes} 个成功回合，目标为 {args.num_episodes}。"
        )


if __name__ == "__main__":
    main()
