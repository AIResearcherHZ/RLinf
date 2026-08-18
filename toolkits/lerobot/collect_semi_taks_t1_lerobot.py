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

"""采集 Semi-Taks-T1 将 cube 放入桌面目标框的 LeRobot 数据。"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

os.environ.pop("WAYLAND_DISPLAY", None)
os.environ["XDG_SESSION_TYPE"] = "x11"

import mujoco
import mujoco.viewer
import numpy as np

if TYPE_CHECKING:
    from rlinf.envs.frankasim.semi_taks_t1_pickcube_env import (
        SemiTaksT1PickCubeEnv,
    )


class _PickPlaceStateMachine:
    """基于到位和稳定条件推进的闭环 pick-and-place 状态机。"""

    _MAX_STATE = 9
    _STATE_NAMES = (
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

    def __init__(self) -> None:
        self.state = 0
        self._settled_steps = 0
        self._dwell_steps = 0
        self._lift_target: np.ndarray | None = None
        self._wrist_bid: int | None = None

    @property
    def done(self) -> bool:
        return self.state >= self._MAX_STATE

    def reset(self) -> None:
        self.state = 0
        self._settled_steps = 0
        self._dwell_steps = 0
        self._lift_target = None

    def state_name(self, state: int | None = None) -> str:
        index = self.state if state is None else state
        return self._STATE_NAMES[index] if index < self._MAX_STATE else "done"

    def _wrist_pos(self, env: SemiTaksT1PickCubeEnv) -> np.ndarray:
        if self._wrist_bid is None:
            self._wrist_bid = env.model.body("right_wrist_pitch_link").id
        return env.data.xpos[self._wrist_bid]

    def target(self, env: SemiTaksT1PickCubeEnv) -> tuple[np.ndarray, float]:
        cube = env.data.xpos[env._block_body_id].copy()
        box = env.data.xpos[env._target_box_body_id].copy()
        if self.state == 0:
            return cube + np.array([0.0, 0.0, 0.08]), -1.0
        if self.state == 1:
            return cube + np.array([0.0, 0.0, 0.005]), -1.0
        if self.state in (2, 3, 4):
            if self.state == 4 and self._lift_target is not None:
                return self._lift_target, 1.0
            return cube + np.array([0.0, 0.0, 0.005]), 1.0
        if self.state == 5:
            return box + np.array([0.0, 0.0, 0.13]), 1.0
        if self.state == 6:
            return box + np.array([0.0, 0.0, 0.065]), 1.0
        if self.state == 7:
            return box + np.array([0.0, 0.0, 0.065]), -1.0
        return box + np.array([0.0, 0.0, 0.12]), -1.0

    def advance(self, env: SemiTaksT1PickCubeEnv) -> None:
        if self.done:
            return
        target, gripper = self.target(env)
        wrist = self._wrist_pos(env)
        ee_error = float(np.linalg.norm(target - wrist))
        ee_speed = float(np.linalg.norm(env.data.cvel[self._wrist_bid, 3:]))
        gripper_pos = float(env.data.qpos[env._gripper_qpos_adr])
        gripper_target = 0.9 if gripper > 0 else 0.0
        gripper_done = abs(gripper_pos - gripper_target) < 0.08
        reached = ee_error < 0.02 and ee_speed < 0.15 and gripper_done
        grasped = self._has_grasp_contacts(env) and gripper_pos > 0.2
        state_reached = (
            ee_error < 0.02 and ee_speed < 0.15 and grasped
            if self.state == 2
            else reached
        )
        if state_reached:
            self._settled_steps += 1
        else:
            self._settled_steps = 0
        if self.state == 2 and self._settled_steps >= 3:
            self._log_transition(env, 3)
            self.state = 3
            self._settled_steps = 0
            self._dwell_steps = 8
        elif self.state == 3:
            self._dwell_steps -= 1
            if self._dwell_steps <= 0:
                self._log_transition(env, 4)
                self._lift_target = env.data.xpos[env._block_body_id].copy()
                self._lift_target[2] += 0.08
                self.state = 4
                self._settled_steps = 0
        elif self.state == 7:
            if reached:
                self._dwell_steps -= 1
            if self._dwell_steps <= 0:
                self._log_transition(env, 8)
                self.state = 8
                self._settled_steps = 0
        elif self.state in (0, 1, 4, 5, 6, 8) and self._settled_steps >= 3:
            self._log_transition(env, self.state + 1)
            self.state += 1
            self._settled_steps = 0
            if self.state == 7:
                self._dwell_steps = 8

    def _log_transition(self, env: SemiTaksT1PickCubeEnv, next_state: int) -> None:
        wrist_z = float(self._wrist_pos(env)[2])
        cube_z = float(env.data.xpos[env._block_body_id][2])
        print(
            f"[状态机] {self.state}:{self.state_name()} -> "
            f"{next_state}:{self.state_name(next_state)} "
            f"wrist_z={wrist_z:.3f} cube_z={cube_z:.3f}"
        )

    @staticmethod
    def _has_grasp_contacts(env: SemiTaksT1PickCubeEnv) -> bool:
        block_geom_id = env.model.geom("block").id
        finger_geom_ids = {
            env.model.geom("dm_right_gripper_left_finger_collision").id,
            env.model.geom("dm_right_gripper_right_finger_collision").id,
        }
        contacted_fingers = set()
        for contact_idx in range(env.data.ncon):
            contact = env.data.contact[contact_idx]
            if contact.geom1 == block_geom_id and contact.geom2 in finger_geom_ids:
                contacted_fingers.add(contact.geom2)
            elif contact.geom2 == block_geom_id and contact.geom1 in finger_geom_ids:
                contacted_fingers.add(contact.geom1)
        return contacted_fingers == finger_geom_ids


class _SdkRightArmIK:
    """使用 Taks SDK 的 T1DualArmIK，将末端目标转换为右臂关节目标。"""

    _JOINTS = (
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_yaw_joint",
        "right_wrist_pitch_joint",
    )

    _IK_STEPS_PER_CONTROL = 4

    def __init__(self, sdk_path: Path) -> None:
        sdk_path = sdk_path.expanduser().resolve()
        if str(sdk_path) not in sys.path:
            sys.path.insert(0, str(sdk_path))
        from taks.ik import T1DualArmIK

        self._solver = T1DualArmIK()
        self._qpos_ids = np.array(
            [self._solver.model.jnt_qposadr[self._solver.model.joint(n).id] for n in self._JOINTS],
            dtype=np.intp,
        )

    def reset_from_env(self, env: SemiTaksT1PickCubeEnv) -> None:
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

    def solve(
        self,
        position: np.ndarray,
        quaternion: np.ndarray,
    ) -> np.ndarray:
        position = np.asarray(position, dtype=np.float64)
        quaternion = np.asarray(quaternion, dtype=np.float64)
        for _ in range(self._IK_STEPS_PER_CONTROL):
            self._solver.step(right_pos=position, right_quat=quaternion)
        return self._solver.data.qpos[self._qpos_ids].copy()


def _apply_sdk_joint_target(
    env: SemiTaksT1PickCubeEnv,
    joint_target: np.ndarray,
    gripper: float,
) -> np.ndarray:
    env.data.ctrl[env._arm_actuator_ids] = joint_target
    env.data.ctrl[env._gripper_actuator_id] = 0.9 if gripper > 0 else 0.0
    for _ in range(env.control_substeps):
        env.data.ctrl[env._arm_actuator_ids] = joint_target
        mujoco.mj_step(env.model, env.data)
    return np.r_[joint_target, gripper].astype(np.float32)


_MOCAP_CACHE: dict[int, int] = {}


def _update_mocap_target_marker(
    env: SemiTaksT1PickCubeEnv,
    target_position: np.ndarray | None,
    target_quaternion: np.ndarray | None = None,
) -> None:
    if target_position is None:
        return
    mocap_slot = _MOCAP_CACHE.get(id(env.model))
    if mocap_slot is None:
        mocap_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "right_hand_target"
        )
        if mocap_id < 0:
            _MOCAP_CACHE[id(env.model)] = -1
            return
        mocap_slot = int(env.model.body_mocapid[mocap_id])
        _MOCAP_CACHE[id(env.model)] = mocap_slot
    if mocap_slot >= 0:
        env.data.mocap_pos[mocap_slot] = np.asarray(target_position, dtype=np.float64)
        if target_quaternion is not None:
            env.data.mocap_quat[mocap_slot] = np.asarray(target_quaternion, dtype=np.float64)


def _light_observation(env: SemiTaksT1PickCubeEnv) -> dict[str, Any]:
    ee_pos = env.data.site_xpos[env._ee_site_id].astype(np.float32).copy()
    block_pos = env.data.xpos[env._block_body_id].astype(np.float32).copy()
    gripper = np.array(
        [env.data.qpos[env._gripper_qpos_adr] / 0.9], dtype=np.float32
    )
    return {
        "states": np.concatenate((ee_pos, block_pos, gripper)),
        "images": {
            "left_eye": env._render_camera("left_eye_camera"),
            "wrist": env._render_camera("right_wrist_camera"),
        },
    }


class _VRHandReference:
    def __init__(self, env: SemiTaksT1PickCubeEnv, body_id: int) -> None:
        self._env = env
        self._body_id = body_id

    def mocap_pos(self) -> np.ndarray:
        return self._env.data.xpos[self._body_id].copy()

    def mocap_quat(self) -> np.ndarray:
        return self._env.data.xquat[self._body_id].copy()


class _VRRobotReference:
    def __init__(self, env: SemiTaksT1PickCubeEnv) -> None:
        left_body_id = env.model.body("left_wrist_pitch_link").id
        right_body_id = env.model.body("right_wrist_pitch_link").id
        self.left_hand = _VRHandReference(env, left_body_id)
        self.right_hand = _VRHandReference(env, right_body_id)
        shoulder_id = env.model.joint("right_shoulder_pitch_joint").id
        elbow_id = env.model.joint("right_elbow_joint").id
        shoulder = env.data.xanchor[shoulder_id]
        elbow = env.data.xanchor[elbow_id]
        hand = env.data.xpos[right_body_id]
        self.arm_length = float(
            np.linalg.norm(elbow - shoulder) + np.linalg.norm(hand - elbow)
        )


class _VRActionSource:
    def __init__(
        self,
        env: SemiTaksT1PickCubeEnv,
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
        quaternion = np.asarray(targets.get("right_quat", self._robot.right_hand.mocap_quat()), dtype=np.float64)
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
    from rlinf.envs.frankasim.semi_taks_t1_pickcube_env import (
        SemiTaksT1PickCubeEnv,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="semi_taks_t1_put_cube")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--max-attempts", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=100)
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
    parser.add_argument(
        "--vr-pos-scale",
        type=float,
        default=None,
        help="覆盖自动计算的 VR 位移缩放系数",
    )
    parser.add_argument(
        "--vr-gripper-threshold",
        type=float,
        default=0.5,
        help="右手柄夹爪值切换开合的阈值",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="打开 MuJoCo 可视化窗口",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps 必须大于 0")
    if args.control_mode == "vr" and not 0.0 <= args.vr_gripper_threshold <= 1.0:
        parser.error("--vr-gripper-threshold 必须在 [0, 1] 内")
    if not args.viewer:
        os.environ.setdefault("MUJOCO_GL", "egl")

    env = SemiTaksT1PickCubeEnv(image_obs=True, control_substeps=20)
    viewer = None
    vr_source = None
    sdk_ik = None
    writer = None
    writer_created = False
    step_period = 1.0 / args.fps
    successes = 0
    try:
        if args.viewer:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
        if args.control_mode == "vr":
            vr_source = _VRActionSource(
                env=env,
                sdk_path=args.taks_sdk_path,
                ip=args.vr_ip,
                port=args.vr_port,
                operator_height=args.operator_height,
                pos_scale=args.vr_pos_scale,
                gripper_threshold=args.vr_gripper_threshold,
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
        stop_requested = False
        wrist_bid = env.model.body("right_wrist_pitch_link").id
        for attempt in range(args.max_attempts):
            if successes >= args.num_episodes or stop_requested:
                break
            env.reset(seed=args.seed + attempt)
            obs = _light_observation(env)
            state_machine = _PickPlaceStateMachine()
            sdk_ik.reset_from_env(env)
            _update_mocap_target_marker(
                env, env.data.xpos[wrist_bid], env.data.xquat[wrist_bid]
            )
            if vr_source is None:
                print(
                    f"[状态机] 回合 {attempt + 1}，"
                    f"进入 0:{state_machine.state_name()}"
                )
            if vr_source is not None:
                vr_source.reset()
            frames = []
            for step in range(args.max_steps):
                loop_start = time.monotonic()
                wrist_pos = env.data.xpos[wrist_bid]
                if vr_source is not None:
                    target, quaternion, gripper = vr_source.target_pose()
                    _update_mocap_target_marker(env, target, quaternion)
                    if target is None:
                        joint_target = env.data.qpos[env._arm_qpos_adr].copy()
                        gripper = -1.0
                    else:
                        joint_target = sdk_ik.solve(target, quaternion)
                    action = np.r_[np.clip((target - wrist_pos) / 0.025, -1.0, 1.0) if target is not None else np.zeros(3), gripper].astype(np.float32)
                else:
                    target, gripper = state_machine.target(env)
                    quaternion = env.data.xquat[wrist_bid].copy()
                    _update_mocap_target_marker(env, target, quaternion)
                    joint_target = sdk_ik.solve(target, quaternion)
                    action = np.r_[np.clip((target - wrist_pos) / 0.025, -1.0, 1.0), gripper].astype(np.float32)
                _apply_sdk_joint_target(env, joint_target, gripper)
                info = env._info()
                next_obs, terminated, truncated = _light_observation(env), bool(info["success"]), False
                if vr_source is None:
                    state_machine.advance(env)
                frames.append(
                    {
                        "image": obs["images"]["left_eye"],
                        "wrist_image": obs["images"]["wrist"],
                        "state": obs["states"].astype(np.float32),
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
                if terminated or truncated:
                    break
                if vr_source is None and state_machine.done:
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
