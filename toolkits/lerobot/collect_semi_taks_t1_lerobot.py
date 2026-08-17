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
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import mujoco
import mujoco.viewer
import numpy as np

if TYPE_CHECKING:
    from rlinf.envs.frankasim.semi_taks_t1_pickcube_env import (
        SemiTaksT1PickCubeEnv,
    )


def _heuristic_action(env: SemiTaksT1PickCubeEnv, phase: int) -> np.ndarray:
    ee = env.data.site_xpos[env._ee_site_id]
    cube = env.data.xpos[env._block_body_id]
    box = env.data.xpos[env._target_box_body_id]
    target = cube if phase < 25 else box
    delta = np.clip((target - ee) / 0.025, -1.0, 1.0)
    if phase < 25:
        return np.r_[delta, -1.0].astype(np.float32)
    if phase < 45:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return np.r_[delta, 1.0].astype(np.float32)


def _cartesian_action(
    target_position: np.ndarray,
    control_position: np.ndarray,
    gripper: float,
    gripper_threshold: float,
) -> np.ndarray:
    translation = np.clip(
        (np.asarray(target_position) - np.asarray(control_position)) / 0.025,
        -1.0,
        1.0,
    )
    gripper_action = 1.0 if gripper >= gripper_threshold else -1.0
    return np.r_[translation, gripper_action].astype(np.float32)


class _VRHandReference:
    def __init__(self, env: SemiTaksT1PickCubeEnv, site_id: int) -> None:
        self._env = env
        self._site_id = site_id

    def mocap_pos(self) -> np.ndarray:
        return self._env.data.site_xpos[self._site_id].copy()

    def mocap_quat(self) -> np.ndarray:
        quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            quaternion,
            self._env.data.site_xmat[self._site_id],
        )
        return quaternion


class _VRRobotReference:
    def __init__(self, env: SemiTaksT1PickCubeEnv) -> None:
        left_site_id = mujoco.mj_name2id(
            env.model,
            mujoco.mjtObj.mjOBJ_SITE,
            "left_ee_site",
        )
        if left_site_id < 0:
            left_site_id = env._ee_site_id
        self.left_hand = _VRHandReference(env, left_site_id)
        self.right_hand = _VRHandReference(env, env._ee_site_id)
        shoulder_id = env.model.joint("right_shoulder_pitch_joint").id
        elbow_id = env.model.joint("right_elbow_joint").id
        hand_body_id = env.model.site_bodyid[env._ee_site_id]
        shoulder = env.data.xanchor[shoulder_id]
        elbow = env.data.xanchor[elbow_id]
        hand = env.data.xpos[hand_body_id]
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

    def start(self, timeout: float) -> None:
        self._controller.start()
        deadline = time.monotonic() + timeout
        while not self._controller.tracking_enabled:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"等待 VR 追踪超时（{timeout:.1f} 秒）")
            time.sleep(0.05)
        print("VR 追踪已连接")

    def reset(self) -> None:
        self._controller.reset_offset()
        self._controller.init_offset(self._robot)
        self._last_gripper_action = -1.0

    def action(self) -> np.ndarray:
        if not self._controller.tracking_enabled:
            return np.r_[np.zeros(3), self._last_gripper_action].astype(np.float32)
        targets = self._controller.step()
        if not targets:
            return np.r_[np.zeros(3), self._last_gripper_action].astype(np.float32)
        action = _cartesian_action(
            targets["right_pos"],
            self._env._target_position,
            self._controller.gripper[1],
            self._gripper_threshold,
        )
        self._last_gripper_action = float(action[3])
        return action

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
        "--vr-timeout",
        type=float,
        default=30.0,
        help="启动时等待 VR 追踪的最长秒数",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="打开 MuJoCo 交互窗口实时显示采集过程（不要设 MUJOCO_GL=egl）",
    )
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps 必须大于 0")
    if args.control_mode == "vr" and not 0.0 <= args.vr_gripper_threshold <= 1.0:
        parser.error("--vr-gripper-threshold 必须在 [0, 1] 内")
    if args.control_mode == "vr" and args.vr_timeout <= 0.0:
        parser.error("--vr-timeout 必须大于 0")

    env = SemiTaksT1PickCubeEnv(image_obs=True, control_substeps=20)
    viewer = None
    vr_source = None
    writer = None
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
            vr_source.start(args.vr_timeout)
        writer = LeRobotDatasetWriter()
        writer.create(
            repo_id=args.repo_id,
            robot_type="semi_taks_t1",
            fps=args.fps,
            image_shape=(128, 128, 3),
            wrist_image_keys={"wrist_image": (128, 128, 3)},
            state_dim=7,
            action_dim=4,
        )
        stop_requested = False
        for attempt in range(args.max_attempts):
            if successes >= args.num_episodes or stop_requested:
                break
            obs, _ = env.reset(seed=args.seed + attempt)
            if vr_source is not None:
                vr_source.reset()
            frames = []
            for step in range(args.max_steps):
                loop_start = time.monotonic()
                action = (
                    vr_source.action()
                    if vr_source is not None
                    else _heuristic_action(env, step)
                )
                next_obs, _, terminated, truncated, info = env.step(action)
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
                if vr_source is not None or viewer is not None:
                    time.sleep(max(0.0, step_period - (time.monotonic() - loop_start)))
                if terminated or truncated:
                    break
            if frames and bool(info["success"]):
                writer.add_episode(frames)
                successes += 1
                print(f"成功回合 {successes}/{args.num_episodes}")
            elif attempt % 10 == 0:
                print(f"尝试 {attempt + 1}/{args.max_attempts}，成功 {successes}")
    finally:
        if writer is not None:
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
