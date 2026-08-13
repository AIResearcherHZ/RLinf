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

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


def main() -> None:
    """Open a lightweight MuJoCo viewer for FrankaSim or Semi-Taks-T1."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot", choices=("franka", "t1"), default="t1")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    model_paths = {
        "franka": repo_root / "assets/frankasim_xmls/arena.xml",
        "t1": repo_root / "assets/Semi_Taks_T1/pickcube.xml",
    }
    model = mujoco.MjModel.from_xml_path(str(model_paths[args.robot]))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    paused = False
    single_step = False

    def key_callback(keycode: int) -> None:
        nonlocal paused, single_step
        if keycode == ord(" "):
            paused = not paused
        elif keycode == ord("R"):
            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
        elif keycode == ord("N"):
            single_step = True

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.monotonic()
            if not paused or single_step:
                mujoco.mj_step(model, data)
                single_step = False
            viewer.sync()
            remaining = model.opt.timestep - (time.monotonic() - step_start)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
