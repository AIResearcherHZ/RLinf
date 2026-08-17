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

import numpy as np

from toolkits.lerobot.collect_semi_taks_t1_lerobot import (
    _cartesian_action,
    _VRActionSource,
)


def test_cartesian_action_scales_clips_and_closes_gripper() -> None:
    action = _cartesian_action(
        target_position=np.array([0.525, -0.25, 0.40]),
        control_position=np.array([0.50, -0.20, 0.45]),
        gripper=0.75,
        gripper_threshold=0.5,
    )

    np.testing.assert_allclose(action, np.array([1.0, -1.0, -1.0, 1.0]))
    assert action.dtype == np.float32


def test_cartesian_action_opens_gripper_below_threshold() -> None:
    action = _cartesian_action(
        target_position=np.array([0.50, -0.20, 0.45]),
        control_position=np.array([0.50, -0.20, 0.45]),
        gripper=0.49,
        gripper_threshold=0.5,
    )

    np.testing.assert_array_equal(action, np.array([0.0, 0.0, 0.0, -1.0]))


def test_vr_action_holds_previous_gripper_when_tracking_is_lost() -> None:
    controller = type("Controller", (), {"tracking_enabled": False})()
    source = _VRActionSource.__new__(_VRActionSource)
    source._controller = controller
    source._last_gripper_action = 1.0

    action = source.action()

    np.testing.assert_array_equal(action, np.array([0.0, 0.0, 0.0, 1.0]))
