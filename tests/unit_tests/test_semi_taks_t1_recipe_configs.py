# Copyright 2026 The RLinf Authors.
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

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIGS = {
    "examples/embodiment/config": (
        "semi_taks_t1_rlt_stage2_ac_mlp",
        "semi_taks_t1_rlt_stage2_td3_mlp",
        "semi_taks_t1_ppo_co_training_openpi_pi05",
        "semi_taks_t1_dagger_openpi_lerobot",
    ),
    "examples/sft/config": (
        "semi_taks_t1_rlt_stage1_sft_openpi_pi05",
        "semi_taks_t1_sft_openpi_pi05",
    ),
    "examples/offline_rl/config": (
        "semi_taks_t1_steam_value_model_sft",
        "semi_taks_t1_steam_compute_advantages",
        "semi_taks_t1_steam_cfg_rl_openpi",
        "semi_taks_t1_recap_compute_returns",
        "semi_taks_t1_recap_value_model_sft",
        "semi_taks_t1_recap_compute_advantages",
        "semi_taks_t1_cfg_rl_openpi",
    ),
}


def test_recipe_yaml_is_valid() -> None:
    for config_dir, names in CONFIGS.items():
        for name in names:
            path = REPO_ROOT / config_dir / f"{name}.yaml"
            assert yaml.safe_load(path.read_text()) is not None


def test_recipe_hydra_composition(monkeypatch: pytest.MonkeyPatch) -> None:
    hydra = pytest.importorskip("hydra")
    from omegaconf import OmegaConf

    env = {
        "EMBODIED_PATH": str(REPO_ROOT / "examples/embodiment"),
        "REPO_PATH": str(REPO_ROOT),
        "PI05_BASE_PATH": "/tmp/pi05",
        "T1_EXPERT_CKPT": "/tmp/expert",
        "T1_LEROBOT_DATA": "/tmp/lerobot",
        "T1_REAL_LEROBOT_DATA": "/tmp/real_lerobot",
        "T1_RLT_STAGE1_CKPT": "/tmp/rlt_stage1",
        "T1_ROLLOUT_DATA": "/tmp/rollout",
        "T1_RECAP_VALUE_CKPT": "/tmp/recap_value",
        "T1_SFT_CKPT": "/tmp/sft",
        "T1_SFT_DATA": "/tmp/sft_data",
        "T1_STEAM_VALUE_CKPT": "/tmp/steam_value",
        "T1_STUDENT_CKPT": "/tmp/student",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    for config_dir, names in CONFIGS.items():
        with hydra.initialize_config_dir(
            version_base=None,
            config_dir=str(REPO_ROOT / config_dir),
        ):
            for name in names:
                cfg = hydra.compose(config_name=name)
                OmegaConf.to_container(cfg, resolve=True)


def test_openpi_dataset_contract() -> None:
    pytest.importorskip("openpi")
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config
    from rlinf.models.embodiment.value_model.recap.checkpoint_utils import (
        build_input_transforms,
    )

    for name in ("pi0_semi_taks_t1", "pi05_semi_taks_t1"):
        data = get_openpi_config(name).data
        assert data.image_key == "image"
        assert data.wrist_image_key == "wrist_image"
        assert data.state_key == "state"
        assert data.action_key == "actions"
        assert data.task_key == "task"
        assert data.output_action_dim == 4

    transforms = build_input_transforms(
        env_type="semi_taks_t1",
        model_type="pi05",
        action_dim=4,
        default_prompt="Pick up the cube.",
        norm_stats=None,
        use_quantile_norm=True,
    )
    assert type(transforms[1]).__name__ == "ManiSkillInputs"
    assert transforms[1].use_wrist_image is True
