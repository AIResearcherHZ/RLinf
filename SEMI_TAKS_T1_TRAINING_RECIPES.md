# Semi-Taks-T1 训练配方适配记录

本文记录 Semi-Taks-T1 在 RLinf 中新增的训练配方、统一数据契约、运行顺序、已解决问题和当前验证边界。

## 完成内容

本次为 `SemiTaksT1PickCubeVision-v0` 补齐以下配方：

| 方法 | 配置 |
| --- | --- |
| RLT Stage 1 | `examples/sft/config/semi_taks_t1_rlt_stage1_sft_openpi_pi05.yaml` |
| RLT Stage 2 AC | `examples/embodiment/config/semi_taks_t1_rlt_stage2_ac_mlp.yaml` |
| RLT Stage 2 TD3 | `examples/embodiment/config/semi_taks_t1_rlt_stage2_td3_mlp.yaml` |
| Sim-Real Co-Training | `examples/embodiment/config/semi_taks_t1_ppo_co_training_openpi_pi05.yaml` |
| OpenPI SFT | `examples/sft/config/semi_taks_t1_sft_openpi_pi05.yaml` |
| Online DAgger | `examples/embodiment/config/semi_taks_t1_dagger_openpi_lerobot.yaml` |
| STEAM | `examples/offline_rl/config/semi_taks_t1_steam_*.yaml` |
| RECAP | `examples/offline_rl/config/semi_taks_t1_recap_*.yaml` 和 `semi_taks_t1_cfg_rl_openpi.yaml` |

环境配置位于：

- `examples/embodiment/config/env/semi_taks_t1_pickcube_openpi.yaml`
- `examples/embodiment/config/env/semi_taks_t1_pickcube_rlt.yaml`

OpenPI 新增了 `pi0_semi_taks_t1` 和 `pi05_semi_taks_t1` 两个数据配置。RECAP 的 value 训练和 advantage 推理也都注册了 `semi_taks_t1`，确保双相机输入不会在推理阶段丢失。

## 数据契约

所有配方统一使用 Online DAgger 写出的 LeRobot 字段：

| 含义 | 字段 | 形状或说明 |
| --- | --- | --- |
| 主相机 | `image` | RGB 图像 |
| 腕部相机 | `wrist_image` | RGB 图像 |
| 本体状态 | `state` | 7 维 |
| 动作 | `actions` | 4 维，训练时按 action horizon 取序列 |
| 任务文本 | `task` | 默认任务为 `pick up the cube` |

环境侧 OpenPI 观测使用 `main_images`、`wrist_images`、`extra_view_images`、`states` 和 `task_descriptions`。RLT 环境额外返回 `rlt_switch_flags`。

不要把 `observation.images.front`、`observation.images.wrist`、`observation.state` 或单数 `action` 当作本批数据的固定 schema。数据来自其他采集器时，应先检查 `meta/info.json` 中的实际 features。

## 环境变量

先配置仓库和数据路径：

```bash
export REPO_PATH=/home/xhz/RLinf
export EMBODIED_PATH=$REPO_PATH/examples/embodiment
export PI05_BASE_PATH=/path/to/pi05_base_pytorch
export T1_LEROBOT_DATA=/path/to/semi_taks_t1_sft_lerobot
export T1_SFT_DATA=/path/to/semi_taks_t1_sft_lerobot
export T1_ROLLOUT_DATA=/path/to/semi_taks_t1_rollout_lerobot
export T1_NORM_STATS=/path/to/norm_stats.json
```

按所选方法再设置 checkpoint：

```bash
export T1_RLT_STAGE1_CKPT=/path/to/rlt_stage1/checkpoints/global_step_N/actor
export T1_SFT_CKPT=/path/to/openpi_sft/checkpoints/global_step_N/actor
export T1_STUDENT_CKPT=/path/to/dagger_student
export T1_EXPERT_CKPT=/path/to/dagger_expert
export T1_REAL_LEROBOT_DATA=/path/to/real_robot_lerobot
export T1_STEAM_VALUE_CKPT=/path/to/steam_value/checkpoints/global_step_N/actor
export T1_RECAP_VALUE_CKPT=/path/to/recap_value/checkpoints/global_step_N/actor
```

`T1_NORM_STATS` 可不设置，配置会回退到 `null`；其余变量只在相应配方中需要。

## 运行顺序

### RLT

```bash
bash examples/sft/run_vla_sft.sh semi_taks_t1_rlt_stage1_sft_openpi_pi05
bash examples/embodiment/run_embodiment.sh semi_taks_t1_rlt_stage2_ac_mlp
# 或使用 TD3
bash examples/embodiment/run_embodiment.sh semi_taks_t1_rlt_stage2_td3_mlp
```

Stage 2 启动前，将 `T1_RLT_STAGE1_CKPT` 指向 Stage 1 的 actor checkpoint。

### OpenPI SFT 与 Sim-Real Co-Training

```bash
bash examples/sft/run_vla_sft.sh semi_taks_t1_sft_openpi_pi05
bash examples/embodiment/run_embodiment.sh semi_taks_t1_ppo_co_training_openpi_pi05
```

Co-Training 需要 `T1_SFT_CKPT` 和 `T1_REAL_LEROBOT_DATA`。

### Online DAgger

```bash
bash examples/embodiment/run_embodiment.sh semi_taks_t1_dagger_openpi_lerobot
```

`T1_STUDENT_CKPT` 和 `T1_EXPERT_CKPT` 都应指向 OpenPI 模型目录。`runner.expert_ckpt_path` 保持 `null`，因为该字段只接受可由 `torch.load()` 读取的单文件权重，不能填模型目录。

### STEAM

```bash
bash examples/offline_rl/advantage_labeling/steam/run_steam_sft.sh \
  semi_taks_t1_steam_value_model_sft

bash examples/offline_rl/advantage_labeling/steam/process/run_compute_advantages_ensemble.sh \
  semi_taks_t1_steam_compute_advantages

bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh \
  semi_taks_t1_steam_cfg_rl_openpi
```

第二步前设置 `T1_STEAM_VALUE_CKPT`。第二步会在 SFT 和 rollout 数据的 `meta` 目录生成带 `semi_taks_t1_steam` tag 的 advantage 文件，第三步使用同一 tag。

### RECAP

```bash
bash examples/offline_rl/advantage_labeling/recap/process/run_compute_returns.sh \
  semi_taks_t1_recap_compute_returns

bash examples/offline_rl/advantage_labeling/recap/run_value_sft.sh \
  semi_taks_t1_recap_value_model_sft

bash examples/offline_rl/advantage_labeling/recap/process/run_compute_advantages.sh \
  semi_taks_t1_recap_compute_advantages

bash examples/offline_rl/policy_optimization/cfg_rl/run_cfg_rl.sh \
  semi_taks_t1_cfg_rl_openpi
```

第三步前设置 `T1_RECAP_VALUE_CKPT`。四步统一使用 `semi_taks_t1_recap` tag。SFT 和 rollout 两类数据都必须完成 returns 与 advantages 处理，否则 CFG 混合数据加载时会缺少 sidecar。

## 踩坑与修复

1. **LeRobot 字段名不能凭示例推断。** 原配置沿用了其他数据集的 `observation.*` 和单数 `action`，与 Online DAgger writer 的真实输出不一致。现已统一为 `image`、`wrist_image`、`state`、`actions`、`task`。
2. **Hydra 主配置不能继承另一个带 `hydra.searchpath` 的主配置。** 这种写法会报 `Overriding hydra.searchpath is only supported from the primary config`。7 个 Semi-Taks-T1 离线配置现均为独立主配置，并直接声明所需 config group。
3. **RECAP 必须同时适配训练和推理。** 只在 `ValueDataset` 注册 robot type 不够，checkpoint 推理和 `compute_advantages.py` 也必须识别相同类型与双相机映射。
4. **RECAP 的 rollout 数据也要计算 advantage。** CFG 同时读取 SFT 和 rollout；只处理 SFT 会在训练阶段找不到 rollout 的 `advantages_<tag>.parquet`。
5. **STEAM/RECAP 的 tag 必须贯穿所有阶段。** 生成 sidecar 和 CFG 读取使用不同 tag 时不会自动回退到正确文件。
6. **DAgger 的模型目录和单文件 checkpoint 是两种接口。** `rollout.expert_model.model_path` 接受模型目录，而 `runner.expert_ckpt_path` 会直接执行 `torch.load()`。
7. **TD3 配置键必须与 worker 实际读取一致。** 已使用 `actor_update_action_noise`、`actor_agg_q` 和 `critic_actor_ratio`，并为 `TwinQCritic` 配置 FSDP wrap。
8. **OpenPI 和 FrankaSim 当前没有组合预构建镜像。** 安装使用 `bash requirements/install.sh embodied --model openpi --env frankasim`，本批配方优先使用本地虚拟环境。

## 验证结果

已完成以下检查：

- 13 个新增 YAML 均可解析，并可由 Hydra 完整组合和解析插值。
- Semi-Taks-T1 RLT 双相机观测契约测试通过。
- 目标单元测试结果为 `5 passed, 1 skipped`。
- 修改的 Python 文件通过 Ruff lint 和 format check。

跳过项是 OpenPI 数据变换契约测试：当前可用的 IsaacLab Python 环境没有安装 `openpi`。配置和相关代码已静态核对，但未在本机执行完整 OpenPI 模型加载、GPU 训练、真实数据解码或端到端收敛验证。开始长训练前，建议先用实际数据各跑一个短 step，确认 checkpoint、norm stats、图像编码和 GPU 显存配置。
