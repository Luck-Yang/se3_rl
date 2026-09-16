# flat_ly 平地 GRU 任务

`flat_ly` 是独立的 34D actor 观测、6D 动作平地任务。当前提供一个统一 base 和四个后续阶段：

| 阶段 | task id | 初始化语义 |
| --- | --- | --- |
| base | `SE3-WheelLegged-Flat-LY-GRU` | 随机初始化，默认 `resume=False` |
| speed | `SE3-WheelLegged-Flat-LY-Speed-GRU` | 从低速 base 权重 warm-start，双向逐档扩到 ±2.0 m/s |
| stand | `SE3-WheelLegged-Flat-LY-Stand-GRU` | 从本次 base 权重 warm-start |
| turn | `SE3-WheelLegged-Flat-LY-Turn-GRU` | 推荐从完成的 speed 权重 warm-start，双向逐档学习原地 yaw |
| arc | `SE3-WheelLegged-Flat-LY-Arc-GRU` | 从本次 turn 权重 warm-start |

这里的 warm-start 只加载 actor/critic 权重。optimizer、迭代号和环境计数从 0 开始，且新阶段配置的 `init_std`、学习率会重新应用。它不是继续旧 run 的完整 resume。

## 从低速 base 训练双向中高速

中高速阶段按以下速度上限逐档训练，每一档都同时采样正向和反向：

```text
±0.2 -> ±0.4 -> ±0.6 -> ±0.8 -> ±1.0
-> ±1.2 -> ±1.4 -> ±1.6 -> ±1.8 -> ±2.0 m/s
```

正向、反向的运动物理分数必须分别达标，并且严格静站分数不能低于防遗忘护栏，课程才会晋级。默认每档至少停留 250 轮。用已有低速 checkpoint 启动：

```bash
bash scripts/train_flat_ly_speed.sh logs/rsl_rl/se3_wheel_leg_flat_ly/<低速_run>/model_999.pt
```

脚本默认使用 4096 个环境和 3700 轮。按 7.7 秒一轮估算，纯迭代时间约 7.91 小时。可用环境变量覆盖：

```bash
SE3_FLAT_LY_SPEED_NUM_ENVS=4096 \
SE3_FLAT_LY_SPEED_ITERATIONS=3700 \
bash scripts/train_flat_ly_speed.sh logs/rsl_rl/se3_wheel_leg_flat_ly/<低速_run>/model_999.pt
```

## 从中高速模型继续训练 yaw 转向

完成 speed 阶段后，用它的最终 checkpoint 启动独立 turn 阶段：

```bash
bash scripts/train_flat_ly_yaw.sh \
  logs/rsl_rl/se3_wheel_leg_flat_ly/<speed_run>/model_3699.pt
```

默认课程如下：

```text
±0.3 -> ±0.5 -> ±0.8 -> ±1.2 -> ±1.8
-> ±2.5 -> ±3.5 -> ±4.5 -> ±6.0 rad/s
```

turn 阶段先分离能力：25% 静站、约 56% 原地 yaw、约 19% `±0.3 m/s` 双向低速直行，不在第一档混入 `±2 m/s` 与 yaw 的组合命令。课程只有在以下四个条件同时满足时才晋级：

- 当前 yaw 上限附近的正向和负向分桶都达到阈值；
- 静站稳定分数不低于防遗忘护栏；
- 正向和反向低速直行的最弱分数不低于防遗忘护栏；
- 当前档位至少训练 200 轮。

`6 rad/s` 原地转向时，每侧理论轮速约为 `22 rad/s`；相对 `45 rad/s` 的轮速上限仍保留约一半余量给倒立平衡。默认使用 4096 个环境和 3000 轮，可覆盖：

```bash
SE3_FLAT_LY_YAW_NUM_ENVS=4096 \
SE3_FLAT_LY_YAW_ITERATIONS=3000 \
bash scripts/train_flat_ly_yaw.sh \
  logs/rsl_rl/se3_wheel_leg_flat_ly/<speed_run>/model_3699.pt
```

重点观察 TensorBoard 指标：

- `Curriculum/command_vel/yaw_*`：当前档位、正负边界分数、静站与直行护栏；
- `Locomotion/flat_ly_turn_stable_yaw_reward`：姿态稳定门控后的 yaw 跟踪奖励；
- `Locomotion/flat_ly_turn_translation_speed_positive` 与 `*negative`：正负原地转向时的非期望平移；
- `Locomotion/flat_ly_course_yaw_boundary_positive_score` 与 `*negative_score`：当前边界的方向不对称。

训练任务的自动采样回放会直接开放终档 `±6 rad/s`，并继续受差速轮轮速包络裁剪：

```bash
uv run se3-play SE3-WheelLegged-Flat-LY-Turn-GRU \
  --checkpoint-file logs/rsl_rl/se3_wheel_leg_flat_ly/<yaw_run>/model_<iter>.pt \
  --viewer viser --num-envs 1
```

`se3-play` 会按 turn 配置自动切换静站、低速直行和正负 yaw；它不是固定工况。需要对同一个 yaw 指令做正负 A/B 时，使用 sim2sim 的固定 `--command`，并关闭额外 yaw PID，避免 PID 替策略补偿：

```bash
uv run se3-sim2sim \
  --checkpoint logs/rsl_rl/se3_wheel_leg_flat_ly/<yaw_run>/model_<iter>.pt \
  --viewer rerun --max-steps 3000 --course none --no-yaw-pid \
  --command 0.0 1.0 0.0 0.0 0.22 0.0 0.2 0.0
```

把第二个数字 `1.0` 改为 `-1.0` 即可对照负 yaw。`arc` 的 `se3-play` 同样开放 `±6 rad/s`，但组合命令会按剩余轮速预算自动裁剪。

## 直接从零训练统一 base

下面的命令不会查找或加载旧 checkpoint：

```bash
uv run se3-train SE3-WheelLegged-Flat-LY-GRU \
  --env.scene.num-envs 2048 \
  --agent.resume False \
  --agent.max-iterations 5000 \
  --agent.run-name fresh_base_manual
```

即使省略 `--agent.resume False`，任务配置也默认为随机初始化；命令中保留该参数是为了让训练日志和操作意图清晰。输出保存在 `logs/rsl_rl/se3_wheel_leg_flat_ly/`。

## fresh base + 三阶段流水线

```bash
bash scripts/train_flat_ly_three_stage.sh
```

这个旧流水线先训练新的随机初始化 base，再依次运行 stand、turn 和 arc；它没有插入新建的 speed 阶段。当前已有中高速 checkpoint 时，优先使用上面的 `train_flat_ly_yaw.sh`，避免从 stand 直接进入 yaw。流水线不搜索旧 V8，也不会自动选择历史目录。每次执行会生成唯一 `pipeline_id`，后续阶段只读取本次流水线刚生成且唯一匹配的 checkpoint；若 run-name 已存在、输出不唯一或 checkpoint 缺失，脚本会立即失败。

可通过环境变量调整规模：

```bash
SE3_FLAT_LY_NUM_ENVS=2048 \
SE3_FLAT_LY_BASE_ITERATIONS=5000 \
SE3_FLAT_LY_STAND_ITERATIONS=2000 \
SE3_FLAT_LY_TURN_ITERATIONS=2800 \
SE3_FLAT_LY_ARC_ITERATIONS=1500 \
bash scripts/train_flat_ly_three_stage.sh
```

如需让多次运行具有可辨认的名字，可设置只包含字母、数字、下划线和连字符的 `SE3_FLAT_LY_PIPELINE_ID`。同名输出已存在时脚本拒绝运行，避免把部分失败结果误当成新 checkpoint。

## smoke 检查

Windows PowerShell：

```powershell
$env:SE3_SMOKE = "1"
uv run se3-train SE3-WheelLegged-Flat-LY-GRU --env.scene.num-envs 1 --gpu-ids None --agent.resume False
Remove-Item Env:SE3_SMOKE
```

Linux / WSL：

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-LY-GRU \
  --env.scene.num-envs 1 --gpu-ids None --agent.resume False
```

打开环境但不加载策略：

```bash
uv run se3-play SE3-WheelLegged-Flat-LY-GRU \
  --agent zero --viewer viser --num-envs 1
```
