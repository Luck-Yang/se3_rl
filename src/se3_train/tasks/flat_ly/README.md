# flat_ly 平地 GRU 任务

`flat_ly` 是独立的 34D actor 观测、6D 动作平地任务。当前提供一个统一 base 和三个后续阶段：

| 阶段 | task id | 初始化语义 |
| --- | --- | --- |
| base | `SE3-WheelLegged-Flat-LY-GRU` | 随机初始化，默认 `resume=False` |
| stand | `SE3-WheelLegged-Flat-LY-Stand-GRU` | 从本次 base 权重 warm-start |
| turn | `SE3-WheelLegged-Flat-LY-Turn-GRU` | 从本次 stand 权重 warm-start |
| arc | `SE3-WheelLegged-Flat-LY-Arc-GRU` | 从本次 turn 权重 warm-start |

这里的 warm-start 只加载 actor/critic 权重。optimizer、迭代号和环境计数从 0 开始，且新阶段配置的 `init_std`、学习率会重新应用。它不是继续旧 run 的完整 resume。

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

脚本先训练新的随机初始化 base，再依次运行 stand、turn 和 arc。它不再搜索旧 V8，也不会自动选择历史目录。每次执行会生成唯一 `pipeline_id`，后续阶段只读取本次流水线刚生成且唯一匹配的 checkpoint；若 run-name 已存在、输出不唯一或 checkpoint 缺失，脚本会立即失败。

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
