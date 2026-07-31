# flat_ly 学习任务

`flat_ly` 是从 `flat` 派生的独立学习任务。它保留相同的机器人、平地场景、
actor/critic 观测、6 维动作、速度与机身高度指令、终止条件、课程、事件以及
GRU PPO 参数，只把奖励表清空，留给你自己设计。

## 入口和数据流

训练命令使用 task id `SE3-WheelLegged-Flat-LY-GRU`。注册入口在 `__init__.py`，
环境由 `env_cfg.py` 组装，PPO/GRU 参数在 `rl_cfg.py`。环境每一步的主要数据流是：

```text
commands -> observations -> policy -> actions -> simulator
                                      simulator -> rewards / terminations
```

本目录的配置接口如下：

| 文件 | 负责内容 | 当前状态 |
| --- | --- | --- |
| `observations.py` | actor/critic 能看到什么 | 沿用 flat |
| `actions.py` | policy 输出如何变成关节目标 | 沿用 flat |
| `commands.py` | 速度、高度等任务目标 | 沿用 flat |
| `rewards.py` | 每一步分数及权重 | 空白，等待你设计 |
| `terminations.py` | episode 何时结束 | 沿用 flat |
| `curriculums.py` | 训练难度如何递增 | 沿用 flat |
| `events.py` | reset、域随机化和外部扰动 | 沿用 flat |

`env_cfg.py` 会依次调用这些接口。学习某一部分时，只需修改对应文件；如果要
查看当前继承的完整细节，对照相邻的 `../flat/env_cfg.py` 和 `../flat/rl_cfg.py`。

## 先完成奖励

现在 `rewards.configure_rewards()` 会执行 `cfg.rewards.clear()`，所以任务可以被
导入和构造，但总奖励恒为 0，不会学到有效策略。PPO 还可能因为全零 Advantage
（优势）而输出 `NaN` 损失；这是奖励尚未完成的预期现象，不是可继续使用的
checkpoint。请先在 `rewards.py` 中至少接入一个有效目标奖励，再运行训练 smoke。

每一项奖励都用 `RewardTermCfg` 接到环境。例如下面只说明接口形状，不代表推荐
你使用这个奖励或权重：

```python
from mjlab.managers.reward_manager import RewardTermCfg
from se3_train.mdp import rewards as mdp_rewards


def configure_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    cfg.rewards.clear()
    cfg.rewards["你的奖励名"] = RewardTermCfg(
        func=mdp_rewards.某个奖励函数,
        weight=你设计的权重,
        params={"该函数需要的参数名": 参数值},
    )
```

已有奖励函数和参数可从 `../../mdp/rewards.py` 阅读；也可以直接在本任务的
`rewards.py` 编写新函数。奖励函数应接收环境对象，返回形状为 `(num_envs,)` 的
Tensor。正权重表示奖励，负权重表示惩罚；本项目默认再乘环境步长 `dt`。

建议按以下顺序迭代：先写最小目标奖励，做 1 环境 smoke；再逐项增加安全、姿态
和动作平滑约束；每次只加少量项，并比较每个 `Episode_Reward/<名称>` 指标。

## 奖励完成前的配置检查

这条命令只构造配置，不进行 PPO 参数更新，适合检查各接口是否接通：

```bash
uv run python -c "from se3_train.tasks import flat_ly; cfg = flat_ly.env_cfg(); print(flat_ly.TASK_ID, len(cfg.rewards), flat_ly.rl_cfg(smoke=True).max_iterations)"
```

预期输出包含 `SE3-WheelLegged-Flat-LY-GRU 0 5`。

## 奖励完成后的命令

Windows PowerShell：

```powershell
$env:SE3_SMOKE = "1"
uv run se3-train SE3-WheelLegged-Flat-LY-GRU --env.scene.num-envs 1 --gpu-ids None
Remove-Item Env:SE3_SMOKE
```

Linux / WSL：

```bash
SE3_SMOKE=1 uv run se3-train SE3-WheelLegged-Flat-LY-GRU \
  --env.scene.num-envs 1 --gpu-ids None
```

奖励完成并通过 smoke 后，使用 NVIDIA GPU 正式训练：

```bash
uv run --env-file .env se3-train SE3-WheelLegged-Flat-LY-GRU \
  --env.scene.num-envs 1024
```

打开环境但不加载策略，可先检查机器人、观测和指令管线：

```bash
uv run se3-play SE3-WheelLegged-Flat-LY-GRU \
  --agent zero --viewer viser --num-envs 1
```

训练产物会单独保存到 `logs/rsl_rl/se3_wheel_leg_flat_ly/`，不会和正式 `flat`
任务的 checkpoint 混在一起。
