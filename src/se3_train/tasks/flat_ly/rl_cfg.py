"""flat_ly 学习任务的 PPO/GRU 配置。"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.flat.rl_cfg import rl_cfg as flat_rl_cfg


def rl_cfg(smoke: bool = False) -> RslRlOnPolicyRunnerCfg:
    """沿用 flat 的 PPO/GRU 参数，并使用独立实验目录保存训练结果。"""
    cfg = flat_rl_cfg(smoke=smoke)
    cfg.experiment_name = "se3_wheel_leg_flat_ly"
    return cfg
