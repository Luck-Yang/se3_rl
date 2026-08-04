"""flat_ly 学习任务的 PPO/GRU 配置。"""

from __future__ import annotations

import os
from typing import Literal

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.flat.rl_cfg import rl_cfg as flat_rl_cfg


def rl_cfg(
    smoke: bool = False,
    phase: Literal["base", "stand", "turn", "arc"] = "base",
) -> RslRlOnPolicyRunnerCfg:
    """沿用 flat 的 PPO/GRU 参数，并使用独立实验目录保存训练结果。"""
    cfg = flat_rl_cfg(smoke=smoke)
    cfg.experiment_name = "se3_wheel_leg_flat_ly"
    # 所有入口默认随机初始化。只有阶段流水线显式传入 resume/load-run/checkpoint 时才 warm-start。
    cfg.resume = False
    if phase == "stand":
        # 静站属于已有策略的精修，降低更新幅度与熵压力，避免后期重新放大动作抖动。
        cfg.actor.distribution_cfg["init_std"] = 0.20
        cfg.algorithm.learning_rate = 5.0e-5
        cfg.algorithm.entropy_coef = 1.0e-3
        cfg.algorithm.desired_kl = 0.006
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        # 单环境 smoke 不能切成 4 个 mini-batch，否则 RSL-RL 会生成空批次并产生 NaN。
        cfg.algorithm.num_mini_batches = 1
    return cfg
