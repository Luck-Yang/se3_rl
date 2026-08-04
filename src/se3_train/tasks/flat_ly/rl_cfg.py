"""flat_ly 学习任务的 PPO/GRU 配置。"""

from __future__ import annotations

import os
from typing import Literal

from mjlab.rl import RslRlOnPolicyRunnerCfg

from se3_train.tasks.flat.rl_cfg import rl_cfg as flat_rl_cfg


def rl_cfg(
    smoke: bool = False,
    phase: Literal["base", "speed", "stand", "turn", "arc"] = "base",
) -> RslRlOnPolicyRunnerCfg:
    """沿用 flat 的 PPO/GRU 参数，并使用独立实验目录保存训练结果。"""
    cfg = flat_rl_cfg(smoke=smoke)
    cfg.experiment_name = "se3_wheel_leg_flat_ly"
    # RSL-RL adaptive schedule 每个 mini-batch 都可能把学习率放大 1.5 倍，默认
    # 上限高达 1e-2。model0 第一轮使用 7x4 次更新时会直接撞到该上限并破坏姿态。
    cfg.algorithm.class_name = "se3_train.tasks.flat_ly.ppo:FlatLyBoundedPPO"
    cfg.algorithm.learning_rate = 1.0e-4
    cfg.algorithm.desired_kl = 0.006
    cfg.algorithm.clip_param = 0.10
    cfg.algorithm.num_learning_epochs = 5
    cfg.algorithm.max_grad_norm = 0.5
    if phase == "base":
        # 第 0 轮从 2e-5 起步，此后跨 iteration 逐步增长，约第 6 轮达到 2e-4 上限。
        cfg.algorithm.class_name = "se3_train.tasks.flat_ly.ppo:FlatLyBasePPO"
    # 所有入口默认随机初始化。只有阶段流水线显式传入 resume/load-run/checkpoint 时才 warm-start。
    cfg.resume = False
    if phase == "speed":
        # 已有低速策略只需要适度恢复探索；过大的 std 会立即破坏姿态和扭矩控制。
        cfg.algorithm.class_name = "se3_train.tasks.flat_ly.ppo:FlatLySpeedPPO"
        cfg.actor.distribution_cfg["init_std"] = 0.15
        cfg.algorithm.learning_rate = 8.0e-5
        cfg.algorithm.entropy_coef = 2.0e-3
        cfg.algorithm.desired_kl = 0.006
    if phase == "stand":
        # 静站属于已有策略的精修，降低更新幅度与熵压力，避免后期重新放大动作抖动。
        cfg.algorithm.class_name = "se3_train.tasks.flat_ly.ppo:FlatLyFineTunePPO"
        cfg.actor.distribution_cfg["init_std"] = 0.20
        cfg.algorithm.learning_rate = 5.0e-5
        cfg.algorithm.entropy_coef = 1.0e-3
        cfg.algorithm.desired_kl = 0.006
    if smoke or os.environ.get("SE3_SMOKE", "0") == "1":
        # 单环境 smoke 不能切成 4 个 mini-batch，否则 RSL-RL 会生成空批次并产生 NaN。
        cfg.algorithm.num_mini_batches = 1
    return cfg
