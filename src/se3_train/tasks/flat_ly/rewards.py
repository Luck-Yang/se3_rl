"""flat_ly 奖励设计入口；具体奖励由学习者完成。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def configure_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    """清空继承自 flat 的奖励；在此函数中逐项完成自己的奖励设计。"""
    cfg.rewards.clear()
