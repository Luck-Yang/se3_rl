"""flat_ly 动作配置学习接口。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def configure_actions(cfg: ManagerBasedRlEnvCfg) -> None:
    """修改 ``cfg.actions``；当前完整沿用 flat 的 6 维延迟动作配置。"""
