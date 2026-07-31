"""flat_ly 终止条件配置学习接口。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def configure_terminations(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """修改 ``cfg.terminations``；当前完整沿用 flat 的终止条件。"""
