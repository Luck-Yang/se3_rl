"""flat_ly 指令配置学习接口。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def configure_commands(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """修改 ``cfg.commands``；当前完整沿用 flat 的速度与机身高度指令。"""
