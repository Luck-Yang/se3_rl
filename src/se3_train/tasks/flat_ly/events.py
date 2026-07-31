"""flat_ly reset、startup 与 interval 事件配置学习接口。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def configure_events(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """修改 ``cfg.events``；当前完整沿用 flat 的重置和域随机化事件。"""
