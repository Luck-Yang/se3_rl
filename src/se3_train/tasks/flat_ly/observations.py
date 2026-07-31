"""flat_ly 观测配置学习接口。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def configure_observations(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """修改 ``cfg.observations``；当前完整沿用 flat 的 actor/critic 观测。"""
