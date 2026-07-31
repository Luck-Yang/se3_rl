"""flat_ly 课程配置学习接口。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def configure_curriculums(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """修改 ``cfg.curriculum``；当前完整沿用 flat 的指令与扰动课程。"""
