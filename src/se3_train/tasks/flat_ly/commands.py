"""flat_ly 指令配置学习接口。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def configure_commands(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """固定水平站立目标，同时保留 flat 的速度与机身高度指令配置。"""
    del play
    command_cfg = cfg.commands["velocity_height"]
    command_cfg.pitch_range = (0.0, 0.0)
    command_cfg.roll_range = (0.0, 0.0)
