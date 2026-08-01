"""flat_ly 指令配置学习接口。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def configure_commands(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """固定稳态水平目标，训练时对速度指令施加加速度限幅。"""
    command_cfg = cfg.commands["velocity_height"]
    command_cfg.pitch_range = (0.0, 0.0)
    command_cfg.roll_range = (0.0, 0.0)
    command_cfg.lin_vel_slew_rate = None if play else 0.8
