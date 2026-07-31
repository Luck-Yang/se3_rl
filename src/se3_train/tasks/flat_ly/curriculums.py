"""flat_ly 课程配置学习接口。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def configure_curriculums(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """把第一阶段速度课程限制在直线低速范围。"""
    if play:
        return

    command_vel_cfg = cfg.curriculum["command_vel"]
    command_vel_cfg.params.update(
        {
            "lin_vel_x_step": 0.2,
            "max_lin_vel_x": 2.0,
            "init_lin_vel_x": 0.0,
            "ang_vel_yaw_step": 0.0,
            "max_ang_vel_yaw": 0.0,
            "init_ang_vel_yaw": 0.0,
            "advance_threshold": 0.5,
            "ema_alpha": 0.05,
        }
    )
