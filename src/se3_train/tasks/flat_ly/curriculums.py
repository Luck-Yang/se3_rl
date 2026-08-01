"""flat_ly 课程配置学习接口。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.mdp.commands import VelocityHeightCommandCfg

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_LIN_X_MAX_ATTR = "_flat_ly_physical_lin_x_max"
_COURSE_EMA_ATTR = "_flat_ly_physical_course_ema"


def commands_vel_physical(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    lin_vel_x_step: float = 0.1,
    max_lin_vel_x: float = 2.0,
    init_lin_vel_x: float = 0.0,
    advance_threshold: float = 0.65,
    ema_alpha: float = 0.05,
) -> dict[str, torch.Tensor]:
    """只在速度、姿态、轮轴、承载和滚动综合达标后扩大速度范围。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    cfg: VelocityHeightCommandCfg = term.cfg  # type: ignore[assignment]

    if not hasattr(env, _LIN_X_MAX_ATTR):
        setattr(env, _LIN_X_MAX_ATTR, float(init_lin_vel_x))
        setattr(env, _COURSE_EMA_ATTR, 0.0)

    lin_x_max = float(getattr(env, _LIN_X_MAX_ATTR))
    ema = float(getattr(env, _COURSE_EMA_ATTR))
    log = getattr(env, "extras", {}).get("log", {})
    course_score = log.get("Locomotion/flat_ly_course_score", None)

    if course_score is not None:
        alpha = min(max(float(ema_alpha), 0.0), 1.0)
        ema = (1.0 - alpha) * ema + alpha * float(course_score)
        if ema > float(advance_threshold) and lin_x_max < float(max_lin_vel_x):
            lin_x_max = min(lin_x_max + float(lin_vel_x_step), float(max_lin_vel_x))
            # 每个新档位都必须重新证明稳定，避免高 EMA 连续跳档。
            ema = 0.0
        setattr(env, _LIN_X_MAX_ATTR, lin_x_max)
        setattr(env, _COURSE_EMA_ATTR, ema)

    cfg.lin_vel_x_range = (-lin_x_max, lin_x_max)
    cfg.ang_vel_yaw_range = (0.0, 0.0)

    return {
        "lin_vel_x_max": torch.tensor(lin_x_max, device=env.device),
        "ang_vel_yaw_max": torch.tensor(0.0, device=env.device),
        "physical_ema": torch.tensor(ema, device=env.device),
        "physical_score": torch.tensor(
            float(course_score) if course_score is not None else float("nan"),
            device=env.device,
        ),
    }


def configure_curriculums(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """使用综合物理稳定分数扩展直线速度课程。"""
    if play:
        return

    command_vel_cfg = cfg.curriculum["command_vel"]
    command_vel_cfg.func = commands_vel_physical
    command_vel_cfg.params.clear()
    command_vel_cfg.params.update(
        {
            "command_name": "velocity_height",
            "lin_vel_x_step": 0.1,
            "max_lin_vel_x": 2.0,
            "init_lin_vel_x": 0.0,
            "advance_threshold": 0.65,
            "ema_alpha": 0.05,
        }
    )

    push_cfg = cfg.curriculum.get("push_disturbance")
    if push_cfg is not None:
        push_cfg.params["push_stages"] = [
            {"iteration": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
            {"iteration": 3000, "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
            {"iteration": 6000, "velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}},
        ]
