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
_STAGE_START_STEP_ATTR = "_flat_ly_physical_stage_start_step"
_YAW_MAX_ATTR = "_flat_ly_physical_yaw_max"
_YAW_EMA_ATTR = "_flat_ly_physical_yaw_ema"
_YAW_STAGE_START_STEP_ATTR = "_flat_ly_physical_yaw_stage_start_step"


def commands_vel_physical(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    lin_vel_x_step: float = 0.2,
    max_lin_vel_x: float = 2.0,
    init_lin_vel_x: float = 0.2,
    ang_vel_yaw_step: float = 0.1,
    max_ang_vel_yaw: float = 0.3,
    init_ang_vel_yaw: float = 0.1,
    advance_threshold: float = 0.65,
    yaw_advance_threshold: float = 0.65,
    ema_alpha: float = 0.05,
    min_stage_iterations: int = 100,
    yaw_min_stage_iterations: int = 300,
    steps_per_policy_iter: int = 64,
) -> dict[str, torch.Tensor]:
    """线速度和 yaw 分别证明跟踪能力后，独立扩大各自的采样范围。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    cfg: VelocityHeightCommandCfg = term.cfg  # type: ignore[assignment]

    if not hasattr(env, _LIN_X_MAX_ATTR):
        setattr(env, _LIN_X_MAX_ATTR, float(init_lin_vel_x))
        setattr(env, _COURSE_EMA_ATTR, 0.0)
        setattr(env, _STAGE_START_STEP_ATTR, int(getattr(env, "common_step_counter", 0)))
        setattr(env, _YAW_MAX_ATTR, float(init_ang_vel_yaw))
        setattr(env, _YAW_EMA_ATTR, 0.0)
        setattr(env, _YAW_STAGE_START_STEP_ATTR, int(getattr(env, "common_step_counter", 0)))

    lin_x_max = float(getattr(env, _LIN_X_MAX_ATTR))
    ema = float(getattr(env, _COURSE_EMA_ATTR))
    yaw_max = float(getattr(env, _YAW_MAX_ATTR))
    yaw_ema = float(getattr(env, _YAW_EMA_ATTR))
    common_step = int(getattr(env, "common_step_counter", 0))
    stage_start_step = int(getattr(env, _STAGE_START_STEP_ATTR, common_step))
    yaw_stage_start_step = int(getattr(env, _YAW_STAGE_START_STEP_ATTR, common_step))
    min_stage_steps = max(0, int(min_stage_iterations)) * max(1, int(steps_per_policy_iter))
    yaw_min_stage_steps = max(0, int(yaw_min_stage_iterations)) * max(1, int(steps_per_policy_iter))
    stage_dwell_complete = common_step - stage_start_step >= min_stage_steps
    yaw_stage_dwell_complete = common_step - yaw_stage_start_step >= yaw_min_stage_steps
    log = getattr(env, "extras", {}).get("log", {})
    course_score = log.get("Locomotion/flat_ly_course_score", None)
    yaw_course_score = log.get("Locomotion/flat_ly_course_yaw_tracking_score", None)

    if course_score is not None:
        alpha = min(max(float(ema_alpha), 0.0), 1.0)
        ema = (1.0 - alpha) * ema + alpha * float(course_score)
        if (
            ema > float(advance_threshold)
            and stage_dwell_complete
            and lin_x_max < float(max_lin_vel_x)
        ):
            lin_x_max = min(lin_x_max + float(lin_vel_x_step), float(max_lin_vel_x))
            # 每个新档位都必须重新证明稳定，避免高 EMA 连续跳档。
            ema = 0.0
            stage_start_step = common_step
            stage_dwell_complete = False
        setattr(env, _LIN_X_MAX_ATTR, lin_x_max)
        setattr(env, _COURSE_EMA_ATTR, ema)
        setattr(env, _STAGE_START_STEP_ATTR, stage_start_step)

    if yaw_course_score is not None:
        alpha = min(max(float(ema_alpha), 0.0), 1.0)
        yaw_ema = (1.0 - alpha) * yaw_ema + alpha * float(yaw_course_score)
        if (
            yaw_ema > float(yaw_advance_threshold)
            and yaw_stage_dwell_complete
            and yaw_max < float(max_ang_vel_yaw)
        ):
            yaw_max = min(yaw_max + float(ang_vel_yaw_step), float(max_ang_vel_yaw))
            yaw_ema = 0.0
            yaw_stage_start_step = common_step
        setattr(env, _YAW_MAX_ATTR, yaw_max)
        setattr(env, _YAW_EMA_ATTR, yaw_ema)
        setattr(env, _YAW_STAGE_START_STEP_ATTR, yaw_stage_start_step)

    cfg.lin_vel_x_range = (-lin_x_max, lin_x_max)
    cfg.ang_vel_yaw_range = (-yaw_max, yaw_max)

    return {
        "lin_vel_x_max": torch.tensor(lin_x_max, device=env.device),
        "ang_vel_yaw_max": torch.tensor(yaw_max, device=env.device),
        "physical_ema": torch.tensor(ema, device=env.device),
        "yaw_ema": torch.tensor(yaw_ema, device=env.device),
        "stage_dwell_progress": torch.tensor(
            min(
                max((common_step - stage_start_step) / max(float(min_stage_steps), 1.0), 0.0),
                1.0,
            ),
            device=env.device,
        ),
        "yaw_stage_dwell_progress": torch.tensor(
            min(
                max(
                    (common_step - yaw_stage_start_step) / max(float(yaw_min_stage_steps), 1.0),
                    0.0,
                ),
                1.0,
            ),
            device=env.device,
        ),
        "physical_score": torch.tensor(
            float(course_score) if course_score is not None else float("nan"),
            device=env.device,
        ),
        "yaw_physical_score": torch.tensor(
            float(yaw_course_score) if yaw_course_score is not None else float("nan"),
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
            "lin_vel_x_step": 0.2,
            "max_lin_vel_x": 2.0,
            "init_lin_vel_x": 0.2,
            "ang_vel_yaw_step": 0.1,
            "max_ang_vel_yaw": 0.3,
            "init_ang_vel_yaw": 0.1,
            "advance_threshold": 0.65,
            "yaw_advance_threshold": 0.65,
            "ema_alpha": 0.05,
            "min_stage_iterations": 100,
            "yaw_min_stage_iterations": 300,
            "steps_per_policy_iter": 64,
        }
    )

    push_cfg = cfg.curriculum.get("push_disturbance")
    if push_cfg is not None:
        push_cfg.params["push_stages"] = [
            {"iteration": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
            {"iteration": 3000, "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)}},
            {"iteration": 6000, "velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)}},
        ]
