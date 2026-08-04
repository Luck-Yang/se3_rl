"""flat_ly 课程配置学习接口。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

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
_YAW_STAGE_INDEX_ATTR = "_flat_ly_yaw_stage_index"
_SPEED_STAGE_INDEX_ATTR = "_flat_ly_speed_stage_index"
_SPEED_EMA_ATTR = "_flat_ly_speed_course_ema"
_SPEED_STANDING_EMA_ATTR = "_flat_ly_speed_standing_ema"
_SPEED_STAGE_START_STEP_ATTR = "_flat_ly_speed_stage_start_step"
_SPEED_POSITIVE_SCORE_ATTR = "_flat_ly_speed_positive_score"
_SPEED_NEGATIVE_SCORE_ATTR = "_flat_ly_speed_negative_score"
_SPEED_STANDING_SCORE_ATTR = "_flat_ly_speed_standing_score"


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


def commands_yaw_staged(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    yaw_stages: tuple[float, ...],
    advance_threshold: float = 0.60,
    ema_alpha: float = 0.05,
    min_stage_iterations: int = 200,
    steps_per_policy_iter: int = 64,
) -> dict[str, torch.Tensor]:
    """正负方向同时达标后，按显式档位逐步扩大高速 yaw-rate。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    cfg: VelocityHeightCommandCfg = term.cfg  # type: ignore[assignment]
    stages = tuple(sorted({abs(float(value)) for value in yaw_stages if float(value) > 0.0}))
    if not stages:
        raise ValueError("yaw_stages 必须包含至少一个正数档位")

    common_step = int(getattr(env, "common_step_counter", 0))
    if not hasattr(env, _YAW_STAGE_INDEX_ATTR):
        setattr(env, _YAW_STAGE_INDEX_ATTR, 0)
        setattr(env, _YAW_EMA_ATTR, 0.0)
        setattr(env, _YAW_STAGE_START_STEP_ATTR, common_step)

    stage_index = min(int(getattr(env, _YAW_STAGE_INDEX_ATTR)), len(stages) - 1)
    yaw_ema = float(getattr(env, _YAW_EMA_ATTR))
    stage_start_step = int(getattr(env, _YAW_STAGE_START_STEP_ATTR))
    min_stage_steps = max(0, int(min_stage_iterations)) * max(1, int(steps_per_policy_iter))
    dwell_progress = min(
        max((common_step - stage_start_step) / max(float(min_stage_steps), 1.0), 0.0),
        1.0,
    )

    log = getattr(env, "extras", {}).get("log", {})
    positive_score = log.get("Locomotion/flat_ly_course_yaw_positive_score")
    negative_score = log.get("Locomotion/flat_ly_course_yaw_negative_score")
    if positive_score is not None and negative_score is not None:
        physical_score = min(float(positive_score), float(negative_score))
        alpha = min(max(float(ema_alpha), 0.0), 1.0)
        yaw_ema = (1.0 - alpha) * yaw_ema + alpha * physical_score
        if (
            yaw_ema > float(advance_threshold)
            and dwell_progress >= 1.0
            and stage_index < len(stages) - 1
        ):
            stage_index += 1
            yaw_ema = 0.0
            stage_start_step = common_step
            dwell_progress = 0.0
        setattr(env, _YAW_STAGE_INDEX_ATTR, stage_index)
        setattr(env, _YAW_EMA_ATTR, yaw_ema)
        setattr(env, _YAW_STAGE_START_STEP_ATTR, stage_start_step)
    else:
        physical_score = float("nan")

    yaw_max = stages[stage_index]
    cfg.ang_vel_yaw_range = (-yaw_max, yaw_max)
    return {
        "ang_vel_yaw_max": torch.tensor(yaw_max, device=env.device),
        "yaw_stage_index": torch.tensor(float(stage_index), device=env.device),
        "yaw_ema": torch.tensor(yaw_ema, device=env.device),
        "yaw_stage_dwell_progress": torch.tensor(dwell_progress, device=env.device),
        "yaw_physical_score": torch.tensor(physical_score, device=env.device),
    }


def commands_speed_staged(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor,
    command_name: str,
    lin_stages: tuple[float, ...],
    advance_threshold: float = 0.55,
    standing_guard_threshold: float = 0.25,
    ema_alpha: float = 0.05,
    min_stage_iterations: int = 250,
    steps_per_policy_iter: int = 64,
) -> dict[str, torch.Tensor]:
    """正向、反向和静站护栏分别达标后，扩大双向线速度范围。"""
    del env_ids
    term = env.command_manager.get_term(command_name)
    cfg: VelocityHeightCommandCfg = term.cfg  # type: ignore[assignment]
    stages = tuple(sorted({abs(float(value)) for value in lin_stages if float(value) > 0.0}))
    if not stages:
        raise ValueError("lin_stages 必须包含至少一个正数档位")

    common_step = int(getattr(env, "common_step_counter", 0))
    if not hasattr(env, _SPEED_STAGE_INDEX_ATTR):
        setattr(env, _SPEED_STAGE_INDEX_ATTR, 0)
        setattr(env, _SPEED_EMA_ATTR, 0.0)
        setattr(env, _SPEED_STANDING_EMA_ATTR, 0.0)
        setattr(env, _SPEED_STAGE_START_STEP_ATTR, common_step)
        setattr(env, _SPEED_POSITIVE_SCORE_ATTR, float("nan"))
        setattr(env, _SPEED_NEGATIVE_SCORE_ATTR, float("nan"))
        setattr(env, _SPEED_STANDING_SCORE_ATTR, float("nan"))

    stage_index = min(int(getattr(env, _SPEED_STAGE_INDEX_ATTR)), len(stages) - 1)
    speed_ema = float(getattr(env, _SPEED_EMA_ATTR))
    standing_ema = float(getattr(env, _SPEED_STANDING_EMA_ATTR))
    stage_start_step = int(getattr(env, _SPEED_STAGE_START_STEP_ATTR))
    min_stage_steps = max(0, int(min_stage_iterations)) * max(1, int(steps_per_policy_iter))
    dwell_progress = min(
        max((common_step - stage_start_step) / max(float(min_stage_steps), 1.0), 0.0),
        1.0,
    )

    log = getattr(env, "extras", {}).get("log", {})
    positive_score = log.get("Locomotion/flat_ly_course_linear_positive_score")
    negative_score = log.get("Locomotion/flat_ly_course_linear_negative_score")
    standing_score = log.get("Locomotion/flat_ly_course_standing_score")
    if positive_score is not None and negative_score is not None and standing_score is not None:
        positive_score = float(positive_score)
        negative_score = float(negative_score)
        standing_score = float(standing_score)
        setattr(env, _SPEED_POSITIVE_SCORE_ATTR, positive_score)
        setattr(env, _SPEED_NEGATIVE_SCORE_ATTR, negative_score)
        setattr(env, _SPEED_STANDING_SCORE_ATTR, standing_score)
        physical_score = min(positive_score, negative_score)
        alpha = min(max(float(ema_alpha), 0.0), 1.0)
        speed_ema = (1.0 - alpha) * speed_ema + alpha * physical_score
        standing_ema = (1.0 - alpha) * standing_ema + alpha * standing_score
        if (
            speed_ema > float(advance_threshold)
            and standing_ema > float(standing_guard_threshold)
            and dwell_progress >= 1.0
            and stage_index < len(stages) - 1
        ):
            stage_index += 1
            speed_ema = 0.0
            standing_ema = 0.0
            stage_start_step = common_step
            dwell_progress = 0.0
        setattr(env, _SPEED_STAGE_INDEX_ATTR, stage_index)
        setattr(env, _SPEED_EMA_ATTR, speed_ema)
        setattr(env, _SPEED_STANDING_EMA_ATTR, standing_ema)
        setattr(env, _SPEED_STAGE_START_STEP_ATTR, stage_start_step)
    else:
        positive_score = float(getattr(env, _SPEED_POSITIVE_SCORE_ATTR))
        negative_score = float(getattr(env, _SPEED_NEGATIVE_SCORE_ATTR))
        standing_score = float(getattr(env, _SPEED_STANDING_SCORE_ATTR))
        physical_score = min(positive_score, negative_score)

    lin_x_max = stages[stage_index]
    cfg.lin_vel_x_range = (-lin_x_max, lin_x_max)
    return {
        "lin_vel_x_max": torch.tensor(lin_x_max, device=env.device),
        "linear_stage_index": torch.tensor(float(stage_index), device=env.device),
        "speed_ema": torch.tensor(speed_ema, device=env.device),
        "standing_guard_ema": torch.tensor(standing_ema, device=env.device),
        "stage_dwell_progress": torch.tensor(dwell_progress, device=env.device),
        "physical_score": torch.tensor(physical_score, device=env.device),
        "linear_positive_score": torch.tensor(
            positive_score,
            device=env.device,
        ),
        "linear_negative_score": torch.tensor(
            negative_score,
            device=env.device,
        ),
        "standing_guard_score": torch.tensor(
            standing_score,
            device=env.device,
        ),
    }


def configure_curriculums(
    cfg: ManagerBasedRlEnvCfg,
    *,
    play: bool,
    phase: Literal["base", "speed", "stand", "turn", "arc"] = "base",
) -> None:
    """使用综合物理稳定分数扩展直线速度课程。"""
    if play:
        return

    command_vel_cfg = cfg.curriculum.get("command_vel")
    if phase == "stand":
        cfg.curriculum.pop("command_vel", None)
    elif phase == "speed" and command_vel_cfg is not None:
        command_vel_cfg.func = commands_speed_staged
        command_vel_cfg.params.clear()
        command_vel_cfg.params.update(
            {
                "command_name": "velocity_height",
                "lin_stages": (0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0),
                "advance_threshold": 0.55,
                "standing_guard_threshold": 0.25,
                "ema_alpha": 0.05,
                "min_stage_iterations": 250,
                "steps_per_policy_iter": 64,
            }
        )
    elif phase in {"turn", "arc"} and command_vel_cfg is not None:
        command_vel_cfg.func = commands_yaw_staged
        command_vel_cfg.params.clear()
        command_vel_cfg.params.update(
            {
                "command_name": "velocity_height",
                "yaw_stages": (0.3, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0),
                "advance_threshold": 0.60,
                "ema_alpha": 0.05,
                "min_stage_iterations": 200 if phase == "turn" else 50,
                "steps_per_policy_iter": 64,
            }
        )
    elif command_vel_cfg is not None:
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
        if phase == "speed":
            push_cfg.params.update(
                {
                    "use_iterations": True,
                    "fixed_iteration": 0,
                    "push_stages": [
                        {
                            "iteration": 0,
                            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                        },
                        {
                            "iteration": 2800,
                            "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
                        },
                    ],
                }
            )
        elif phase == "stand":
            push_cfg.params.update(
                {
                    "use_iterations": True,
                    "fixed_iteration": 0,
                    "push_stages": [
                        {
                            "iteration": 0,
                            "velocity_range": {"x": (-0.10, 0.10), "y": (-0.05, 0.05)},
                        }
                    ],
                }
            )
        elif phase == "turn":
            push_cfg.params.update(
                {
                    "use_iterations": True,
                    "fixed_iteration": 0,
                    "push_stages": [
                        {
                            "iteration": 0,
                            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)},
                        }
                    ],
                }
            )
        elif phase == "arc":
            push_cfg.params.update(
                {
                    "use_iterations": True,
                    "fixed_iteration": 0,
                    "push_stages": [
                        {
                            "iteration": 0,
                            "velocity_range": {"x": (-0.10, 0.10), "y": (-0.10, 0.10)},
                        }
                    ],
                }
            )
        else:
            push_cfg.params["push_stages"] = [
                {"iteration": 0, "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
                {
                    "iteration": 3000,
                    "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                },
                {
                    "iteration": 6000,
                    "velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)},
                },
            ]
