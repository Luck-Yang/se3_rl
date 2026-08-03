"""用于学习完整训练流程的 flat_ly 平地 GRU 任务。"""

from __future__ import annotations

from mjlab.tasks.registry import register_mjlab_task

from se3_train.tasks.common import Se3ProfiledOnPolicyRunner

from .env_cfg import env_cfg
from .rl_cfg import rl_cfg

TASK_ID = "SE3-WheelLegged-Flat-LY-GRU"
STAND_TASK_ID = "SE3-WheelLegged-Flat-LY-Stand-GRU"
TURN_TASK_ID = "SE3-WheelLegged-Flat-LY-Turn-GRU"
ARC_TASK_ID = "SE3-WheelLegged-Flat-LY-Arc-GRU"


def register() -> None:
    """注册基线、静站精修、原地转向和行进转弯任务。"""
    register_mjlab_task(
        task_id=TASK_ID,
        env_cfg=env_cfg(),
        play_env_cfg=env_cfg(play=True),
        rl_cfg=rl_cfg(),
        runner_cls=Se3ProfiledOnPolicyRunner,
    )
    for task_id, phase in (
        (STAND_TASK_ID, "stand"),
        (TURN_TASK_ID, "turn"),
        (ARC_TASK_ID, "arc"),
    ):
        register_mjlab_task(
            task_id=task_id,
            env_cfg=env_cfg(phase=phase),
            play_env_cfg=env_cfg(play=True, phase=phase),
            rl_cfg=rl_cfg(phase=phase),
            runner_cls=Se3ProfiledOnPolicyRunner,
        )


__all__ = [
    "ARC_TASK_ID",
    "STAND_TASK_ID",
    "TASK_ID",
    "TURN_TASK_ID",
    "env_cfg",
    "register",
    "rl_cfg",
]
