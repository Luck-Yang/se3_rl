"""用于学习完整训练流程的 flat_ly 平地 GRU 任务。"""

from __future__ import annotations

import math
import os

import torch
from mjlab.tasks.registry import register_mjlab_task

from se3_train.tasks.common import Se3ProfiledOnPolicyRunner, Se3WarmStartRunner

from .env_cfg import env_cfg
from .rl_cfg import rl_cfg

TASK_ID = "SE3-WheelLegged-Flat-LY-GRU"
STAND_TASK_ID = "SE3-WheelLegged-Flat-LY-Stand-GRU"
TURN_TASK_ID = "SE3-WheelLegged-Flat-LY-Turn-GRU"
ARC_TASK_ID = "SE3-WheelLegged-Flat-LY-Arc-GRU"


def _full_resume_enabled() -> bool:
    """判断是否显式请求完整恢复。"""
    return os.environ.get("SE3_FULL_RESUME", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class FlatLyWarmStartRunner(Se3WarmStartRunner):
    """flat_ly 阶段切换 runner，保留新阶段配置的探索噪声和学习率。"""

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict:
        """加载网络权重，并恢复新阶段在配置中声明的优化超参数。"""
        infos = super().load(
            path,
            load_cfg=load_cfg,
            strict=strict,
            map_location=map_location,
        )
        if load_cfg is not None or _full_resume_enabled():
            return infos

        # actor checkpoint 包含 distribution 参数；warm-start 后必须重新应用新阶段的 init_std。
        distribution_cfg = self.cfg["actor"]["distribution_cfg"]
        init_std = float(distribution_cfg["init_std"])
        distribution = self.alg._raw_actor.distribution
        with torch.no_grad():
            if distribution.std_type == "scalar":
                distribution.std_param.fill_(init_std)
            elif distribution.std_type == "log":
                distribution.log_std_param.fill_(math.log(init_std))
            else:
                raise ValueError(f"不支持的 std_type：{distribution.std_type}")

        # optimizer 是新建的；这里显式重申配置值，防止阶段加载语义被后续改动破坏。
        learning_rate = float(self.cfg["algorithm"]["learning_rate"])
        self.alg.learning_rate = learning_rate
        for param_group in self.alg.optimizer.param_groups:
            param_group["lr"] = learning_rate

        if not self.is_distributed or self.gpu_global_rank == 0:
            print(
                "[Flat LY WarmStart] reapplied stage config "
                f"init_std={init_std:g} learning_rate={learning_rate:g}",
                flush=True,
            )
        return infos


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
            runner_cls=FlatLyWarmStartRunner,
        )


__all__ = [
    "ARC_TASK_ID",
    "STAND_TASK_ID",
    "TASK_ID",
    "TURN_TASK_ID",
    "FlatLyWarmStartRunner",
    "env_cfg",
    "register",
    "rl_cfg",
]
