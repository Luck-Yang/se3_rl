"""flat_ly 终止条件配置学习接口。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactSensor

from se3_train.mdp.contact_utils import finite_contact_force_norm

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class BaseContactDelayed:
    """base_link 持续接地后终止，过滤单步碰撞同时排除趴地策略。"""

    def __init__(self) -> None:
        self._contact_steps: torch.Tensor | None = None

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        sensor_name: str,
        force_threshold: float = 1.0,
        max_steps: int = 8,
    ) -> torch.Tensor:
        if (
            self._contact_steps is None
            or self._contact_steps.shape[0] != env.num_envs
            or self._contact_steps.device != env.device
        ):
            self._contact_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)

        sensor: ContactSensor = env.scene[sensor_name]
        if sensor.data.force is None:
            return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        contact_force = finite_contact_force_norm(sensor.data.force)
        in_contact = contact_force.max(dim=1).values > float(force_threshold)
        self._contact_steps[in_contact] += 1
        self._contact_steps[~in_contact] = 0
        self._contact_steps[env.episode_length_buf <= 1] = 0
        return self._contact_steps > int(max_steps)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """清空已重置环境的连续接地计数。"""
        if self._contact_steps is not None and env_ids is not None:
            self._contact_steps[env_ids] = 0


base_contact_delayed = BaseContactDelayed()


def configure_terminations(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """收紧长期坏姿态，并禁止依靠 base_link 接地维持平衡。"""
    del play
    cfg.terminations["bad_orientation"].params.update(
        {
            "limit_angle": 0.5236,
            "max_steps": 15,
        }
    )
    cfg.terminations["base_contact"] = TerminationTermCfg(
        func=base_contact_delayed,
        time_out=False,
        params={
            "sensor_name": "collision_sensor",
            "force_threshold": 1.0,
            "max_steps": 8,
        },
    )
