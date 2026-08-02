"""flat_ly 指令配置学习接口。"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.mdp.jump_commands import JumpCommandCfg, JumpCommandTerm

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


@dataclass
class FlatLyCommandCfg(JumpCommandCfg):
    """flat_ly 专用的静站、直行和 yaw 分桶指令配置。"""

    straight_motion_ratio: float = 0.50
    """非静站样本中强制 yaw_rate=0 的直行比例。"""
    yaw_only_ratio: float = 0.25
    """非静站样本中强制 vx=0 的原地 yaw 纠偏比例。"""

    def build(self, env: ManagerBasedRlEnv) -> FlatLyCommandTerm:
        return FlatLyCommandTerm(self, env)


class FlatLyCommandTerm(JumpCommandTerm):
    """flat_ly 专用指令项，在非静站样本内显式拆分运动能力。"""

    cfg: FlatLyCommandCfg

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """采样基础指令后，拆分直行、原地 yaw 和组合样本。"""
        super()._resample_command(env_ids)
        moving_ids = env_ids[~self._standing_mask[env_ids]]
        if len(moving_ids) == 0:
            return

        yaw_only_ratio = min(max(float(self.cfg.yaw_only_ratio), 0.0), 1.0)
        straight_ratio = min(
            max(float(self.cfg.straight_motion_ratio), 0.0),
            1.0 - yaw_only_ratio,
        )
        profile = torch.rand(len(moving_ids), device=self.device)
        yaw_only_ids = moving_ids[profile < yaw_only_ratio]
        straight_ids = moving_ids[
            (profile >= yaw_only_ratio) & (profile < yaw_only_ratio + straight_ratio)
        ]

        if len(yaw_only_ids) > 0:
            self._lin_vel_target[yaw_only_ids] = 0.0
            if self.cfg.lin_vel_slew_rate is None or self._resampling_for_reset:
                self._command[yaw_only_ids, 0] = 0.0
                self._lin_vel_accel[yaw_only_ids] = 0.0
        if len(straight_ids) > 0:
            self._command[straight_ids, 1] = 0.0


def configure_commands(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """固定稳态水平目标，训练时对速度指令施加加速度限幅。"""
    source_cfg = cfg.commands["velocity_height"]
    inherited = {
        field.name: getattr(source_cfg, field.name)
        for field in fields(JumpCommandCfg)
        if field.init
    }
    command_cfg = FlatLyCommandCfg(**inherited)
    cfg.commands["velocity_height"] = command_cfg
    command_cfg.pitch_range = (0.0, 0.0)
    command_cfg.roll_range = (0.0, 0.0)
    # 保留足够的独立静站样本，避免运动样本掩盖长时间漂移。
    command_cfg.standing_ratio = 0.35
    # 非静站样本拆成直行 50%、原地 yaw 25%、组合运动 25%。
    command_cfg.straight_motion_ratio = 0.50
    command_cfg.yaw_only_ratio = 0.25
    # 课程第一档为 ±0.2 m/s，死区必须明显小于课程步长。
    command_cfg.lin_vel_deadband = 0.03
    command_cfg.yaw_deadband = 0.03
    command_cfg.lin_vel_slew_rate = None if play else 0.8
