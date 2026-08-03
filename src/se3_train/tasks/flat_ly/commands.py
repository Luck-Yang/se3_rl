"""flat_ly 指令配置学习接口。"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal

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
    yaw_vel_slew_rate: float | None = None
    """yaw-rate 指令最大变化率(rad/s²)，高速转向时避免瞬时反向。"""
    stratified_yaw_sampling: bool = False
    """是否分层覆盖低速、中速、高速和当前 yaw 边界。"""

    def build(self, env: ManagerBasedRlEnv) -> FlatLyCommandTerm:
        return FlatLyCommandTerm(self, env)


class FlatLyCommandTerm(JumpCommandTerm):
    """flat_ly 专用指令项，在非静站样本内显式拆分运动能力。"""

    cfg: FlatLyCommandCfg

    def __init__(self, cfg: FlatLyCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._yaw_vel_target = torch.zeros(self.num_envs, device=self.device)

    def _sample_stratified_yaw(self, count: int) -> torch.Tensor:
        """按四个速度桶采样 yaw，保证高速课程不遗忘低速精确转向。"""
        if count <= 0:
            return torch.empty(0, device=self.device)
        yaw_max = max(
            abs(float(self.cfg.ang_vel_yaw_range[0])), abs(float(self.cfg.ang_vel_yaw_range[1]))
        )
        yaw_max = max(yaw_max, float(self.cfg.yaw_deadband))
        low_top = min(1.0, yaw_max)
        mid_bottom = min(1.0, yaw_max)
        mid_top = min(4.0, yaw_max)
        high_bottom = min(4.0, yaw_max)
        boundary_bottom = max(0.85 * yaw_max, float(self.cfg.yaw_deadband))

        bucket = torch.randint(0, 4, (count,), device=self.device)
        lower = torch.where(
            bucket == 0,
            torch.full((count,), float(self.cfg.yaw_deadband), device=self.device),
            torch.where(
                bucket == 1,
                torch.full((count,), mid_bottom, device=self.device),
                torch.where(
                    bucket == 2,
                    torch.full((count,), high_bottom, device=self.device),
                    torch.full((count,), boundary_bottom, device=self.device),
                ),
            ),
        )
        upper = torch.where(
            bucket == 0,
            torch.full((count,), low_top, device=self.device),
            torch.where(
                bucket == 1,
                torch.full((count,), mid_top, device=self.device),
                torch.full((count,), yaw_max, device=self.device),
            ),
        )
        lower = torch.minimum(lower, upper)
        magnitude = lower + torch.rand(count, device=self.device) * (upper - lower)
        sign = torch.where(
            torch.rand(count, device=self.device) < 0.5,
            -torch.ones(count, device=self.device),
            torch.ones(count, device=self.device),
        )
        return magnitude * sign

    def _clamp_yaw_to_wheel_budget(
        self,
        lin_vel: torch.Tensor,
        yaw_vel: torch.Tensor,
    ) -> torch.Tensor:
        """按真实差速轮包络裁剪 yaw，同时保留分层采样结果。"""
        if not self.cfg.constrain_diff_drive_commands:
            return yaw_vel
        wheel_budget = (
            float(self.cfg.diff_drive_wheel_radius)
            * float(self.cfg.diff_drive_max_wheel_speed)
            * float(self.cfg.diff_drive_wheel_speed_fraction)
        )
        half_track = max(float(self.cfg.diff_drive_half_track), 1.0e-6)
        yaw_limit = torch.clamp((wheel_budget - torch.abs(lin_vel)) / half_track, min=0.0)
        return torch.clamp(yaw_vel, min=-yaw_limit, max=yaw_limit)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """采样基础指令后，拆分直行、原地 yaw 和组合样本。"""
        previous_yaw = self._command[env_ids, 1].clone()
        super()._resample_command(env_ids)
        moving_ids = env_ids[~self._standing_mask[env_ids]]
        standing_ids = env_ids[self._standing_mask[env_ids]]
        self._yaw_vel_target[standing_ids] = 0.0
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

        yaw_command_ids = moving_ids[
            (profile < yaw_only_ratio) | (profile >= yaw_only_ratio + straight_ratio)
        ]
        if self.cfg.stratified_yaw_sampling and len(yaw_command_ids) > 0:
            self._command[yaw_command_ids, 1] = self._sample_stratified_yaw(len(yaw_command_ids))

        yaw_target = self._clamp_yaw_to_wheel_budget(
            self._lin_vel_target[moving_ids],
            self._command[moving_ids, 1],
        )
        yaw_target = torch.where(
            torch.abs(yaw_target) < float(self.cfg.yaw_deadband),
            torch.zeros_like(yaw_target),
            yaw_target,
        )
        self._yaw_vel_target[moving_ids] = yaw_target
        self._yaw_vel_target[straight_ids] = 0.0

        if self.cfg.yaw_vel_slew_rate is None:
            self._command[moving_ids, 1] = self._yaw_vel_target[moving_ids]
        elif self._resampling_for_reset:
            self._command[env_ids, 1] = 0.0
        else:
            self._command[env_ids, 1] = previous_yaw

    def _update_command(self) -> None:
        """沿用线速度更新，并对 yaw-rate 目标施加独立斜坡。"""
        super()._update_command()
        slew_rate = self.cfg.yaw_vel_slew_rate
        if slew_rate is None:
            return
        dt = max(float(getattr(self._env, "step_dt", 0.02)), 1.0e-6)
        max_delta = max(float(slew_rate), 0.0) * dt
        previous = self._command[:, 1]
        delta = torch.clamp(self._yaw_vel_target - previous, -max_delta, max_delta)
        yaw_vel = previous + delta
        yaw_vel = torch.where(
            (~self._standing_mask) & (torch.abs(yaw_vel) < float(self.cfg.yaw_deadband)),
            torch.zeros_like(yaw_vel),
            yaw_vel,
        )
        self._command[:, 1] = yaw_vel
        # JumpCommandTerm 已先执行过 recovery 门控；yaw 斜坡写回后必须再次归零。
        self._apply_recovery_command()


def configure_commands(
    cfg: ManagerBasedRlEnvCfg,
    *,
    play: bool,
    phase: Literal["base", "stand", "turn", "arc"] = "base",
) -> None:
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
    # 使用真实轮距和轮半径；0.98 的轮速预算刚好覆盖 12 rad/s 原地旋转。
    command_cfg.constrain_diff_drive_commands = True
    command_cfg.diff_drive_wheel_radius = 0.059
    command_cfg.diff_drive_half_track = 0.21665
    command_cfg.diff_drive_max_wheel_speed = 45.0
    command_cfg.diff_drive_wheel_speed_fraction = 0.98
    # 课程第一档为 ±0.2 m/s，死区必须明显小于课程步长。
    command_cfg.lin_vel_deadband = 0.03
    command_cfg.yaw_deadband = 0.03
    command_cfg.lin_vel_slew_rate = None if play else 0.8

    if phase == "stand":
        command_cfg.standing_ratio = 0.80
        command_cfg.straight_motion_ratio = 0.75
        command_cfg.yaw_only_ratio = 0.25
        command_cfg.lin_vel_x_range = (-2.0, 2.0)
        command_cfg.ang_vel_yaw_range = (-0.3, 0.3)
    elif phase == "turn":
        command_cfg.standing_ratio = 0.30
        command_cfg.straight_motion_ratio = 0.2142857143
        command_cfg.yaw_only_ratio = 0.7857142857
        command_cfg.lin_vel_x_range = (-2.0, 2.0)
        command_cfg.ang_vel_yaw_range = (-0.3, 0.3)
        command_cfg.yaw_vel_slew_rate = None if play else 6.0
        command_cfg.stratified_yaw_sampling = True
        command_cfg.resampling_time_range = (8.0, 8.0)
    elif phase == "arc":
        command_cfg.standing_ratio = 0.25
        command_cfg.straight_motion_ratio = 1.0 / 3.0
        command_cfg.yaw_only_ratio = 1.0 / 3.0
        command_cfg.lin_vel_x_range = (-2.0, 2.0)
        command_cfg.ang_vel_yaw_range = (-0.3, 0.3)
        command_cfg.yaw_vel_slew_rate = None if play else 6.0
        command_cfg.stratified_yaw_sampling = True
        command_cfg.resampling_time_range = (8.0, 8.0)
    else:
        # V8 基线：静站 35%，其余拆成直行 50%、原地 yaw 25%、组合 25%。
        command_cfg.standing_ratio = 0.35
        command_cfg.straight_motion_ratio = 0.50
        command_cfg.yaw_only_ratio = 0.25
