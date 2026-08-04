"""flat_ly 指令配置学习接口。"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat

from se3_train.mdp import recovery_state
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
    enable_stationkeeping_outer_loop: bool = True
    """静站时是否用世界系位姿误差生成小幅速度纠偏指令。"""
    stationkeeping_position_enter: float = 0.04
    """位置误差超过该值时进入返回锚点模式(m)。"""
    stationkeeping_position_exit: float = 0.015
    """位置误差低于该值时退出返回锚点模式(m)，与进入阈值组成滞环。"""
    stationkeeping_yaw_deadband: float = 0.02
    """静站 yaw 纠偏死区(rad)。"""
    stationkeeping_position_gain: float = 1.2
    """返回锚点时距离到线速度的比例增益。"""
    stationkeeping_bearing_gain: float = 2.0
    """返回锚点时朝向误差到 yaw-rate 的比例增益。"""
    stationkeeping_yaw_gain: float = 1.5
    """已回到锚点附近时 yaw 误差到 yaw-rate 的比例增益。"""
    stationkeeping_max_lin_vel: float = 0.18
    """静站外环允许生成的最大绝对线速度(m/s)。"""
    stationkeeping_max_yaw_vel: float = 0.30
    """静站外环允许生成的最大绝对 yaw-rate(rad/s)。"""
    stationkeeping_lin_vel_slew_rate: float = 0.8
    """静站外环线速度指令最大变化率(m/s²)。"""
    stationkeeping_yaw_vel_slew_rate: float = 2.0
    """静站外环 yaw-rate 指令最大变化率(rad/s²)。"""
    stationkeeping_max_tilt_deg: float = 8.0
    """pitch/roll 超过该角度时暂停位姿纠偏，优先让倒立摆恢复平衡。"""

    def build(self, env: ManagerBasedRlEnv) -> FlatLyCommandTerm:
        return FlatLyCommandTerm(self, env)


class FlatLyCommandTerm(JumpCommandTerm):
    """flat_ly 专用指令项，在非静站样本内显式拆分运动能力。"""

    cfg: FlatLyCommandCfg

    def __init__(self, cfg: FlatLyCommandCfg, env: ManagerBasedRlEnv) -> None:
        super().__init__(cfg, env)
        self._yaw_vel_target = torch.zeros(self.num_envs, device=self.device)
        self._stationkeeping_anchor_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
        self._stationkeeping_anchor_yaw = torch.zeros(self.num_envs, device=self.device)
        self._stationkeeping_anchor_valid = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._stationkeeping_anchor_pending = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._stationkeeping_returning = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._stationkeeping_position_error_w = torch.zeros(self.num_envs, 2, device=self.device)
        self._stationkeeping_yaw_error = torch.zeros(self.num_envs, device=self.device)
        self._stationkeeping_correction_mask = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self._stationkeeping_control_allowed = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )

    @property
    def standing_mask(self) -> torch.Tensor:
        """返回由采样器决定的静站样本掩码。"""
        return self._standing_mask

    @property
    def stationkeeping_position_error_w(self) -> torch.Tensor:
        """返回从当前 base 指向静站锚点的世界系平面误差(m)。"""
        return self._stationkeeping_position_error_w

    @property
    def stationkeeping_position_error(self) -> torch.Tensor:
        """返回静站锚点平面距离误差(m)。"""
        return torch.linalg.vector_norm(self._stationkeeping_position_error_w, dim=1)

    @property
    def stationkeeping_yaw_error(self) -> torch.Tensor:
        """返回包裹到 [-pi, pi] 的静站锚点 yaw 误差(rad)。"""
        return self._stationkeeping_yaw_error

    @property
    def stationkeeping_correction_mask(self) -> torch.Tensor:
        """返回外环当前正在发出非零纠偏命令的样本掩码。"""
        return self._stationkeeping_correction_mask

    @property
    def stationkeeping_settled_mask(self) -> torch.Tensor:
        """返回锚点有效且当前无需纠偏的静站样本掩码。"""
        position_settled = self.stationkeeping_position_error <= float(
            self.cfg.stationkeeping_position_exit
        )
        yaw_settled = torch.abs(self._stationkeeping_yaw_error) < float(
            self.cfg.stationkeeping_yaw_deadband
        )
        return (
            self._standing_mask
            & self._stationkeeping_anchor_valid
            & self._stationkeeping_control_allowed
            & position_settled
            & yaw_settled
            & ~self._stationkeeping_correction_mask
        )

    @staticmethod
    def _wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
        """把弧度角包裹到 [-pi, pi]。"""
        return torch.atan2(torch.sin(angle), torch.cos(angle))

    def _capture_stationkeeping_anchors(self) -> None:
        """在 reset 写入完成或首次进入静站后锁定世界系位姿锚点。"""
        pending = self._stationkeeping_anchor_pending & self._standing_mask
        if not pending.any():
            return
        robot = self._env.scene["robot"]
        self._stationkeeping_anchor_pos_w[pending] = robot.data.root_link_pos_w[pending, :2]
        _, _, yaw = euler_xyz_from_quat(robot.data.root_link_quat_w[pending])
        self._stationkeeping_anchor_yaw[pending] = yaw
        self._stationkeeping_anchor_valid[pending] = True
        self._stationkeeping_anchor_pending[pending] = False
        self._stationkeeping_returning[pending] = False

    def _stationkeeping_targets(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """把静站位姿误差转换为可部署的差速车速度外环目标。"""
        self._capture_stationkeeping_anchors()
        robot = self._env.scene["robot"]
        current_pos = robot.data.root_link_pos_w[:, :2]
        _, _, current_yaw = euler_xyz_from_quat(robot.data.root_link_quat_w)
        position_error_w = self._stationkeeping_anchor_pos_w - current_pos
        yaw_error = self._wrap_to_pi(self._stationkeeping_anchor_yaw - current_yaw)

        recovery = recovery_state.recovery_active_mask(self._env).to(
            device=self.device, dtype=torch.bool
        )
        jump_active = self._command[:, 5] > 0.5
        stationkeeping = (
            self._standing_mask & self._stationkeeping_anchor_valid & ~recovery & ~jump_active
        )
        position_error_w = torch.where(
            stationkeeping[:, None], position_error_w, torch.zeros_like(position_error_w)
        )
        yaw_error = torch.where(stationkeeping, yaw_error, torch.zeros_like(yaw_error))
        self._stationkeeping_position_error_w.copy_(position_error_w)
        self._stationkeeping_yaw_error.copy_(yaw_error)

        projected_gravity = robot.data.projected_gravity_b
        pitch = torch.asin(torch.clamp(projected_gravity[:, 0], -1.0, 1.0))
        roll = torch.asin(torch.clamp(-projected_gravity[:, 1], -1.0, 1.0))
        tilt_limit = math.radians(max(float(self.cfg.stationkeeping_max_tilt_deg), 0.0))
        control_allowed = (
            stationkeeping & (torch.abs(pitch) <= tilt_limit) & (torch.abs(roll) <= tilt_limit)
        )
        self._stationkeeping_control_allowed.copy_(control_allowed)

        distance = torch.linalg.vector_norm(position_error_w, dim=1)
        enter = max(float(self.cfg.stationkeeping_position_enter), 0.0)
        exit_ = min(max(float(self.cfg.stationkeeping_position_exit), 0.0), enter)
        self._stationkeeping_returning |= control_allowed & (distance >= enter)
        self._stationkeeping_returning &= control_allowed & (distance > exit_)

        target_bearing = torch.atan2(position_error_w[:, 1], position_error_w[:, 0])
        bearing_error = self._wrap_to_pi(target_bearing - current_yaw)
        reverse = torch.abs(bearing_error) > (0.5 * torch.pi)
        bearing_error = torch.where(
            reverse,
            self._wrap_to_pi(bearing_error + torch.pi),
            bearing_error,
        )
        direction = torch.where(reverse, -torch.ones_like(distance), torch.ones_like(distance))
        lin_target = direction * float(self.cfg.stationkeeping_position_gain) * distance
        lin_target *= torch.clamp(torch.cos(bearing_error), min=0.0)
        yaw_target = float(self.cfg.stationkeeping_bearing_gain) * bearing_error

        yaw_hold = float(self.cfg.stationkeeping_yaw_gain) * yaw_error
        lin_target = torch.where(self._stationkeeping_returning, lin_target, 0.0)
        yaw_target = torch.where(self._stationkeeping_returning, yaw_target, yaw_hold)
        yaw_target = torch.where(
            self._stationkeeping_returning
            | (torch.abs(yaw_error) >= float(self.cfg.stationkeeping_yaw_deadband)),
            yaw_target,
            0.0,
        )
        lin_target = torch.clamp(
            lin_target,
            min=-float(self.cfg.stationkeeping_max_lin_vel),
            max=float(self.cfg.stationkeeping_max_lin_vel),
        )
        yaw_target = torch.clamp(
            yaw_target,
            min=-float(self.cfg.stationkeeping_max_yaw_vel),
            max=float(self.cfg.stationkeeping_max_yaw_vel),
        )
        lin_target = torch.where(control_allowed, lin_target, 0.0)
        yaw_target = torch.where(control_allowed, yaw_target, 0.0)
        correction = control_allowed & ((lin_target != 0.0) | (yaw_target != 0.0))
        self._stationkeeping_correction_mask.copy_(correction)
        return lin_target, yaw_target, stationkeeping

    @staticmethod
    def _slew_target(
        previous: torch.Tensor,
        target: torch.Tensor,
        rate: float,
        dt: float,
    ) -> torch.Tensor:
        """按给定变化率把指令平滑推进到目标。"""
        max_delta = max(float(rate), 0.0) * dt
        return previous + torch.clamp(target - previous, -max_delta, max_delta)

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
        previous_standing = self._standing_mask[env_ids].clone()
        previous_yaw = self._command[env_ids, 1].clone()
        super()._resample_command(env_ids)
        moving_ids = env_ids[~self._standing_mask[env_ids]]
        standing_ids = env_ids[self._standing_mask[env_ids]]
        self._stationkeeping_anchor_valid[moving_ids] = False
        self._stationkeeping_anchor_pending[moving_ids] = False
        self._stationkeeping_returning[moving_ids] = False
        newly_standing = self._standing_mask[env_ids] & ~previous_standing
        if self._resampling_for_reset:
            newly_standing = self._standing_mask[env_ids]
        pending_ids = env_ids[newly_standing]
        self._stationkeeping_anchor_valid[pending_ids] = False
        self._stationkeeping_anchor_pending[pending_ids] = True
        self._stationkeeping_returning[pending_ids] = False
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
        """更新运动指令，并在静站样本上叠加位姿外环。"""
        previous_lin = self._command[:, 0].clone()
        previous_yaw = self._command[:, 1].clone()
        super()._update_command()
        slew_rate = self.cfg.yaw_vel_slew_rate
        dt = max(float(getattr(self._env, "step_dt", 0.02)), 1.0e-6)
        if slew_rate is not None:
            yaw_vel = self._slew_target(
                previous_yaw,
                self._yaw_vel_target,
                float(slew_rate),
                dt,
            )
            yaw_vel = torch.where(
                (~self._standing_mask) & (torch.abs(yaw_vel) < float(self.cfg.yaw_deadband)),
                torch.zeros_like(yaw_vel),
                yaw_vel,
            )
            self._command[:, 1] = yaw_vel

        if self.cfg.enable_stationkeeping_outer_loop:
            lin_target, yaw_target, active = self._stationkeeping_targets()
            lin_vel = self._slew_target(
                previous_lin,
                lin_target,
                self.cfg.stationkeeping_lin_vel_slew_rate,
                dt,
            )
            yaw_vel = self._slew_target(
                previous_yaw,
                yaw_target,
                self.cfg.stationkeeping_yaw_vel_slew_rate,
                dt,
            )
            self._command[:, 0] = torch.where(active, lin_vel, self._command[:, 0])
            self._command[:, 1] = torch.where(active, yaw_vel, self._command[:, 1])
            self._lin_vel_accel[:] = torch.where(
                active,
                (self._command[:, 0] - previous_lin) / dt,
                self._lin_vel_accel,
            )
            self._stationkeeping_correction_mask.copy_(
                active
                & (
                    (torch.abs(self._command[:, 0]) > 1.0e-4)
                    | (torch.abs(self._command[:, 1]) > 1.0e-4)
                )
            )
        else:
            self._stationkeeping_position_error_w.zero_()
            self._stationkeeping_yaw_error.zero_()
            self._stationkeeping_correction_mask.zero_()

        # JumpCommandTerm 已先执行过 recovery 门控；外环写回后必须再次归零。
        self._apply_recovery_command()


def configure_commands(
    cfg: ManagerBasedRlEnvCfg,
    *,
    play: bool,
    phase: Literal["base", "speed", "stand", "turn", "arc"] = "base",
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

    if phase == "speed":
        # 中高速阶段保留 20% 静站样本防止遗忘，其余样本以双向直行为主。
        command_cfg.standing_ratio = 0.20
        command_cfg.straight_motion_ratio = 0.85
        command_cfg.yaw_only_ratio = 0.0
        # 训练从第一档开始；回放时直接开放完整范围，便于验证最终 checkpoint。
        command_cfg.lin_vel_x_range = (-2.0, 2.0) if play else (-0.2, 0.2)
        command_cfg.ang_vel_yaw_range = (-0.3, 0.3)
        command_cfg.yaw_vel_slew_rate = None if play else 2.0
        command_cfg.resampling_time_range = (8.0, 8.0)
    elif phase == "stand":
        command_cfg.standing_ratio = 0.80
        command_cfg.straight_motion_ratio = 0.75
        command_cfg.yaw_only_ratio = 0.25
        # 第一阶段从 model_0 随机策略起步，非静站样本仅用于学习小幅速度响应。
        command_cfg.lin_vel_x_range = (-0.2, 0.2)
        command_cfg.ang_vel_yaw_range = (-0.1, 0.1)
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
