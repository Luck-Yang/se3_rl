"""flat_ly 奖励设计入口；具体奖励由学习者完成。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Literal

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

from se3_shared import wrap_front_action_value_delta_torch
from se3_train.mdp import recovery_state
from se3_train.mdp import rewards as mdp_rewards
from se3_train.mdp.contact_utils import finite_contact_force_norm
from se3_train.mdp.diagnostic_logging import should_log_diagnostics
from se3_train.mdp.joint_indices import wheel_joint_ids

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


_GRAVITY_M_S2 = 9.81


def _standing_mask(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """返回指令生成器显式采样的静站环境。"""
    term = env.command_manager.get_term(command_name)
    mask = getattr(term, "_standing_mask", None)
    if isinstance(mask, torch.Tensor) and mask.shape == (env.num_envs,):
        return mask.to(device=env.device, dtype=torch.bool)
    command = env.command_manager.get_command(command_name)
    return torch.abs(command[:, 0]) <= 0.02


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """返回分桶均值；无样本时返回零。"""
    if mask.any():
        return value[mask].mean()
    return torch.zeros((), device=value.device, dtype=value.dtype)


def _orientation_state(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_accel: float,
    max_pitch_deg: float,
) -> dict[str, torch.Tensor]:
    """只按指令斜率计算短暂 pitch 前馈目标，稳态目标为零。"""
    robot = env.scene["robot"]
    term = env.command_manager.get_term(command_name)
    command_accel = getattr(term, "lin_vel_accel", None)
    if not isinstance(command_accel, torch.Tensor) or command_accel.shape[0] != env.num_envs:
        command_accel = torch.zeros(env.num_envs, device=env.device)

    projected_gravity = robot.data.projected_gravity_b
    pitch = torch.asin(torch.clamp(projected_gravity[:, 0], -1.0, 1.0))
    roll = torch.asin(torch.clamp(-projected_gravity[:, 1], -1.0, 1.0))

    # 速度误差由速度奖励处理；不允许大误差长期改写姿态平衡点。
    accel_ref = torch.clamp(command_accel, -float(max_accel), float(max_accel))
    pitch_ref = torch.atan(accel_ref / _GRAVITY_M_S2)
    pitch_limit = math.radians(float(max_pitch_deg))
    pitch_ref = torch.clamp(pitch_ref, -pitch_limit, pitch_limit)
    # 静站的物理平衡点必须是 pitch=0，不允许用长期倾斜换取滚动速度。
    pitch_ref = torch.where(_standing_mask(env, command_name), 0.0, pitch_ref)

    return {
        "pitch": pitch,
        "roll": roll,
        "pitch_ref": pitch_ref,
        "accel_ref": accel_ref,
    }


def _scaled_deadband_square(
    error: torch.Tensor,
    *,
    deadband: float,
    scale: float,
    max_penalty: float,
) -> torch.Tensor:
    """返回带死区、物理尺度和上限的平方惩罚。"""
    excess = torch.clamp(torch.abs(error) - float(deadband), min=0.0)
    penalty = torch.square(excess / max(float(scale), 1.0e-6))
    return torch.clamp(penalty, max=float(max_penalty))


def dynamic_pitch_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_accel: float = 0.5,
    max_pitch_deg: float = 3.0,
    deadband_deg: float = 1.5,
    scale_deg: float = 4.0,
    max_penalty: float = 9.0,
) -> torch.Tensor:
    """惩罚 pitch 偏离指令加速度前馈目标，恒速和静站目标均为零。"""
    state = _orientation_state(env, command_name, max_accel, max_pitch_deg)
    error = state["pitch"] - state["pitch_ref"]
    penalty = _scaled_deadband_square(
        error,
        deadband=math.radians(float(deadband_deg)),
        scale=math.radians(float(scale_deg)),
        max_penalty=max_penalty,
    )

    if should_log_diagnostics(env, 64, attr_name="_se3_reward_log_interval_steps"):
        log = env.extras.setdefault("log", {})
        if isinstance(log, dict):
            log.update(
                {
                    "Locomotion/flat_ly_pitch_deg": torch.rad2deg(state["pitch"])
                    .abs()
                    .mean()
                    .item(),
                    "Locomotion/flat_ly_pitch_ref_deg": torch.rad2deg(state["pitch_ref"])
                    .abs()
                    .mean()
                    .item(),
                    "Locomotion/flat_ly_accel_ref": state["accel_ref"].abs().mean().item(),
                    "Locomotion/flat_ly_dynamic_pitch_penalty": penalty.mean().item(),
                }
            )
    return penalty


def roll_stability_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    deadband_deg: float = 2.0,
    scale_deg: float = 5.0,
    max_penalty: float = 9.0,
) -> torch.Tensor:
    """惩罚超出误差死区的 roll，平地直行时目标为零。"""
    state = _orientation_state(env, command_name, 0.5, 3.0)
    return _scaled_deadband_square(
        state["roll"],
        deadband=math.radians(float(deadband_deg)),
        scale=math.radians(float(scale_deg)),
        max_penalty=max_penalty,
    )


def yaw_rate_tracking_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    deadband: float = 0.01,
    scale: float = 0.15,
    max_penalty: float = 9.0,
) -> torch.Tensor:
    """惩罚 yaw 角速度跟踪误差，静站指令为零时自然压制漂移。"""
    yaw_rate = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    yaw_rate_command = env.command_manager.get_command(command_name)[:, 1]
    yaw_rate_error = yaw_rate - yaw_rate_command
    penalty = _scaled_deadband_square(
        yaw_rate_error,
        deadband=deadband,
        scale=scale,
        max_penalty=max_penalty,
    )
    if should_log_diagnostics(env, 64, attr_name="_se3_reward_log_interval_steps"):
        env.extras.setdefault("log", {}).update(
            {
                "Locomotion/flat_ly_yaw_rate_abs": yaw_rate.abs().mean().item(),
                "Locomotion/flat_ly_yaw_rate_command_abs": yaw_rate_command.abs().mean().item(),
                "Locomotion/flat_ly_yaw_rate_error_abs": yaw_rate_error.abs().mean().item(),
                "Locomotion/flat_ly_yaw_rate_tracking_penalty": penalty.mean().item(),
            }
        )
    return penalty


def yaw_rate_huber_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    deadband: float = 0.03,
    base_scale: float = 0.5,
    relative_scale: float = 0.05,
) -> torch.Tensor:
    """高速 yaw 跟踪 Huber 惩罚，大误差区仍保留线性学习信号。"""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)[:, 1]
    error = torch.abs(robot.data.root_link_ang_vel_b[:, 2] - command)
    excess = torch.clamp(error - float(deadband), min=0.0)
    scale = float(base_scale) + float(relative_scale) * torch.abs(command)
    normalized = excess / torch.clamp(scale, min=1.0e-6)
    return torch.where(normalized <= 1.0, 0.5 * normalized**2, normalized - 0.5)


def yaw_rate_precision_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    base_scale: float = 0.2,
    relative_scale: float = 0.08,
) -> torch.Tensor:
    """按相对误差奖励高速和低速 yaw-rate 的近目标精度。"""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)[:, 1]
    error = robot.data.root_link_ang_vel_b[:, 2] - command
    scale = float(base_scale) + float(relative_scale) * torch.abs(command)
    return torch.exp(-torch.square(error / torch.clamp(scale, min=1.0e-6)))


def yaw_rate_progress_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    min_command: float = 0.05,
) -> torch.Tensor:
    """奖励沿指令符号建立角速度，避免高速目标初期只有稀疏精度奖励。"""
    command = env.command_manager.get_command(command_name)[:, 1]
    yaw_rate = env.scene["robot"].data.root_link_ang_vel_b[:, 2]
    active = torch.abs(command) > float(min_command)
    progress = yaw_rate * torch.sign(command) / torch.clamp(torch.abs(command), min=min_command)
    return torch.clamp(progress, min=-1.0, max=1.0) * active.float()


def in_place_translation_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_linear_command: float = 0.05,
    min_yaw_command: float = 0.1,
    velocity_scale: float = 0.25,
) -> torch.Tensor:
    """原地转向时惩罚机身平移，防止用画圈代替绕自身中心旋转。"""
    command = env.command_manager.get_command(command_name)
    active = (torch.abs(command[:, 0]) <= float(max_linear_command)) & (
        torch.abs(command[:, 1]) >= float(min_yaw_command)
    )
    lin_vel_xy = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
    penalty = torch.sum(torch.square(lin_vel_xy / max(float(velocity_scale), 1.0e-6)), dim=1)
    return penalty * active.float()


def lateral_velocity_penalty(
    env: ManagerBasedRlEnv,
    deadband: float = 0.03,
    scale: float = 0.20,
    max_penalty: float = 9.0,
) -> torch.Tensor:
    """惩罚机身侧向速度，平地差速轮直行时目标为零。"""
    lateral_velocity = env.scene["robot"].data.root_link_lin_vel_b[:, 1]
    penalty = _scaled_deadband_square(
        lateral_velocity,
        deadband=deadband,
        scale=scale,
        max_penalty=max_penalty,
    )
    if should_log_diagnostics(env, 64, attr_name="_se3_reward_log_interval_steps"):
        env.extras.setdefault("log", {}).update(
            {
                "Locomotion/flat_ly_lateral_velocity_abs": lateral_velocity.abs().mean().item(),
                "Locomotion/flat_ly_lateral_velocity_penalty": penalty.mean().item(),
            }
        )
    return penalty


def _standing_stationkeeping_values(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """返回静站的有梯度惩罚、指数分数和样本掩码。"""
    robot = env.scene["robot"]
    standing = _standing_mask(env, command_name)
    orientation = _orientation_state(env, command_name, 0.5, 3.0)

    components = torch.stack(
        (
            _scaled_deadband_square(
                robot.data.root_link_lin_vel_b[:, 0],
                deadband=0.03,
                scale=0.20,
                max_penalty=16.0,
            ),
            _scaled_deadband_square(
                robot.data.root_link_lin_vel_b[:, 1],
                deadband=0.03,
                scale=0.20,
                max_penalty=16.0,
            ),
            _scaled_deadband_square(
                orientation["pitch"],
                deadband=math.radians(1.5),
                scale=math.radians(5.0),
                max_penalty=16.0,
            ),
            _scaled_deadband_square(
                robot.data.root_link_ang_vel_b[:, 2],
                deadband=0.02,
                scale=0.15,
                max_penalty=16.0,
            ),
        ),
        dim=1,
    )
    penalty = torch.clamp(components.sum(dim=1), max=16.0)
    score = torch.exp(-penalty)
    return penalty * standing.float(), score * standing.float(), standing


def standing_stationkeeping_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """静站专属惩罚：压制机身漂移与姿态，但允许轮子来回平衡。"""
    penalty, _, standing = _standing_stationkeeping_values(env, command_name)
    if should_log_diagnostics(env, 64, attr_name="_se3_reward_log_interval_steps"):
        robot = env.scene["robot"]
        orientation = _orientation_state(env, command_name, 0.5, 3.0)
        wheel_vel = robot.data.joint_vel[:, wheel_joint_ids(robot)]
        wheel_forward_speed = torch.stack(
            (wheel_vel[:, 0] * 0.059, -wheel_vel[:, 1] * 0.059),
            dim=1,
        )
        env.extras.setdefault("log", {}).update(
            {
                "Locomotion/flat_ly_stand_vx_abs": _masked_mean(
                    robot.data.root_link_lin_vel_b[:, 0].abs(), standing
                ).item(),
                "Locomotion/flat_ly_stand_vy_abs": _masked_mean(
                    robot.data.root_link_lin_vel_b[:, 1].abs(), standing
                ).item(),
                "Locomotion/flat_ly_stand_pitch_abs_deg": torch.rad2deg(
                    _masked_mean(orientation["pitch"].abs(), standing)
                ).item(),
                "Locomotion/flat_ly_stand_yaw_rate_abs": _masked_mean(
                    robot.data.root_link_ang_vel_b[:, 2].abs(), standing
                ).item(),
                "Locomotion/flat_ly_stand_wheel_speed_abs": _masked_mean(
                    wheel_forward_speed.abs().mean(dim=1), standing
                ).item(),
                "Locomotion/flat_ly_stand_penalty": _masked_mean(penalty, standing).item(),
            }
        )
    return penalty


def standing_stationkeeping_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """静站专属正奖励，只在零速指令子集生效。"""
    _, score, _ = _standing_stationkeeping_values(env, command_name)
    return score


def standing_low_frequency_drift_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    tau_s: float = 0.75,
    base_deadband: float = 0.01,
    base_scale: float = 0.08,
    wheel_deadband: float = 0.03,
    wheel_scale: float = 0.12,
    wheel_weight: float = 0.25,
) -> torch.Tensor:
    """惩罚静站的低频速度偏置，同时允许轮子高频往复纠偏。"""
    robot = env.scene["robot"]
    standing = _standing_mask(env, command_name)
    wheel_vel = robot.data.joint_vel[:, wheel_joint_ids(robot)]
    wheel_forward = 0.5 * (wheel_vel[:, 0] * 0.059 - wheel_vel[:, 1] * 0.059)
    base_vx = robot.data.root_link_lin_vel_b[:, 0]

    base_ema = getattr(env, "_flat_ly_stand_base_vx_ema", None)
    wheel_ema = getattr(env, "_flat_ly_stand_wheel_vx_ema", None)
    if not isinstance(base_ema, torch.Tensor) or base_ema.shape[0] != env.num_envs:
        base_ema = torch.zeros(env.num_envs, device=env.device)
        wheel_ema = torch.zeros(env.num_envs, device=env.device)
        env._flat_ly_stand_base_vx_ema = base_ema
        env._flat_ly_stand_wheel_vx_ema = wheel_ema
    assert isinstance(wheel_ema, torch.Tensor)

    dt = max(float(getattr(env, "step_dt", 0.02)), 1.0e-6)
    alpha = 1.0 - math.exp(-dt / max(float(tau_s), dt))
    reset_mask = env.episode_length_buf <= 1
    base_ema[reset_mask] = 0.0
    wheel_ema[reset_mask] = 0.0
    base_ema.lerp_(base_vx, alpha)
    wheel_ema.lerp_(wheel_forward, alpha)

    base_penalty = torch.square(
        torch.clamp(torch.abs(base_ema) - float(base_deadband), min=0.0)
        / max(float(base_scale), 1.0e-6)
    )
    wheel_penalty = torch.square(
        torch.clamp(torch.abs(wheel_ema) - float(wheel_deadband), min=0.0)
        / max(float(wheel_scale), 1.0e-6)
    )
    penalty = (base_penalty + float(wheel_weight) * wheel_penalty) * standing.float()
    if should_log_diagnostics(env, 64, attr_name="_se3_reward_log_interval_steps"):
        env.extras.setdefault("log", {}).update(
            {
                "Locomotion/flat_ly_stand_base_vx_ema_abs": _masked_mean(
                    torch.abs(base_ema), standing
                ).item(),
                "Locomotion/flat_ly_stand_wheel_vx_ema_abs": _masked_mean(
                    torch.abs(wheel_ema), standing
                ).item(),
                "Locomotion/flat_ly_stand_low_frequency_penalty": _masked_mean(
                    penalty, standing
                ).item(),
            }
        )
    return penalty


def _wheel_body_ids(env: ManagerBasedRlEnv) -> list[int]:
    """返回左右轮 body 索引并在 env 上缓存。"""
    cached = getattr(env, "_flat_ly_wheel_body_ids", None)
    if isinstance(cached, list) and len(cached) == 2:
        return cached
    robot = env.scene["robot"]
    body_ids, body_names = robot.find_bodies(
        ("l_wheel_Link", "r_wheel_Link"),
        preserve_order=True,
    )
    if len(body_ids) != 2:
        raise RuntimeError(f"必须找到左右轮 body，实际找到: {body_names}")
    env._flat_ly_wheel_body_ids = body_ids
    return body_ids


def _wheel_positions_body(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回左右轮心在机身坐标系的位置。"""
    robot = env.scene["robot"]
    body_ids = _wheel_body_ids(env)
    delta_w = robot.data.body_link_pos_w[:, body_ids, :] - robot.data.root_link_pos_w[:, None, :]
    quat = robot.data.root_link_quat_w[:, None, :].expand(-1, len(body_ids), -1)
    return quat_apply_inverse(quat.reshape(-1, 4), delta_w.reshape(-1, 3)).reshape(
        env.num_envs,
        len(body_ids),
        3,
    )


def _wheel_geometry_values(
    env: ManagerBasedRlEnv,
    x_deadband: float,
    z_deadband: float,
    x_scale: float,
    z_scale: float,
    max_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    wheel_pos = _wheel_positions_body(env)
    x_error = torch.abs(wheel_pos[:, 0, 0] - wheel_pos[:, 1, 0])
    z_error = torch.abs(wheel_pos[:, 0, 2] - wheel_pos[:, 1, 2])
    x_penalty = torch.square(
        torch.clamp(x_error - float(x_deadband), min=0.0) / max(float(x_scale), 1.0e-6)
    )
    z_penalty = torch.square(
        torch.clamp(z_error - float(z_deadband), min=0.0) / max(float(z_scale), 1.0e-6)
    )
    return torch.clamp(x_penalty + z_penalty, max=float(max_penalty)), x_error, z_error


def wheel_axle_geometry_penalty(
    env: ManagerBasedRlEnv,
    x_deadband: float = 0.005,
    z_deadband: float = 0.003,
    x_scale: float = 0.02,
    z_scale: float = 0.015,
    max_penalty: float = 16.0,
) -> torch.Tensor:
    """惩罚左右轮心的前后错位与高度差，保持物理轮轴共线。"""
    penalty, x_error, z_error = _wheel_geometry_values(
        env,
        x_deadband,
        z_deadband,
        x_scale,
        z_scale,
        max_penalty,
    )
    if should_log_diagnostics(env, 64, attr_name="_se3_reward_log_interval_steps"):
        env.extras.setdefault("log", {}).update(
            {
                "Locomotion/flat_ly_wheel_axle_x_error_m": x_error.mean().item(),
                "Locomotion/flat_ly_wheel_axle_z_error_m": z_error.mean().item(),
                "Locomotion/flat_ly_wheel_axle_penalty": penalty.mean().item(),
            }
        )
    return penalty


def _wheel_load_values(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float,
    deadband: float,
    scale: float,
    max_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force is None:
        zeros = torch.zeros(env.num_envs, device=env.device)
        return zeros, zeros, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    force = torch.nan_to_num(sensor.data.force, nan=0.0, posinf=5000.0, neginf=-5000.0)
    normal_force = torch.abs(force[:, :, 2])
    both_contact = (normal_force > float(force_threshold)).all(dim=1)
    asymmetry = torch.abs(normal_force[:, 0] - normal_force[:, 1]) / (
        normal_force[:, 0] + normal_force[:, 1] + 1.0
    )
    penalty = torch.square(
        torch.clamp(asymmetry - float(deadband), min=0.0) / max(float(scale), 1.0e-6)
    )
    penalty = torch.clamp(penalty, max=float(max_penalty)) * both_contact.float()
    return penalty, asymmetry, both_contact


def wheel_load_balance_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    force_threshold: float = 1.0,
    deadband: float = 0.10,
    scale: float = 0.25,
    max_penalty: float = 9.0,
) -> torch.Tensor:
    """惩罚两轮接地时的归一化承载差，消除长期单轮偏载。"""
    penalty, asymmetry, both_contact = _wheel_load_values(
        env,
        sensor_name,
        force_threshold,
        deadband,
        scale,
        max_penalty,
    )
    if should_log_diagnostics(env, 64, attr_name="_se3_reward_log_interval_steps"):
        env.extras.setdefault("log", {}).update(
            {
                "Locomotion/flat_ly_wheel_load_asymmetry": asymmetry.mean().item(),
                "Locomotion/flat_ly_wheel_load_penalty": penalty.mean().item(),
                "Locomotion/flat_ly_both_wheel_contact": both_contact.float().mean().item(),
            }
        )
    return penalty


def _rolling_values(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    wheel_radius: float,
    half_track: float,
    force_threshold: float,
    deadband: float,
    scale: float,
    max_penalty: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    robot = env.scene["robot"]
    wheel_vel = robot.data.joint_vel[:, wheel_joint_ids(robot)]
    wheel_forward_speed = torch.stack(
        (wheel_vel[:, 0] * float(wheel_radius), -wheel_vel[:, 1] * float(wheel_radius)),
        dim=1,
    )
    base_vx = robot.data.root_link_lin_vel_b[:, 0]
    yaw_rate = robot.data.root_link_ang_vel_b[:, 2]
    expected_wheel_speed = torch.stack(
        (
            base_vx - float(half_track) * yaw_rate,
            base_vx + float(half_track) * yaw_rate,
        ),
        dim=1,
    )
    rolling_error = torch.abs(wheel_forward_speed - expected_wheel_speed)

    sensor: ContactSensor = env.scene[sensor_name]
    if sensor.data.force is None:
        contact = torch.zeros_like(rolling_error, dtype=torch.bool)
    else:
        contact = finite_contact_force_norm(sensor.data.force) > float(force_threshold)
    per_wheel = torch.square(
        torch.clamp(rolling_error - float(deadband), min=0.0) / max(float(scale), 1.0e-6)
    )
    contact_count = contact.float().sum(dim=1).clamp_min(1.0)
    penalty = (per_wheel * contact.float()).sum(dim=1) / contact_count
    penalty = torch.clamp(penalty, max=float(max_penalty))
    return penalty, rolling_error.mean(dim=1), contact.all(dim=1)


def wheel_rolling_consistency_penalty(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    wheel_radius: float = 0.059,
    half_track: float = 0.21665,
    force_threshold: float = 1.0,
    deadband: float = 0.15,
    scale: float = 0.50,
    max_penalty: float = 9.0,
) -> torch.Tensor:
    """按差速轮刚体运动学惩罚轮面速度误差，兼容直行和原地转向。"""
    penalty, error, both_contact = _rolling_values(
        env,
        sensor_name,
        wheel_radius,
        half_track,
        force_threshold,
        deadband,
        scale,
        max_penalty,
    )
    if should_log_diagnostics(env, 64, attr_name="_se3_reward_log_interval_steps"):
        env.extras.setdefault("log", {}).update(
            {
                "Locomotion/flat_ly_rolling_error_mps": error.mean().item(),
                "Locomotion/flat_ly_rolling_penalty": penalty.mean().item(),
                "Locomotion/flat_ly_rolling_both_contact": both_contact.float().mean().item(),
            }
        )
    return penalty


def leg_differential_action_penalty(
    env: ManagerBasedRlEnv,
    base_deadband: float = 0.10,
    roll_allowance_gain: float = 2.0,
    yaw_allowance_gain: float = 0.25,
    max_penalty: float = 16.0,
) -> torch.Tensor:
    """限制长期左右腿差分动作，但按 roll/yaw 扰动自适应放宽。"""
    action = env.action_manager.action
    robot = env.scene["robot"]
    front_diff = wrap_front_action_value_delta_torch(action[:, 0] - action[:, 2])
    back_diff = action[:, 1] - action[:, 3]
    projected_gravity = robot.data.projected_gravity_b
    roll = torch.asin(torch.clamp(-projected_gravity[:, 1], -1.0, 1.0))
    yaw_rate = robot.data.root_link_ang_vel_b[:, 2]
    allowance = (
        float(base_deadband)
        + float(roll_allowance_gain) * torch.abs(roll)
        + float(yaw_allowance_gain) * torch.abs(yaw_rate)
    )
    front_excess = torch.clamp(torch.abs(front_diff) - allowance, min=0.0)
    back_excess = torch.clamp(torch.abs(back_diff) - allowance, min=0.0)
    penalty = torch.clamp(front_excess**2 + back_excess**2, max=float(max_penalty))
    if should_log_diagnostics(env, 64, attr_name="_se3_reward_log_interval_steps"):
        env.extras.setdefault("log", {}).update(
            {
                "Locomotion/flat_ly_leg_action_diff": (
                    0.5 * (torch.abs(front_diff) + torch.abs(back_diff))
                )
                .mean()
                .item(),
                "Locomotion/flat_ly_leg_diff_penalty": penalty.mean().item(),
            }
        )
    return penalty


def physical_stability_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
    velocity_sigma: float = 0.03,
    yaw_score_base_scale: float = 0.15,
    yaw_score_relative_scale: float = 0.0,
) -> torch.Tensor:
    """返回物理稳定分数，并输出供课程使用的最弱项综合分。"""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)
    velocity_error = robot.data.root_link_lin_vel_b[:, 0] - command[:, 0]
    velocity_score = torch.exp(-(velocity_error**2) / max(float(velocity_sigma), 1.0e-6))
    yaw_rate = robot.data.root_link_ang_vel_b[:, 2]
    yaw_rate_error = yaw_rate - command[:, 1]
    yaw_scale = float(yaw_score_base_scale) + float(yaw_score_relative_scale) * torch.abs(
        command[:, 1]
    )
    yaw_excess = torch.clamp(torch.abs(yaw_rate_error) - 0.01, min=0.0)
    yaw_penalty = torch.clamp(
        torch.square(yaw_excess / torch.clamp(yaw_scale, min=1.0e-6)),
        max=9.0,
    )
    yaw_score = torch.exp(-yaw_penalty)
    lateral_velocity = robot.data.root_link_lin_vel_b[:, 1]
    lateral_penalty = _scaled_deadband_square(
        lateral_velocity,
        deadband=0.03,
        scale=0.20,
        max_penalty=9.0,
    )
    lateral_score = torch.exp(-lateral_penalty)

    orientation = _orientation_state(env, command_name, 0.5, 3.0)
    pitch_penalty = _scaled_deadband_square(
        orientation["pitch"] - orientation["pitch_ref"],
        deadband=math.radians(1.5),
        scale=math.radians(4.0),
        max_penalty=9.0,
    )
    roll_penalty = _scaled_deadband_square(
        orientation["roll"],
        deadband=math.radians(2.0),
        scale=math.radians(5.0),
        max_penalty=9.0,
    )
    orientation_score = torch.exp(-(pitch_penalty + roll_penalty))

    geometry_penalty, _, _ = _wheel_geometry_values(env, 0.005, 0.003, 0.02, 0.015, 16.0)
    geometry_score = torch.exp(-geometry_penalty)

    load_penalty, _, load_contact = _wheel_load_values(
        env,
        sensor_name,
        1.0,
        0.10,
        0.25,
        9.0,
    )
    load_score = torch.exp(-load_penalty) * load_contact.float()

    rolling_penalty, _, rolling_contact = _rolling_values(
        env,
        sensor_name,
        0.059,
        0.21665,
        1.0,
        0.15,
        0.50,
        9.0,
    )
    rolling_score = torch.exp(-rolling_penalty) * rolling_contact.float()

    components = torch.stack(
        (
            velocity_score,
            orientation_score,
            yaw_score,
            lateral_score,
            geometry_score,
            load_score,
            rolling_score,
        ),
        dim=1,
    )
    reward = components.mean(dim=1)
    per_env_course_score = components.min(dim=1).values

    # 课程使用分桶最弱项，避免运动样本掩盖静站或扰动恢复失败。
    standing_mask = _standing_mask(env, command_name)
    recovery_mask = recovery_state.recovery_episode_mask(env)

    zero_speed_penalty = _scaled_deadband_square(
        robot.data.root_link_lin_vel_b[:, 0],
        deadband=0.03,
        scale=0.15,
        max_penalty=9.0,
    )
    zero_speed_score = torch.exp(-zero_speed_penalty)
    _, standing_stationkeeping_score, _ = _standing_stationkeeping_values(env, command_name)
    standing_per_env = (
        torch.stack(
            (
                standing_stationkeeping_score,
                load_score,
                rolling_score,
            ),
            dim=1,
        )
        .min(dim=1)
        .values
    )
    recovery_per_env = (
        torch.stack(
            (
                zero_speed_score,
                orientation_score,
                yaw_score,
                lateral_score,
                load_score,
            ),
            dim=1,
        )
        .min(dim=1)
        .values
    )

    base_course_mean = per_env_course_score.mean()
    standing_course_mean = (
        standing_per_env[standing_mask].mean() if standing_mask.any() else base_course_mean
    )
    recovery_course_mean = (
        recovery_per_env[recovery_mask].mean() if recovery_mask.any() else base_course_mean
    )
    yaw_command_mask = torch.abs(command[:, 1]) > 0.03
    yaw_course_mean = (
        per_env_course_score[yaw_command_mask].mean()
        if yaw_command_mask.any()
        else torch.zeros((), device=env.device)
    )
    yaw_positive_mask = command[:, 1] > 0.03
    yaw_negative_mask = command[:, 1] < -0.03
    yaw_positive_mean = (
        per_env_course_score[yaw_positive_mask].mean()
        if yaw_positive_mask.any()
        else torch.zeros((), device=env.device)
    )
    yaw_negative_mean = (
        per_env_course_score[yaw_negative_mask].mean()
        if yaw_negative_mask.any()
        else torch.zeros((), device=env.device)
    )
    course_score = torch.stack((base_course_mean, standing_course_mean, recovery_course_mean)).min()

    if should_log_diagnostics(env, 64, attr_name="_se3_reward_log_interval_steps"):
        env.extras.setdefault("log", {}).update(
            {
                "Locomotion/flat_ly_physical_stability_reward": reward.mean().item(),
                "Locomotion/flat_ly_course_score": course_score.item(),
                "Locomotion/flat_ly_course_orientation_score": orientation_score.mean().item(),
                "Locomotion/flat_ly_course_yaw_score": yaw_score.mean().item(),
                "Locomotion/flat_ly_course_lateral_score": lateral_score.mean().item(),
                "Locomotion/flat_ly_course_geometry_score": geometry_score.mean().item(),
                "Locomotion/flat_ly_course_load_score": load_score.mean().item(),
                "Locomotion/flat_ly_course_rolling_score": rolling_score.mean().item(),
                "Locomotion/flat_ly_course_standing_score": standing_course_mean.item(),
                "Locomotion/flat_ly_course_recovery_score": recovery_course_mean.item(),
                "Locomotion/flat_ly_course_yaw_tracking_score": yaw_course_mean.item(),
                "Locomotion/flat_ly_course_yaw_positive_score": yaw_positive_mean.item(),
                "Locomotion/flat_ly_course_yaw_negative_score": yaw_negative_mean.item(),
                "Locomotion/flat_ly_standing_sample_ratio": standing_mask.float().mean().item(),
                "Locomotion/flat_ly_recovery_sample_ratio": recovery_mask.float().mean().item(),
                "Locomotion/flat_ly_yaw_command_sample_ratio": yaw_command_mask.float()
                .mean()
                .item(),
                "Locomotion/flat_ly_straight_command_sample_ratio": (
                    (torch.abs(command[:, 0]) > 0.03) & ~yaw_command_mask
                )
                .float()
                .mean()
                .item(),
                "Locomotion/flat_ly_yaw_only_command_sample_ratio": (
                    (torch.abs(command[:, 0]) <= 0.03) & yaw_command_mask
                )
                .float()
                .mean()
                .item(),
            }
        )
    return reward


def pitch_angle_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回机身 pitch 相对零角度的平方误差。"""
    projected_gravity = env.scene["robot"].data.projected_gravity_b
    pitch = torch.asin(torch.clamp(projected_gravity[:, 0], -1.0, 1.0))
    return torch.square(pitch)


def roll_angle_penalty(env: ManagerBasedRlEnv) -> torch.Tensor:
    """返回机身 roll 相对零角度的平方误差。"""
    projected_gravity = env.scene["robot"].data.projected_gravity_b
    roll = torch.asin(torch.clamp(-projected_gravity[:, 1], -1.0, 1.0))
    return torch.square(roll)


def upright_orientation_reward(
    env: ManagerBasedRlEnv,
    sigma: float = 0.05,
) -> torch.Tensor:
    """返回 pitch/roll 接近零时的指数直立奖励。"""
    projected_gravity = env.scene["robot"].data.projected_gravity_b
    pitch = torch.asin(torch.clamp(projected_gravity[:, 0], -1.0, 1.0))
    roll = torch.asin(torch.clamp(-projected_gravity[:, 1], -1.0, 1.0))
    error_sq = torch.square(pitch) + torch.square(roll)
    reward = torch.exp(-error_sq / max(float(sigma), 1.0e-6))

    if should_log_diagnostics(
        env,
        64,
        attr_name="_se3_reward_log_interval_steps",
    ):
        log = env.extras.setdefault("log", {})
        if isinstance(log, dict):
            log["Locomotion/flat_ly_upright_score"] = reward.mean().item()

    return reward


def velocity_tracking_reward(
    env: ManagerBasedRlEnv,
    command_name: str,
    score_sigma: float = 0.03,
) -> torch.Tensor:
    """返回前后速度指数奖励，并输出供速度课程使用的平均分。"""
    robot = env.scene["robot"]
    command = env.command_manager.get_command(command_name)

    lin_error = robot.data.root_link_lin_vel_b[:, 0] - command[:, 0]
    lin_error_sq = torch.square(lin_error)

    jump_flag = (
        command[:, 5] > 0.5
        if command.shape[1] > 5
        else torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    )
    locomotion = ~jump_flag

    sigma = max(float(score_sigma), 1.0e-6)
    score = torch.exp(-lin_error_sq / sigma)
    reward = score * locomotion.float()

    if should_log_diagnostics(
        env,
        64,
        attr_name="_se3_reward_log_interval_steps",
    ):
        log = env.extras.setdefault("log", {})
        if isinstance(log, dict):
            moving = (torch.abs(command[:, 0]) > 0.0) & locomotion
            score_mask = moving if moving.any() else locomotion
            score_mean = score[score_mask].mean().item() if score_mask.any() else 0.0
            error_sq_mean = lin_error_sq[score_mask].mean().item() if score_mask.any() else 0.0
            log.update(
                {
                    "Locomotion/flat_ly_velocity_error_sq": error_sq_mean,
                    "Locomotion/flat_ly_velocity_score": score_mean,
                    "Locomotion/tracking_lin_vel_reward_all": score_mean,
                }
            )

    return reward


def configure_rewards(
    cfg: ManagerBasedRlEnvCfg,
    *,
    phase: Literal["base", "stand", "turn", "arc"] = "base",
) -> None:
    """按训练阶段配置静站、原地转向和弧线转弯奖励。"""
    cfg.rewards.clear()
    cfg.rewards.update(
        {
            # 只在指令斜坡期间允许小幅 pitch 前馈，匀速期回到零。
            "dynamic_pitch": RewardTermCfg(
                func=dynamic_pitch_penalty,
                weight=-2.0,
                params={
                    "command_name": "velocity_height",
                    "max_accel": 0.5,
                    "max_pitch_deg": 3.0,
                    "deadband_deg": 1.5,
                    "scale_deg": 4.0,
                    "max_penalty": 9.0,
                },
            ),
            "roll_stability": RewardTermCfg(
                func=roll_stability_penalty,
                weight=-2.0,
                params={
                    "command_name": "velocity_height",
                    "deadband_deg": 2.0,
                    "scale_deg": 5.0,
                    "max_penalty": 9.0,
                },
            ),
            "ang_vel_xy": RewardTermCfg(
                func=mdp_rewards.ang_vel_xy,
                weight=-0.2,
            ),
            "yaw_rate_tracking": RewardTermCfg(
                func=yaw_rate_tracking_penalty,
                weight=-2.0,
                params={
                    "command_name": "velocity_height",
                    "deadband": 0.01,
                    "scale": 0.15,
                    "max_penalty": 9.0,
                },
            ),
            "lateral_velocity": RewardTermCfg(
                func=lateral_velocity_penalty,
                weight=-1.0,
                params={
                    "deadband": 0.03,
                    "scale": 0.20,
                    "max_penalty": 9.0,
                },
            ),
            "velocity_tracking_reward": RewardTermCfg(
                func=velocity_tracking_reward,
                weight=5.0,
                params={
                    "command_name": "velocity_height",
                    "score_sigma": 0.03,
                },
            ),
            "command_velocity_error": RewardTermCfg(
                func=mdp_rewards.command_velocity_error,
                weight=-5.0,
                params={
                    "command_name": "velocity_height",
                    "lin_vel_scale": 0.5,
                    "yaw_vel_scale": 0.25,
                    "lin_deadband": 0.05,
                    "yaw_deadband": 0.01,
                    "max_penalty": 15.0,
                },
            ),
            # 静站样本单独约束机身零速和水平姿态，不惩罚平衡所需的瞬时轮速。
            "standing_stationkeeping_penalty": RewardTermCfg(
                func=standing_stationkeeping_penalty,
                weight=-2.0,
                params={"command_name": "velocity_height"},
            ),
            "standing_stationkeeping_reward": RewardTermCfg(
                func=standing_stationkeeping_reward,
                weight=3.0,
                params={"command_name": "velocity_height"},
            ),
            # 正奖励同时考虑速度、动态姿态、轮轴、承载和滚动一致性。
            "physical_stability": RewardTermCfg(
                func=physical_stability_reward,
                weight=3.0,
                params={
                    "command_name": "velocity_height",
                    "sensor_name": "wheel_sensor",
                    "velocity_sigma": 0.03,
                },
            ),
            "wheel_axle_geometry": RewardTermCfg(
                func=wheel_axle_geometry_penalty,
                weight=-2.0,
                params={
                    "x_deadband": 0.005,
                    "z_deadband": 0.003,
                    "x_scale": 0.02,
                    "z_scale": 0.015,
                    "max_penalty": 16.0,
                },
            ),
            "wheel_load_balance": RewardTermCfg(
                func=wheel_load_balance_penalty,
                weight=-1.0,
                params={
                    "sensor_name": "wheel_sensor",
                    "force_threshold": 1.0,
                    "deadband": 0.10,
                    "scale": 0.25,
                    "max_penalty": 9.0,
                },
            ),
            "wheel_rolling_consistency": RewardTermCfg(
                func=wheel_rolling_consistency_penalty,
                weight=-0.5,
                params={
                    "sensor_name": "wheel_sensor",
                    "wheel_radius": 0.059,
                    "half_track": 0.21665,
                    "force_threshold": 1.0,
                    "deadband": 0.15,
                    "scale": 0.50,
                    "max_penalty": 9.0,
                },
            ),
            "leg_differential_action": RewardTermCfg(
                func=leg_differential_action_penalty,
                weight=-0.2,
                params={
                    "base_deadband": 0.10,
                    "roll_allowance_gain": 2.0,
                    "yaw_allowance_gain": 0.25,
                    "max_penalty": 16.0,
                },
            ),
            # 内部效率项阻止策略用大扭矩和高频腿动作维持取巧姿态。
            "leg_torques": RewardTermCfg(
                func=mdp_rewards.leg_torques,
                weight=-2.0e-4,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "leg_power": RewardTermCfg(
                func=mdp_rewards.leg_power,
                weight=-1.0e-4,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "action_rate": RewardTermCfg(
                func=mdp_rewards.action_rate,
                weight=-0.1,
            ),
            "action_smoothness": RewardTermCfg(
                func=mdp_rewards.action_smoothness,
                weight=-0.01,
                params={"command_name": "velocity_height"},
            ),
            "dof_pos_limits": RewardTermCfg(
                func=mdp_rewards.dof_pos_limits,
                weight=-5.0,
                params={"asset_cfg": SceneEntityCfg("robot")},
            ),
            "leg_ground_contact": RewardTermCfg(
                func=mdp_rewards.flat_leg_contact_penalty,
                weight=-25.0,
                params={
                    "command_name": "velocity_height",
                    "sensor_name": "leg_contact_sensor",
                    "force_threshold": 1.0,
                },
            ),
            "wheel_airborne": RewardTermCfg(
                func=mdp_rewards.flat_wheel_contact_penalty,
                weight=-10.0,
                params={
                    "command_name": "velocity_height",
                    "sensor_name": "wheel_sensor",
                    "force_threshold": 1.0,
                },
            ),
            "base_height": RewardTermCfg(
                func=mdp_rewards.flat_base_height_penalty_no_jump,
                weight=-4.0,
                params={
                    "command_name": "velocity_height",
                    "height_sensor_name": "base_height_sensor",
                    "sigma": 0.05,
                },
            ),
            "is_alive": RewardTermCfg(
                func=mdp_rewards.is_alive,
                weight=1.0,
            ),
            "base_ground_contact": RewardTermCfg(
                func=mdp_rewards.collision,
                weight=-16.0,
                params={
                    "sensor_name": "collision_sensor",
                    "asset_cfg": SceneEntityCfg("robot"),
                    "use_recovery_gate": False,
                },
            ),
        }
    )

    if phase in {"stand", "turn", "arc"}:
        cfg.rewards["standing_low_frequency_drift"] = RewardTermCfg(
            func=standing_low_frequency_drift_penalty,
            weight=-5.0 if phase == "stand" else -2.0,
            params={
                "command_name": "velocity_height",
                "tau_s": 0.75,
                "base_deadband": 0.01,
                "base_scale": 0.08,
                "wheel_deadband": 0.03,
                "wheel_scale": 0.12,
                "wheel_weight": 0.25,
            },
        )

    if phase == "stand":
        cfg.rewards["standing_stationkeeping_penalty"].weight = -3.0
        cfg.rewards["standing_stationkeeping_reward"].weight = 5.0

    if phase in {"turn", "arc"}:
        cfg.rewards["yaw_rate_tracking"] = RewardTermCfg(
            func=yaw_rate_huber_penalty,
            weight=-2.0,
            params={
                "command_name": "velocity_height",
                "deadband": 0.03,
                "base_scale": 0.5,
                "relative_scale": 0.05,
            },
        )
        cfg.rewards["yaw_rate_precision"] = RewardTermCfg(
            func=yaw_rate_precision_reward,
            weight=4.0,
            params={
                "command_name": "velocity_height",
                "base_scale": 0.2,
                "relative_scale": 0.08,
            },
        )
        cfg.rewards["yaw_rate_progress"] = RewardTermCfg(
            func=yaw_rate_progress_reward,
            weight=2.0,
            params={"command_name": "velocity_height", "min_command": 0.05},
        )
        cfg.rewards["in_place_translation"] = RewardTermCfg(
            func=in_place_translation_penalty,
            weight=-1.0,
            params={
                "command_name": "velocity_height",
                "max_linear_command": 0.05,
                "min_yaw_command": 0.1,
                "velocity_scale": 0.25,
            },
        )
        cfg.rewards["command_velocity_error"].weight = -2.0
        cfg.rewards["command_velocity_error"].params.update(
            {"yaw_vel_scale": 4.0, "max_penalty": 25.0}
        )
        cfg.rewards["physical_stability"].params.update(
            {"yaw_score_base_scale": 0.2, "yaw_score_relative_scale": 0.08}
        )
