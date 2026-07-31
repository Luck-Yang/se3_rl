"""flat_ly 奖励设计入口；具体奖励由学习者完成。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.mdp import rewards as mdp_rewards
from se3_train.mdp.diagnostic_logging import should_log_diagnostics

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


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


def configure_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    """配置第一阶段静止站立所需的最小奖励集合。"""
    cfg.rewards.clear()
    cfg.rewards.update(
        {
            # 原 flat 的姿态项权重为 -12 * (pitch² + roll²)，这里等价拆成两轴。
            "pitch_angle": RewardTermCfg(
                func=pitch_angle_penalty,
                weight=-40.0,
            ),
            "roll_angle": RewardTermCfg(
                func=roll_angle_penalty,
                weight=-30.0,
            ),
            "upright_orientation": RewardTermCfg(
                func=upright_orientation_reward,
                weight=5.0,
                params={"sigma": 0.05},
            ),
            # 抑制 pitch/roll 两轴的快速摆动，避免只约束角度却持续振荡。
            "ang_vel_xy": RewardTermCfg(
                func=mdp_rewards.ang_vel_xy,
                weight=-0.5,
            ),
            # 速度接近目标时给正向指数奖励，并把同一分数交给速度课程。
            "velocity_tracking_reward": RewardTermCfg(
                func=velocity_tracking_reward,
                weight=5.0,
                params={
                    "command_name": "velocity_height",
                    "score_sigma": 0.03,
                },
            ),
            # 保留带死区、归一化和上限的速度违令惩罚。
            "command_velocity_error": RewardTermCfg(
                func=mdp_rewards.command_velocity_error,
                weight=-10.0,
                params={
                    "command_name": "velocity_height",
                    "lin_vel_scale": 0.5,
                    "yaw_vel_scale": 0.5,
                    "lin_deadband": 0.05,
                    "yaw_deadband": 0.10,
                    "max_penalty": 15.0,
                },
            ),
            # 任一腿部 Link 触地都扣分。
            "leg_ground_contact": RewardTermCfg(
                func=mdp_rewards.flat_leg_contact_penalty,
                weight=-25.0,
                params={
                    "command_name": "velocity_height",
                    "sensor_name": "leg_contact_sensor",
                    "force_threshold": 1.0,
                },
            ),
            # 左右轮都触地时原始惩罚为 0，单轮或双轮离地时递增。
            "wheel_airborne": RewardTermCfg(
                func=mdp_rewards.flat_wheel_contact_penalty,
                weight=-10.0,
                params={
                    "command_name": "velocity_height",
                    "sensor_name": "wheel_sensor",
                    "force_threshold": 1.0,
                },
            ),
            # 沿用原 flat 的机身高度 L2 惩罚和参数。
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
            # collision_sensor 监测整个 base_link，可覆盖机身前后端触地。
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
