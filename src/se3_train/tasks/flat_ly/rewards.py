"""flat_ly 奖励设计入口；具体奖励由学习者完成。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from se3_train.mdp import rewards as mdp_rewards

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


def configure_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    """配置第一阶段静止站立所需的最小奖励集合。"""
    cfg.rewards.clear()
    cfg.rewards.update(
        {
            # 原 flat 的姿态项权重为 -12 * (pitch² + roll²)，这里等价拆成两轴。
            "pitch_angle": RewardTermCfg(
                func=pitch_angle_penalty,
                weight=-30.0,
            ),
            "roll_angle": RewardTermCfg(
                func=roll_angle_penalty,
                weight=-15.0,
            ),
            # 相对目标指令惩罚机身前后速度与偏航角速度误差，允许小范围传感器噪声。
            "command_velocity_error": RewardTermCfg(
                func=mdp_rewards.command_velocity_error,
                weight=-10.0,
                params={
                    "command_name": "velocity_height",
                    "lin_vel_scale": 0.5,
                    "yaw_vel_scale": 1.0,
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
