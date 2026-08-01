"""flat_ly reset、startup 与 interval 事件配置学习接口。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def configure_events(cfg: ManagerBasedRlEnvCfg, *, play: bool) -> None:
    """从对称名义姿态起步，再逐步扩大初始关节扰动。"""
    if play:
        return

    reset_joints_cfg = cfg.events.get("reset_joints")
    if reset_joints_cfg is not None:
        reset_joints_cfg.params.update(
            {
                "height_conditioned_default": True,
                "full_joint_randomization": False,
                "joint_offset_range": 0.01,
                "joint_vel_range": (-0.05, 0.05),
                "use_iterations": True,
                "curriculum_stages": [
                    {
                        "iteration": 0,
                        "full_joint_randomization": False,
                        "joint_offset_range": 0.01,
                        "joint_vel_range": (-0.05, 0.05),
                    },
                    {
                        "iteration": 3000,
                        "full_joint_randomization": False,
                        "joint_offset_range": 0.03,
                        "joint_vel_range": (-0.10, 0.10),
                    },
                    {
                        "iteration": 6000,
                        "full_joint_randomization": False,
                        "joint_offset_range": 0.06,
                        "joint_vel_range": (-0.20, 0.20),
                    },
                ],
            }
        )

    # 名义物理阶段仍保留域随机化，但避免宽扰动掩盖基本轮轴动力学。
    randomization_params = {
        "friction": {"friction_range": (0.6, 1.2)},
        "restitution": {"restitution_range": (0.0, 0.1)},
        "base_mass": {"mass_range": (-0.2, 0.5)},
        "inertia": {"inertia_range": (0.95, 1.05)},
        "com": {"com_range": 0.01},
        "pd_gains": {"kp_range": (0.95, 1.05), "kd_range": (0.95, 1.05)},
        "default_dof_pos": {"offset_range": (-0.02, 0.02)},
    }
    for event_name, params in randomization_params.items():
        event_cfg = cfg.events.get(event_name)
        if event_cfg is not None:
            event_cfg.params.update(params)
