"""flat_ly 动作配置学习接口。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg


def configure_actions(cfg: ManagerBasedRlEnvCfg) -> None:
    """让训练端动作零点与 sim2sim 的高度条件腿型保持一致。"""
    action_cfg = cfg.actions["delayed_action"]
    action_cfg.height_conditioned_action_default = True
