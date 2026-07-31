"""flat_ly 学习任务的环境组装入口。"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from se3_train.tasks.flat.env_cfg import env_cfg as flat_env_cfg

from . import actions, commands, curriculums, events, observations, rewards, terminations


def env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """组装 flat_ly 环境，并把各类 MDP 配置交给本目录的学习接口。"""
    cfg = flat_env_cfg(play=play)

    observations.configure_observations(cfg, play=play)
    actions.configure_actions(cfg)
    commands.configure_commands(cfg, play=play)
    rewards.configure_rewards(cfg)
    terminations.configure_terminations(cfg, play=play)
    curriculums.configure_curriculums(cfg, play=play)
    events.configure_events(cfg, play=play)

    return cfg
