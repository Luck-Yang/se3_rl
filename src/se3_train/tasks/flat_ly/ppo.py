"""flat_ly 专用的有界自适应 PPO。"""

from __future__ import annotations

from typing import Any

import torch
from rsl_rl.algorithms import PPO


class FlatLyBoundedPPO(PPO):
    """限制 adaptive schedule 的学习率，并暴露每轮 PPO 的平均 KL。

    RSL-RL 的 adaptive schedule 在每个 mini-batch 后更新学习率，且上限硬编码为
    ``1e-2``。flat_ly 使用多个 epoch/mini-batch 时，第一次 update 内就可能连续放大
    学习率。这里在每次 optimizer step 前施加任务级上限，避免 model0 的第一轮更新
    破坏刚获得的站立策略。
    """

    max_learning_rate = 2.0e-4

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._clamp_learning_rate()

    def _clamp_learning_rate(self) -> None:
        self.learning_rate = min(float(self.learning_rate), self._learning_rate_ceiling())
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

    def _learning_rate_ceiling(self) -> float:
        return self.max_learning_rate

    def update(self) -> dict[str, float]:
        """执行有界 PPO update，并把框架计算的 KL 加入 loss 日志。"""
        kl_samples: list[torch.Tensor] = []
        original_kl = self.actor.get_kl_divergence
        original_step = self.optimizer.step

        def tracked_kl(
            old_params: tuple[torch.Tensor, ...],
            new_params: tuple[torch.Tensor, ...],
        ) -> torch.Tensor:
            kl = original_kl(old_params, new_params)
            kl_samples.append(kl.detach().mean())
            return kl

        def bounded_step(*args: Any, **kwargs: Any) -> Any:
            self._clamp_learning_rate()
            return original_step(*args, **kwargs)

        # PPO.update() 内部没有 KL hook，也没有可配置的 LR 上限，因此仅在本任务
        # 的算法实例上做局部包装；finally 保证异常时也恢复原对象。
        self.actor.get_kl_divergence = tracked_kl  # type: ignore[method-assign]
        self.optimizer.step = bounded_step  # type: ignore[method-assign]
        try:
            loss_dict = super().update()
        finally:
            self.actor.get_kl_divergence = original_kl  # type: ignore[method-assign]
            self.optimizer.step = original_step  # type: ignore[method-assign]
            self._clamp_learning_rate()

        loss_dict["approx_kl"] = (
            torch.stack(kl_samples).mean().item() if kl_samples else float("nan")
        )
        loss_dict["learning_rate_ceiling"] = self._learning_rate_ceiling()
        return loss_dict


class FlatLyBasePPO(FlatLyBoundedPPO):
    """为随机初始化 Base 的最初几轮增加跨 iteration 学习率 warmup。"""

    warmup_start_learning_rate = 2.0e-5
    warmup_growth = 1.5

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._flat_ly_update_count = 0
        super().__init__(*args, **kwargs)

    def _learning_rate_ceiling(self) -> float:
        warmup_ceiling = self.warmup_start_learning_rate * (
            self.warmup_growth**self._flat_ly_update_count
        )
        return min(self.max_learning_rate, warmup_ceiling)

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        self._flat_ly_update_count += 1
        return loss_dict

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["flat_ly_update_count"] = self._flat_ly_update_count
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self._flat_ly_update_count = int(
                loaded_dict.get("flat_ly_update_count", loaded_dict.get("iter", 0))
            )
            self._clamp_learning_rate()
        return load_iteration


class FlatLyFineTunePPO(FlatLyBoundedPPO):
    """保持 Stand 精修阶段声明的 5e-5 学习率上限。"""

    max_learning_rate = 5.0e-5


__all__ = ["FlatLyBasePPO", "FlatLyBoundedPPO", "FlatLyFineTunePPO"]
