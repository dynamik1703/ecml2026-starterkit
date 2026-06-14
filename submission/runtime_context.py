from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RuntimeContext:
    env: Any | None = None
    obs_builder: Any | None = None
    step: int | None = None
    seed: int | None = None
    masks: dict[int, np.ndarray] = field(default_factory=dict)


_CONTEXT = RuntimeContext()


def update(env: Any, obs_builder: Any, masks: dict[int, np.ndarray]) -> None:
    _CONTEXT.env = env
    _CONTEXT.obs_builder = obs_builder
    _CONTEXT.step = getattr(env, "_elapsed_steps", None)
    _CONTEXT.masks = {
        handle: np.asarray(mask, dtype=np.float32).copy()
        for handle, mask in masks.items()
    }


def set_seed(seed: int | None) -> None:
    _CONTEXT.seed = seed


def get() -> RuntimeContext:
    return _CONTEXT
