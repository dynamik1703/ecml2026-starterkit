from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.my_policy import ActorCritic
from submission.rerank_policy import RerankPolicy

DEFAULT_CHECKPOINT = "./submission/checkpoint.pt"


class TimedRerankPolicy:
    """Rerank-first RL policy with a per-environment direct-RL timeout guard."""

    def __init__(self, checkpoint_path: str | None = None):
        checkpoint = checkpoint_path or DEFAULT_CHECKPOINT
        self.rerank_policy = RerankPolicy(checkpoint_path=checkpoint)
        self.direct_policy = ActorCritic(checkpoint_path=checkpoint)
        self.direct_policy.eval()
        self.direct_after_seconds = self._env_float(
            "ECML_TIMED_RERANK_DIRECT_AFTER_SECONDS",
            900.0,
        )
        self.last_env_id = None
        self.last_step = None
        self.env_start_time = time.monotonic()

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except Exception:
            return default

    def _reset_timer_if_needed(self) -> None:
        env = runtime_context.get().env
        step = int(getattr(env, "_elapsed_steps", 0) or 0) if env is not None else 0
        env_id = id(env) if env is not None else None
        if env_id != self.last_env_id or (
            self.last_step is not None and step < self.last_step
        ):
            self.last_env_id = env_id
            self.env_start_time = time.monotonic()
        self.last_step = step

    def _use_direct_policy(self) -> bool:
        if self.direct_after_seconds <= 0.0:
            return False
        self._reset_timer_if_needed()
        return (time.monotonic() - self.env_start_time) >= self.direct_after_seconds

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        if self._use_direct_policy():
            return self.direct_policy.act(observation, **kwargs)
        return self.rerank_policy.act(observation, **kwargs)

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        if self._use_direct_policy():
            return self.direct_policy.act_many(handles, observations, **kwargs)
        return self.rerank_policy.act_many(handles, observations, **kwargs)


MyPolicy = TimedRerankPolicy
