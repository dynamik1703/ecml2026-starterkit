from __future__ import annotations

import os
from typing import Any, Dict, List

from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.adaptive_completion_policy import AdaptiveCompletionPolicy
from submission.dla_first_hybrid_policy import DLAFirstHybridPolicy


class HighComplexityDLAPolicy:
    """Adaptive policy with a DLA floor for large hidden-evaluation instances.

    Local sampled scene_5 cases are often better under the learned rerank path,
    while the official high-level hidden cases have shown that missing a DLA
    fallback can fail an entire level. This wrapper keeps the adaptive/RL path
    for moderate instances and forces DLA only for high-complexity environments.
    """

    def __init__(self, checkpoint_path: str | None = None):
        self.adaptive_policy = AdaptiveCompletionPolicy(checkpoint_path=checkpoint_path)
        self.dla_policy = DLAFirstHybridPolicy(checkpoint_path=checkpoint_path)
        self.min_agents = self._env_int("ECML_HIGH_COMPLEXITY_DLA_MIN_AGENTS", 70)
        self.min_steps = self._env_int("ECML_HIGH_COMPLEXITY_DLA_MIN_STEPS", 850)
        self.mid_agents = self._env_int("ECML_HIGH_COMPLEXITY_DLA_MID_AGENTS", 50)
        self.mid_steps = self._env_int("ECML_HIGH_COMPLEXITY_DLA_MID_STEPS", 850)
        self.min_area = self._env_int("ECML_HIGH_COMPLEXITY_DLA_MIN_AREA", 0)
        self.last_env_id = None
        self.use_dla_for_env = False

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except Exception:
            return default

    def _high_complexity_needs_dla(self, env: Any | None) -> bool:
        if env is None:
            return False
        try:
            num_agents = int(env.get_num_agents())
            max_steps = int(getattr(env, "_max_episode_steps", 0) or 0)
            height = int(getattr(env, "height", 0) or 0)
            width = int(getattr(env, "width", 0) or 0)
        except Exception:
            return False
        if self.min_agents > 0 and num_agents >= self.min_agents:
            return True
        if self.min_steps > 0 and max_steps >= self.min_steps:
            return True
        if (
            self.mid_agents > 0
            and self.mid_steps > 0
            and num_agents >= self.mid_agents
            and max_steps >= self.mid_steps
        ):
            return True
        if self.min_area > 0 and height * width >= self.min_area:
            return True
        return False

    def _refresh_env_choice(self) -> None:
        env = runtime_context.get().env
        env_id = id(env) if env is not None else None
        if env_id == self.last_env_id:
            return
        self.last_env_id = env_id
        self.use_dla_for_env = self._high_complexity_needs_dla(env)

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        self._refresh_env_choice()
        if self.use_dla_for_env:
            return self.dla_policy.act(observation, **kwargs)
        return self.adaptive_policy.act(observation, **kwargs)

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        self._refresh_env_choice()
        if self.use_dla_for_env:
            return self.dla_policy.act_many(handles, observations, **kwargs)
        return self.adaptive_policy.act_many(handles, observations, **kwargs)


MyPolicy = HighComplexityDLAPolicy
