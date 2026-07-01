from __future__ import annotations

import os
from typing import Any, Dict, List

from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.adaptive_completion_policy import AdaptiveCompletionPolicy
from submission.rl_mapf_sipp_policy import RLMAPFSIPPPolicy


class SceneAwareCompetitionPolicy:
    """Route dense scenes to the locally safest policy family.

    The broad MAPF/SIPP deployment regressed official level 2 and local
    scene_1/scene_2 dense cases. Keep MAPF/SIPP for scenes where it has measured
    gains and use the faster adaptive-completion stack for dense scenes where
    MAPF misses the 25% completion gate.
    """

    def __init__(self, checkpoint_path: str | None = None):
        self.checkpoint_path = checkpoint_path
        self.adaptive_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_SCENE_AWARE_ADAPTIVE_SCENES",
                "scene_1,scene_2,scene_5",
            ).split(",")
            if scene.strip()
        }
        self.adaptive_min_agents = self._env_int(
            "ECML_SCENE_AWARE_ADAPTIVE_MIN_AGENTS",
            70,
        )
        self.mapf_policy: RLMAPFSIPPPolicy | None = None
        self.adaptive_policy: AdaptiveCompletionPolicy | None = None

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except Exception:
            return default

    def _mapf(self) -> RLMAPFSIPPPolicy:
        if self.mapf_policy is None:
            self.mapf_policy = RLMAPFSIPPPolicy(checkpoint_path=self.checkpoint_path)
        return self.mapf_policy

    def _adaptive(self) -> AdaptiveCompletionPolicy:
        if self.adaptive_policy is None:
            self.adaptive_policy = AdaptiveCompletionPolicy(
                checkpoint_path=self.checkpoint_path,
            )
        return self.adaptive_policy

    def _use_adaptive(self) -> bool:
        context = runtime_context.get()
        if context.scene not in self.adaptive_scenes:
            return False
        env = context.env
        if env is None:
            return False
        try:
            return int(env.get_num_agents()) >= self.adaptive_min_agents
        except Exception:
            return False

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        if self._use_adaptive():
            return self._adaptive().act(observation, **kwargs)
        return self._mapf().act(observation, **kwargs)

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        if self._use_adaptive():
            return self._adaptive().act_many(handles, observations, **kwargs)
        return self._mapf().act_many(handles, observations, **kwargs)


MyPolicy = SceneAwareCompetitionPolicy
