from __future__ import annotations

import os
from typing import Any, Dict, List

from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.rerank_policy import RerankPolicy
from submission.risk_veto_policy import RiskVetoPolicy
from submission.sequence_success_policy import DEFAULT_CANDIDATE_CHECKPOINT_PATHS


class FastCompetitionPolicy:
    """Fast default submission policy with an optional local RiskVeto fallback.

    The RiskVeto stack is useful for our mined level-0 failure cases but is too
    expensive and too tailored for unknown official level-1 scenarios. This
    wrapper keeps the deployment path fast by using the PPO/BC rerank policy by
    default and only instantiates RiskVeto when explicitly enabled for known
    local sampled scenes.
    """

    LOCAL_SCENES = {"scene_1", "scene_2", "scene_3", "scene_4", "scene_5"}

    def __init__(self, checkpoint_path: str | None = None):
        fast_checkpoint = (
            checkpoint_path
            or os.environ.get("ECML_FAST_POLICY_CHECKPOINT", "").strip()
            or os.environ.get("ECML_RISK_VETO_CANDIDATE_CHECKPOINT", "").strip()
            or DEFAULT_CANDIDATE_CHECKPOINT_PATHS[-1]
        )
        self.fast_policy = RerankPolicy(checkpoint_path=fast_checkpoint)
        self.risk_policy: RiskVetoPolicy | None = None
        self.use_risk_on_local_scenes = self._env_bool(
            "ECML_FAST_USE_RISK_ON_LOCAL_SCENES",
            False,
        )
        self.local_max_agents = self._env_int("ECML_FAST_LOCAL_MAX_AGENTS", 6)
        self.local_max_episode_steps = self._env_int(
            "ECML_FAST_LOCAL_MAX_EPISODE_STEPS",
            900,
        )

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, str(default)))
        except Exception:
            return default

    def _risk_enabled_for_context(self) -> bool:
        if not self.use_risk_on_local_scenes:
            return False
        context = runtime_context.get()
        if context.scene not in self.LOCAL_SCENES:
            return False
        env = context.env
        if env is None:
            return False
        try:
            if int(env.get_num_agents()) > self.local_max_agents:
                return False
            max_episode_steps = int(getattr(env, "_max_episode_steps", 0) or 0)
            if max_episode_steps > self.local_max_episode_steps:
                return False
        except Exception:
            return False
        return True

    def _risk(self) -> RiskVetoPolicy:
        if self.risk_policy is None:
            self.risk_policy = RiskVetoPolicy()
        return self.risk_policy

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        # Single-agent calls are uncommon in the competition runner; keep them
        # on the fast path because RiskVeto only implements batched decisions.
        return self.fast_policy.act(observation, **kwargs)

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        if self._risk_enabled_for_context():
            return self._risk().act_many(handles, observations, **kwargs)
        return self.fast_policy.act_many(handles, observations, **kwargs)


MyPolicy = FastCompetitionPolicy
