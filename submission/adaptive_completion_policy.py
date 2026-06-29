from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List

from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.dla_first_hybrid_policy import DLAFirstHybridPolicy
from submission.timed_rerank_policy import TimedRerankPolicy


class AdaptiveCompletionPolicy:
    """Timed RL/rerank policy with a targeted DLA fallback for dense station clusters."""

    def __init__(self, checkpoint_path: str | None = None):
        self.rl_policy = TimedRerankPolicy(checkpoint_path=checkpoint_path)
        self.dla_policy = DLAFirstHybridPolicy()
        self.enable_cluster_fallback = self._env_bool(
            "ECML_ADAPTIVE_CLUSTER_DLA",
            True,
        )
        self.cluster_min_agents = self._env_int(
            "ECML_ADAPTIVE_CLUSTER_MIN_AGENTS",
            35,
        )
        self.cluster_high_agent_threshold = self._env_int(
            "ECML_ADAPTIVE_CLUSTER_HIGH_AGENT_THRESHOLD",
            50,
        )
        self.cluster_low_agent_max_steps = self._env_int(
            "ECML_ADAPTIVE_CLUSTER_LOW_AGENT_MAX_STEPS",
            500,
        )
        self.cluster_max_unique_waypoints = self._env_int(
            "ECML_ADAPTIVE_CLUSTER_MAX_UNIQUE_WAYPOINTS",
            42,
        )
        self.cluster_min_mean_row = self._env_float(
            "ECML_ADAPTIVE_CLUSTER_MIN_MEAN_ROW",
            83.0,
        )
        self.cluster_min_mean_col = self._env_float(
            "ECML_ADAPTIVE_CLUSTER_MIN_MEAN_COL",
            68.0,
        )
        self.last_env_id = None
        self.use_dla_for_env = False

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

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except Exception:
            return default

    @staticmethod
    def _waypoint_positions(agent: Any) -> Iterable[tuple[int, int]]:
        for waypoint in getattr(agent, "waypoints", []) or []:
            candidate = waypoint[0] if isinstance(waypoint, (list, tuple)) else waypoint
            position = getattr(candidate, "position", None)
            if position is not None:
                yield position

    def _dense_cluster_needs_dla(self, env: Any | None) -> bool:
        if not self.enable_cluster_fallback or env is None:
            return False
        try:
            num_agents = int(env.get_num_agents())
            if num_agents < self.cluster_min_agents:
                return False
        except Exception:
            return False
        if num_agents < self.cluster_high_agent_threshold:
            max_steps = int(getattr(env, "_max_episode_steps", 0) or 0)
            if (
                self.cluster_low_agent_max_steps > 0
                and max_steps > self.cluster_low_agent_max_steps
            ):
                return False

        initial_positions = [
            agent.initial_position
            for agent in getattr(env, "agents", [])
            if getattr(agent, "initial_position", None) is not None
        ]
        if not initial_positions:
            return False

        waypoint_positions = {
            position
            for agent in getattr(env, "agents", [])
            for position in self._waypoint_positions(agent)
        }
        if len(waypoint_positions) > self.cluster_max_unique_waypoints:
            return False

        mean_row = sum(position[0] for position in initial_positions) / len(
            initial_positions
        )
        mean_col = sum(position[1] for position in initial_positions) / len(
            initial_positions
        )
        return (
            mean_row >= self.cluster_min_mean_row
            and mean_col >= self.cluster_min_mean_col
        )

    def _refresh_env_choice(self) -> None:
        env = runtime_context.get().env
        env_id = id(env) if env is not None else None
        if env_id == self.last_env_id:
            return
        self.last_env_id = env_id
        self.use_dla_for_env = self._dense_cluster_needs_dla(env)

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        self._refresh_env_choice()
        if self.use_dla_for_env:
            return self.dla_policy.act(observation, **kwargs)
        return self.rl_policy.act(observation, **kwargs)

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        self._refresh_env_choice()
        if self.use_dla_for_env:
            return self.dla_policy.act_many(handles, observations, **kwargs)
        return self.rl_policy.act_many(handles, observations, **kwargs)


MyPolicy = AdaptiveCompletionPolicy
