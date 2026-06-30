from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List

from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.dla_first_hybrid_policy import DLAFirstHybridPolicy


@dataclass(frozen=True)
class SIPPActionCandidate:
    handle: int
    action_id: int
    source: Any
    target: Any
    direction: Any
    corridor_edges: tuple[tuple[Any, Any], ...]


class RLGuidedSIPPPolicy(DLAFirstHybridPolicy):
    """DLA/SIPP-style safety layer with RL used only for conflict ranking.

    This policy deliberately does not let the learned model emit free actions.
    DLA produces the movement candidates. A lightweight reservation layer then
    finds one-step target conflicts and short opposite-direction corridor
    conflicts. If a conflict group exists, RL action scores are used as a
    tie-breaker together with deadline slack and wait streaks to decide who
    moves and who yields.
    """

    def __init__(self, checkpoint_path: str | None = None):
        super().__init__(checkpoint_path=checkpoint_path)
        self.rl_sipp_enabled = self._env_bool("ECML_RL_SIPP_ENABLED", True)
        self.rl_sipp_min_agents = self._env_int("ECML_RL_SIPP_MIN_AGENTS", 70)
        self.rl_sipp_min_steps = self._env_int("ECML_RL_SIPP_MIN_STEPS", 0)
        self.rl_sipp_lookahead = self._env_int("ECML_RL_SIPP_LOOKAHEAD", 8)
        self.rl_sipp_min_corridor_edges = self._env_int(
            "ECML_RL_SIPP_MIN_CORRIDOR_EDGES",
            100,
        )
        self.rl_sipp_max_waits_per_step = self._env_int(
            "ECML_RL_SIPP_MAX_WAITS_PER_STEP",
            1,
        )
        self.rl_sipp_use_rl_scores = self._env_bool(
            "ECML_RL_SIPP_USE_RL_SCORES",
            True,
        )
        self.rl_sipp_slack_bucket = self._env_float(
            "ECML_RL_SIPP_SLACK_BUCKET",
            25.0,
        )
        self.rl_sipp_rl_weight = self._env_float("ECML_RL_SIPP_RL_WEIGHT", 1.0)
        self.rl_sipp_wait_weight = self._env_float("ECML_RL_SIPP_WAIT_WEIGHT", 0.10)
        self.rl_sipp_corridor_weight = self._env_float(
            "ECML_RL_SIPP_CORRIDOR_WEIGHT",
            0.01,
        )

    def _rl_sipp_should_run(self, env: Any) -> bool:
        if not self.rl_sipp_enabled:
            return False
        try:
            num_agents = int(env.get_num_agents())
            max_steps = int(getattr(env, "_max_episode_steps", 0) or 0)
        except Exception:
            return False
        if num_agents < self.rl_sipp_min_agents:
            return False
        if max_steps < self.rl_sipp_min_steps:
            return False
        return True

    def _sipp_candidate(
        self,
        env: Any,
        handle: int,
        action_id: int,
    ) -> SIPPActionCandidate | None:
        if action_id not in {1, 2, 3}:
            return None
        source, target, direction = self._mini_lock_action_target(
            env,
            handle,
            action_id,
        )
        if source is None or target is None or direction is None or source == target:
            return None
        corridor_edges = tuple(
            self._mini_lock_corridor_edges_with_lookahead(
                env,
                source,
                target,
                direction,
                self.rl_sipp_lookahead,
            )
        )
        return SIPPActionCandidate(
            handle=handle,
            action_id=action_id,
            source=source,
            target=target,
            direction=direction,
            corridor_edges=corridor_edges,
        )

    def _mini_lock_corridor_edges_with_lookahead(
        self,
        env: Any,
        source: Any | None,
        target: Any | None,
        direction: Any | None,
        lookahead: int,
    ) -> list[tuple[Any, Any]]:
        old_lookahead = self.mini_lock_lookahead
        self.mini_lock_lookahead = lookahead
        try:
            return self._mini_lock_corridor_edges(env, source, target, direction)
        finally:
            self.mini_lock_lookahead = old_lookahead

    @staticmethod
    def _reverse(edge: tuple[Any, Any]) -> tuple[Any, Any]:
        return (edge[1], edge[0])

    def _candidates_conflict(
        self,
        left: SIPPActionCandidate,
        right: SIPPActionCandidate,
    ) -> bool:
        if left.target == right.target:
            return True
        if left.target == right.source and right.target == left.source:
            return True
        if (
            len(left.corridor_edges) < self.rl_sipp_min_corridor_edges
            or len(right.corridor_edges) < self.rl_sipp_min_corridor_edges
        ):
            return False
        right_edges = set(right.corridor_edges)
        return any(self._reverse(edge) in right_edges for edge in left.corridor_edges)

    def _conflict_components(
        self,
        candidates: dict[int, SIPPActionCandidate],
    ) -> list[list[int]]:
        handles = list(candidates)
        graph: dict[int, set[int]] = {handle: set() for handle in handles}
        for index, left_handle in enumerate(handles):
            left = candidates[left_handle]
            for right_handle in handles[index + 1 :]:
                right = candidates[right_handle]
                if self._candidates_conflict(left, right):
                    graph[left_handle].add(right_handle)
                    graph[right_handle].add(left_handle)

        components = []
        seen = set()
        for handle in handles:
            if handle in seen or not graph[handle]:
                continue
            queue = deque([handle])
            seen.add(handle)
            component = []
            while queue:
                current = queue.popleft()
                component.append(current)
                for neighbor in graph[current]:
                    if neighbor not in seen:
                        seen.add(neighbor)
                        queue.append(neighbor)
            if len(component) > 1:
                components.append(component)
        return components

    def _rl_advantage(
        self,
        handle: int,
        action_id: int,
        timing_scores: dict[int, dict[str, float]],
        env: Any,
    ) -> float:
        if handle not in timing_scores:
            return 0.0
        agent = env.agents[handle]
        wait_id = self._action_id(self._wait_action_for_agent(agent))
        scores = timing_scores[handle]
        return scores.get(f"score_{action_id}", 0.0) - scores.get(
            f"score_{wait_id}",
            0.0,
        )

    def _component_rank_key(
        self,
        env: Any,
        candidate: SIPPActionCandidate,
        timing_scores: dict[int, dict[str, float]],
    ) -> tuple[float, float, float, float, int]:
        slack, path_len, _ = self._mini_lock_priority(env, candidate.handle)
        slack_key = (
            slack
            if self.rl_sipp_slack_bucket <= 0.0
            else slack // self.rl_sipp_slack_bucket
        )
        wait_streak = float(self.wait_streaks.get(candidate.handle, 0))
        rl_advantage = self._rl_advantage(
            candidate.handle,
            candidate.action_id,
            timing_scores,
            env,
        )
        corridor_len = float(len(candidate.corridor_edges))
        score = (
            self.rl_sipp_rl_weight * rl_advantage
            + self.rl_sipp_wait_weight * wait_streak
            + self.rl_sipp_corridor_weight * corridor_len
        )
        return (slack_key, -score, path_len, -corridor_len, candidate.handle)

    def _apply_rl_guided_sipp(
        self,
        env: Any,
        handles: List[int],
        observations: List[Any],
        actions: Dict[int, RailEnvActions],
    ) -> Dict[int, RailEnvActions]:
        if not self._rl_sipp_should_run(env):
            return actions

        candidates: dict[int, SIPPActionCandidate] = {}
        for handle in handles:
            action_id = self._action_id(actions.get(handle, RailEnvActions.DO_NOTHING))
            candidate = self._sipp_candidate(env, handle, action_id)
            if candidate is not None:
                candidates[handle] = candidate

        components = self._conflict_components(candidates)
        if not components:
            return actions

        timing_scores: dict[int, dict[str, float]] = {}
        if self.rl_sipp_use_rl_scores and len(observations) == len(handles):
            try:
                timing_scores = self._rl_timing_scores(handles, observations)
            except Exception:
                timing_scores = {}

        adjusted = dict(actions)
        waits_added = 0
        for component in components:
            ranked = sorted(
                (candidates[handle] for handle in component),
                key=lambda candidate: self._component_rank_key(
                    env,
                    candidate,
                    timing_scores,
                ),
            )
            winner = ranked[0].handle
            for candidate in ranked[1:]:
                if (
                    self.rl_sipp_max_waits_per_step > 0
                    and waits_added >= self.rl_sipp_max_waits_per_step
                ):
                    break
                agent = env.agents[candidate.handle]
                adjusted[candidate.handle] = self._wait_action_for_agent(agent)
                self.wait_streaks[candidate.handle] = (
                    self.wait_streaks.get(candidate.handle, 0) + 1
                )
                waits_added += 1
            self.wait_streaks[winner] = 0
        return adjusted

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        if not handles:
            return {}

        env = runtime_context.get().env
        if env is not None:
            self._reset_for_env(env)
            actions = self._dla_actions(handles, env)
            if actions:
                actions = self._apply_rl_guided_sipp(
                    env,
                    handles,
                    observations,
                    actions,
                )
                return self._late_rl_rescue_actions(
                    env,
                    handles,
                    observations,
                    actions,
                )

        return {
            handle: RailEnvActions(action)
            for handle, action in zip(handles, self._rl_action_ids(observations))
        }


MyPolicy = RLGuidedSIPPPolicy
