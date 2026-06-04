from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.rerank_policy import RerankPolicy
from submission.reservation_policy import ReservationPolicy


class PPOSequenceGatedPolicy(RerankPolicy):
    """Experimental safety wrapper for 64-feature PPO checkpoints.

    The stable guarded rerank policy remains the baseline. A PPO checkpoint is
    queried in parallel and can only replace baseline actions for narrow,
    distance-neutral side-detour cases. This blocks the observed harmful PPO
    pattern where an initially small route deviation is followed by repeated
    STOP_MOVING overrides.
    """

    SIDE_DISTANCE_MARGIN = 1e-6

    def __init__(self, checkpoint_path: str | None = None):
        super().__init__(checkpoint_path="./submission/checkpoint.pt")
        self.candidate_policy = (
            RerankPolicy(checkpoint_path=checkpoint_path)
            if checkpoint_path is not None
            else None
        )

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        baseline_actions = super().act_many(handles, observations, **kwargs)
        if self.candidate_policy is None:
            return baseline_actions

        candidate_actions = self.candidate_policy.act_many(
            handles,
            observations,
            **kwargs,
        )
        context = runtime_context.get()
        env = context.env
        obs_builder = context.obs_builder
        if (
            env is None
            or obs_builder is None
            or env.get_num_agents() < 6
            or max(env.height, env.width) < 100
        ):
            return baseline_actions

        adjusted = dict(baseline_actions)
        planned_prefixes = {
            handle: self._route_prefix_for_action(
                obs_builder,
                handle,
                self._action_id(action),
                self.FUTURE_RERANK_LOOKAHEAD_CELLS,
            )
            for handle, action in adjusted.items()
        }
        reserved_targets: set[tuple[int, int]] = set()
        observations_by_handle = dict(zip(handles, observations))

        for handle in sorted(adjusted, key=lambda h: self._priority_key(obs_builder, h)):
            baseline_action = self._action_id(
                adjusted.get(handle, RailEnvActions.DO_NOTHING)
            )
            candidate_action = self._action_id(
                candidate_actions.get(handle, RailEnvActions.DO_NOTHING)
            )
            action_id = baseline_action

            if (
                baseline_action != candidate_action
                and self._accept_candidate_action(
                    obs_builder,
                    handle,
                    observations_by_handle.get(handle),
                    baseline_action,
                    candidate_action,
                    planned_prefixes,
                    reserved_targets,
                )
            ):
                adjusted[handle] = RailEnvActions(candidate_action)
                action_id = candidate_action
                planned_prefixes[handle] = self._route_prefix_for_action(
                    obs_builder,
                    handle,
                    candidate_action,
                    self.FUTURE_RERANK_LOOKAHEAD_CELLS,
                )

            self._reserve_action_target(
                reserved_targets,
                obs_builder,
                handle,
                action_id,
            )

        return adjusted

    def _accept_candidate_action(
        self,
        obs_builder: Any,
        handle: int,
        observation: Any,
        baseline_action: int,
        candidate_action: int,
        planned_prefixes: dict[int, list[dict[str, Any]]],
        reserved_targets: set[tuple[int, int]],
    ) -> bool:
        if baseline_action != ReservationPolicy.MOVE_FORWARD:
            return False
        if candidate_action not in (
            ReservationPolicy.MOVE_LEFT,
            ReservationPolicy.MOVE_RIGHT,
        ):
            return False
        if not self._mask_allows(observation, candidate_action):
            return False

        if self._current_slack(obs_builder, handle) < 0.0:
            return False
        if (
            candidate_action == ReservationPolicy.MOVE_LEFT
            and not self._candidate_improves_direction(
                obs_builder,
                handle,
                observation,
                baseline_action,
                candidate_action,
            )
        ):
            return False

        baseline_distance = self._target_distance(obs_builder, handle, baseline_action)
        candidate_distance = self._target_distance(obs_builder, handle, candidate_action)
        if not np.isfinite(candidate_distance):
            return False
        if (
            np.isfinite(baseline_distance)
            and candidate_distance > baseline_distance + self.SIDE_DISTANCE_MARGIN
        ):
            return False

        target, _ = obs_builder._action_target(handle, candidate_action)
        if target is None or target in reserved_targets:
            return False
        if obs_builder._occupied_by_other(target, handle):
            return False

        if self._has_prefix_interaction(
            obs_builder,
            handle,
            candidate_action,
            planned_prefixes,
        ):
            return False
        if self._reject_residual_head_on_detour(
            obs_builder,
            handle,
            candidate_action,
            planned_prefixes,
        ):
            return False
        return True

    @staticmethod
    def _mask_allows(observation: Any, action: int) -> bool:
        if observation is None:
            return False
        values = np.asarray(observation, dtype=np.float32)
        if values.shape[0] < 5 or action >= 5:
            return False
        return bool(values[-5 + action] >= 0.5)

    @staticmethod
    def _target_distance(obs_builder: Any, handle: int, action: int) -> float:
        try:
            target, target_direction = obs_builder._action_target(handle, action)
            if target is None or target_direction is None:
                return float("inf")
            distance_map = obs_builder._get_distance_map(handle)
            return float(distance_map[target[0], target[1], target_direction])
        except Exception:
            return float("inf")

    @staticmethod
    def _current_slack(obs_builder: Any, handle: int) -> float:
        try:
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            return float(obs_builder._deadline_slack(handle, distance))
        except Exception:
            return float("-inf")

    @staticmethod
    def _candidate_improves_direction(
        obs_builder: Any,
        handle: int,
        observation: Any,
        baseline_action: int,
        candidate_action: int,
    ) -> bool:
        if observation is None:
            return False
        values = np.asarray(observation, dtype=np.float32)
        if values.shape[0] < 4:
            return False
        try:
            _, baseline_direction = obs_builder._action_target(handle, baseline_action)
            _, candidate_direction = obs_builder._action_target(handle, candidate_action)
        except Exception:
            return False
        if baseline_direction is None or candidate_direction is None:
            return False
        if baseline_direction < 0 or candidate_direction < 0:
            return False
        if baseline_direction >= 4 or candidate_direction >= 4:
            return False
        return bool(values[candidate_direction] > values[baseline_direction])

    def _has_prefix_interaction(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        planned_prefixes: dict[int, list[dict[str, Any]]],
    ) -> bool:
        own_prefix = self._route_prefix_for_action(
            obs_builder,
            handle,
            action,
            self.FUTURE_RERANK_LOOKAHEAD_CELLS,
        )
        if not own_prefix:
            return True

        own_positions = {
            node["position"] for node in own_prefix if node.get("position") is not None
        }
        own_edges = self._prefix_edges(own_prefix)
        for other, other_prefix in planned_prefixes.items():
            if other == handle or not other_prefix:
                continue
            if any(
                node.get("position") in own_positions
                for node in other_prefix
                if node.get("position") is not None
            ):
                return True

            other_edges = self._prefix_edges(other_prefix)
            for source, target in own_edges:
                if (source, target) in other_edges or (target, source) in other_edges:
                    return True

        return False


MyPolicy = PPOSequenceGatedPolicy
