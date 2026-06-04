from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.rerank_policy import RerankPolicy
from submission.reservation_policy import ReservationPolicy


class LogitDetourPolicy(RerankPolicy):
    """Experimental near-tie side-detour policy.

    This is intentionally not the Docker default. It tests a narrow extra
    candidate-generation rule mined from missed side-detour counterfactuals:
    only take a near-tie side action when it is distance-neutral, conflict-free
    across planned route prefixes, and the torch actor assigns it a competitive
    logit.
    """

    NEAR_TIE_SIDE_MAX_LOGIT_LOSS = 0.3
    NEAR_TIE_SIDE_MIN_SLACK = 150.0
    NEAR_TIE_SIDE_DISTANCE_MARGIN = 1e-6

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        context = runtime_context.get()
        env = context.env
        if env is not None and env.get_num_agents() <= 2:
            return self.reservation_policy.act_many(handles, observations, **kwargs)

        actions = self.rl_policy.act_many(handles, observations, **kwargs)
        if (
            env is None
            or context.obs_builder is None
            or env.get_num_agents() < 6
            or max(env.height, env.width) < 100
        ):
            return actions

        adjusted = self._detour_around_long_opposing(actions, context.obs_builder)
        adjusted = self._break_adjacent_deadlocks(adjusted, context.obs_builder)
        adjusted = self._avoid_implicit_forward_deadlocks(adjusted, context.obs_builder)
        adjusted = self._rerank_future_head_on_detours(adjusted, context.obs_builder)
        adjusted = self._rerank_near_tie_side_detours(
            adjusted,
            context.obs_builder,
            handles,
            observations,
        )
        return self._apply_temporal_corridor_locks(
            handles,
            observations,
            adjusted,
            context.obs_builder,
        )

    def _rerank_near_tie_side_detours(
        self,
        actions: Dict[int, RailEnvActions],
        obs_builder: Any,
        handles: List[int],
        observations: List[Any],
    ) -> Dict[int, RailEnvActions]:
        if not handles:
            return actions

        logits_by_handle = self._logits_by_handle(handles, observations)
        adjusted = dict(actions)
        planned_prefixes = {
            handle: self._route_prefix_for_action(
                obs_builder,
                handle,
                self._action_id(action),
                self.FUTURE_RERANK_LOOKAHEAD_CELLS,
            )
            for handle, action in adjusted.items()
        }

        reserved_targets = set()
        for handle in sorted(adjusted, key=lambda h: self._priority_key(obs_builder, h)):
            action_id = self._action_id(adjusted[handle])
            if action_id != ReservationPolicy.MOVE_FORWARD:
                self._reserve_action_target(
                    reserved_targets,
                    obs_builder,
                    handle,
                    action_id,
                )
                continue

            candidate = self._best_near_tie_side_detour(
                obs_builder,
                handle,
                action_id,
                planned_prefixes,
                reserved_targets,
                logits_by_handle.get(handle),
            )
            if candidate is not None:
                adjusted[handle] = RailEnvActions(candidate)
                action_id = candidate
                planned_prefixes[handle] = self._route_prefix_for_action(
                    obs_builder,
                    handle,
                    candidate,
                    self.FUTURE_RERANK_LOOKAHEAD_CELLS,
                )

            self._reserve_action_target(
                reserved_targets,
                obs_builder,
                handle,
                action_id,
            )

        return adjusted

    def _best_near_tie_side_detour(
        self,
        obs_builder: Any,
        handle: int,
        baseline_action: int,
        planned_prefixes: dict[int, list[dict[str, Any]]],
        reserved_targets: set[tuple[int, int]],
        logits: np.ndarray | None,
    ) -> int | None:
        if logits is None or baseline_action >= len(logits):
            return None
        if self._future_head_on_risk(
            obs_builder,
            handle,
            baseline_action,
            planned_prefixes,
        ) > 0.0:
            return None

        agent = obs_builder.env.agents[handle]
        if agent.position is None:
            return None
        direction = (
            agent.direction if agent.direction is not None else agent.initial_direction
        )
        distance_map = obs_builder._get_distance_map(handle)
        baseline_target, baseline_direction = obs_builder._action_target(
            handle,
            baseline_action,
        )
        if baseline_target is None or baseline_direction is None:
            return None
        current_distance = distance_map[agent.position[0], agent.position[1], direction]
        current_slack = obs_builder._deadline_slack(handle, current_distance)
        if (
            not np.isfinite(current_slack)
            or current_slack < self.NEAR_TIE_SIDE_MIN_SLACK
        ):
            return None
        baseline_distance = distance_map[
            baseline_target[0],
            baseline_target[1],
            baseline_direction,
        ]
        if not np.isfinite(baseline_distance):
            return None

        mask = obs_builder._coordination_masks.get(
            handle,
            obs_builder._build_local_action_mask(handle),
        )
        best_action = None
        best_logit = -np.inf
        for candidate in (ReservationPolicy.MOVE_LEFT, ReservationPolicy.MOVE_RIGHT):
            if candidate >= len(logits) or mask[candidate] < 0.5:
                continue
            target, target_direction = obs_builder._action_target(handle, candidate)
            if target is None or target_direction is None:
                continue
            if target in reserved_targets or obs_builder._occupied_by_other(target, handle):
                continue
            candidate_distance = distance_map[target[0], target[1], target_direction]
            if not np.isfinite(candidate_distance):
                continue
            if (
                candidate_distance
                > baseline_distance + self.NEAR_TIE_SIDE_DISTANCE_MARGIN
            ):
                continue
            logit_loss = float(logits[baseline_action] - logits[candidate])
            if not np.isfinite(logit_loss):
                continue
            if logit_loss > self.NEAR_TIE_SIDE_MAX_LOGIT_LOSS:
                continue
            risk = self._future_head_on_risk(
                obs_builder,
                handle,
                candidate,
                planned_prefixes,
            )
            if risk > 0.0:
                continue
            if self._has_prefix_interaction(
                obs_builder,
                handle,
                candidate,
                planned_prefixes,
            ):
                continue
            if self._reject_residual_head_on_detour(
                obs_builder,
                handle,
                candidate,
                planned_prefixes,
            ):
                continue
            if float(logits[candidate]) > best_logit:
                best_logit = float(logits[candidate])
                best_action = candidate
        return best_action

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

    def _logits_by_handle(
        self,
        handles: List[int],
        observations: List[Any],
    ) -> dict[int, np.ndarray]:
        try:
            with torch.no_grad():
                logits = self.rl_policy.masked_logits(
                    np.asarray(observations, dtype=np.float32)
                )
            return {
                handle: np.asarray(row, dtype=np.float32)
                for handle, row in zip(handles, logits.cpu().numpy())
            }
        except Exception:
            return {}


MyPolicy = LogitDetourPolicy
