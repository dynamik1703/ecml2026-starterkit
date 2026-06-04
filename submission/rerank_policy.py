from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.hybrid_policy import HybridPolicy
from submission.reservation_policy import ReservationPolicy


class RerankPolicy(HybridPolicy):
    FUTURE_RERANK_LOOKAHEAD_CELLS = 45
    FUTURE_RERANK_MAX_STEP = 24
    FUTURE_RERANK_MAX_ETA_GAP = 2.0
    FUTURE_RERANK_MIN_RISK_IMPROVEMENT = 20.0
    FUTURE_RERANK_LEFT_MAX_SLACK = 115.0
    FUTURE_RERANK_DEADLINE_CONFLICT_MAX_STEP = 30
    FUTURE_RERANK_DEADLINE_CONFLICT_MAX_ETA_GAP = 4.0
    FUTURE_RERANK_DEADLINE_TIGHT_SLACK = 220.0
    FUTURE_RERANK_DEADLINE_PENALTY_COEF = 0.75
    FUTURE_RERANK_HEAD_ON_TIGHT_PAIR_SLACK = 90.0

    def act_many(
        self, handles: List[int], observations: List[Any], **kwargs
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
        return self._apply_temporal_corridor_locks(
            handles,
            observations,
            adjusted,
            context.obs_builder,
        )

    def _rerank_future_head_on_detours(
        self,
        actions: Dict[int, RailEnvActions],
        obs_builder: Any,
    ) -> Dict[int, RailEnvActions]:
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

            baseline_risk = self._future_head_on_risk(
                obs_builder,
                handle,
                action_id,
                planned_prefixes,
            )
            if baseline_risk <= 0.0:
                self._reserve_action_target(
                    reserved_targets,
                    obs_builder,
                    handle,
                    action_id,
                )
                continue

            candidate = self._best_future_head_on_detour(
                obs_builder,
                handle,
                action_id,
                baseline_risk,
                planned_prefixes,
                reserved_targets,
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

    def _best_future_head_on_detour(
        self,
        obs_builder: Any,
        handle: int,
        baseline_action: int,
        baseline_risk: float,
        planned_prefixes: dict[int, list[dict[str, Any]]],
        reserved_targets: set[tuple[int, int]],
    ) -> int | None:
        agent = obs_builder.env.agents[handle]
        if agent.position is None:
            return None
        direction = (
            agent.direction if agent.direction is not None else agent.initial_direction
        )
        distance_map = obs_builder._get_distance_map(handle)
        current_distance = distance_map[agent.position[0], agent.position[1], direction]
        current_slack = obs_builder._deadline_slack(handle, current_distance)
        mask = obs_builder._coordination_masks.get(
            handle,
            obs_builder._build_local_action_mask(handle),
        )
        baseline_deadline_penalty = self._deadline_conflict_penalty(
            obs_builder,
            handle,
            baseline_action,
            planned_prefixes,
        )

        best_action = None
        best_score = baseline_risk
        for candidate in (
            ReservationPolicy.MOVE_LEFT,
            ReservationPolicy.MOVE_RIGHT,
            ReservationPolicy.MOVE_FORWARD,
        ):
            if candidate == baseline_action or mask[candidate] < 0.5:
                continue
            if (
                candidate == ReservationPolicy.MOVE_LEFT
                and (
                    not np.isfinite(current_slack)
                    or current_slack > self.FUTURE_RERANK_LEFT_MAX_SLACK
                )
            ):
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
                np.isfinite(current_distance)
                and candidate_distance > current_distance + self.SIDE_DETOUR_MARGIN
            ):
                continue

            risk = self._future_head_on_risk(
                obs_builder,
                handle,
                candidate,
                planned_prefixes,
            )
            deadline_penalty = self._deadline_conflict_penalty(
                obs_builder,
                handle,
                candidate,
                planned_prefixes,
            )
            if self._reject_residual_head_on_detour(
                obs_builder,
                handle,
                candidate,
                planned_prefixes,
            ):
                continue
            distance_penalty = max(0.0, candidate_distance - current_distance)
            score = (
                risk
                + 0.25 * distance_penalty
                + max(0.0, deadline_penalty - baseline_deadline_penalty)
            )
            if score + self.FUTURE_RERANK_MIN_RISK_IMPROVEMENT < best_score:
                best_score = score
                best_action = candidate

        return best_action

    def _reject_residual_head_on_detour(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        planned_prefixes: dict[int, list[dict[str, Any]]],
    ) -> bool:
        min_pair_slack = self._residual_head_on_min_pair_deadline_slack(
            obs_builder,
            handle,
            action,
            planned_prefixes,
        )
        if not np.isfinite(min_pair_slack):
            return False
        if min_pair_slack < self.FUTURE_RERANK_HEAD_ON_TIGHT_PAIR_SLACK:
            return True
        return False

    def _residual_head_on_min_pair_deadline_slack(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        planned_prefixes: dict[int, list[dict[str, Any]]],
    ) -> float:
        own_prefix = self._route_prefix_for_action(
            obs_builder,
            handle,
            action,
            self.FUTURE_RERANK_LOOKAHEAD_CELLS,
        )
        if not own_prefix:
            return np.inf
        own_edges = self._prefix_edges(own_prefix)
        if not own_edges:
            return np.inf

        min_pair_slack = np.inf
        for other, other_prefix in planned_prefixes.items():
            if other == handle or not other_prefix:
                continue
            other_edges = self._prefix_edges(other_prefix)
            for source, target in own_edges:
                other_node = other_edges.get((target, source))
                if other_node is None:
                    continue
                own_slack = self._prefix_deadline_slack(
                    obs_builder,
                    handle,
                    own_edges[(source, target)],
                )
                other_slack = self._prefix_deadline_slack(
                    obs_builder,
                    other,
                    other_node,
                )
                if np.isfinite(own_slack) and np.isfinite(other_slack):
                    min_pair_slack = min(min_pair_slack, own_slack, other_slack)
        return min_pair_slack

    def _deadline_conflict_penalty(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        planned_prefixes: dict[int, list[dict[str, Any]]],
    ) -> float:
        own_prefix = self._route_prefix_for_action(
            obs_builder,
            handle,
            action,
            self.FUTURE_RERANK_LOOKAHEAD_CELLS,
        )
        if not own_prefix:
            return 0.0

        own_speed = self._agent_speed(obs_builder.env.agents[handle])
        penalty = 0.0
        seen_conflicts = set()
        for other, other_prefix in planned_prefixes.items():
            if other == handle or not other_prefix:
                continue

            other_speed = self._agent_speed(obs_builder.env.agents[other])
            other_positions = {}
            other_edges = {}
            for node in other_prefix:
                other_positions.setdefault(node["position"], node)
                previous = node["prev_position"]
                if previous is None or previous == node["position"]:
                    continue
                other_edges.setdefault((previous, node["position"]), node)

            for own_node in own_prefix:
                if int(own_node["step"]) > self.FUTURE_RERANK_DEADLINE_CONFLICT_MAX_STEP:
                    break

                same_cell_node = other_positions.get(own_node["position"])
                if same_cell_node is not None:
                    key = (other, own_node["step"], same_cell_node["step"], 0)
                    if key not in seen_conflicts:
                        penalty += self._deadline_conflict_node_penalty(
                            obs_builder,
                            handle,
                            other,
                            own_node,
                            same_cell_node,
                            own_speed,
                            other_speed,
                            head_on=False,
                        )
                        seen_conflicts.add(key)

                previous = own_node["prev_position"]
                if previous is None or previous == own_node["position"]:
                    continue
                reverse_node = other_edges.get((own_node["position"], previous))
                if reverse_node is None:
                    continue
                key = (other, own_node["step"], reverse_node["step"], 1)
                if key in seen_conflicts:
                    continue
                penalty += self._deadline_conflict_node_penalty(
                    obs_builder,
                    handle,
                    other,
                    own_node,
                    reverse_node,
                    own_speed,
                    other_speed,
                    head_on=True,
                )
                seen_conflicts.add(key)

        return penalty

    def _deadline_conflict_node_penalty(
        self,
        obs_builder: Any,
        handle: int,
        other: int,
        own_node: dict[str, Any],
        other_node: dict[str, Any],
        own_speed: float,
        other_speed: float,
        head_on: bool,
    ) -> float:
        own_step = int(own_node["step"])
        other_step = int(other_node["step"])
        if own_step < 1 or own_step > self.FUTURE_RERANK_DEADLINE_CONFLICT_MAX_STEP:
            return 0.0
        eta_gap = abs(own_step / own_speed - other_step / other_speed)
        if eta_gap > self.FUTURE_RERANK_DEADLINE_CONFLICT_MAX_ETA_GAP:
            return 0.0

        own_slack = self._prefix_deadline_slack(obs_builder, handle, own_node)
        other_slack = self._prefix_deadline_slack(obs_builder, other, other_node)
        if not np.isfinite(own_slack) or not np.isfinite(other_slack):
            return 0.0

        pair_slack = min(own_slack, other_slack)
        if pair_slack >= self.FUTURE_RERANK_DEADLINE_TIGHT_SLACK:
            return 0.0

        slack_deficit = min(
            300.0,
            self.FUTURE_RERANK_DEADLINE_TIGHT_SLACK - pair_slack,
        )
        eta_factor = (
            1.0
            + (
                self.FUTURE_RERANK_DEADLINE_CONFLICT_MAX_ETA_GAP
                - eta_gap
            )
            / max(1.0, self.FUTURE_RERANK_DEADLINE_CONFLICT_MAX_ETA_GAP)
        )
        head_on_factor = 1.35 if head_on else 1.0
        priority_factor = 1.15 if other_slack < own_slack else 1.0
        return (
            self.FUTURE_RERANK_DEADLINE_PENALTY_COEF
            * slack_deficit
            * eta_factor
            * head_on_factor
            * priority_factor
        )

    def _prefix_deadline_slack(
        self,
        obs_builder: Any,
        handle: int,
        node: dict[str, Any],
    ) -> float:
        position = node["position"]
        direction = node["direction"]
        if position is None or direction is None or not obs_builder._is_in_bounds(position):
            return np.inf
        distance_map = obs_builder._get_distance_map(handle)
        remaining_distance = distance_map[position[0], position[1], direction]
        if not np.isfinite(remaining_distance):
            return np.inf
        total_distance = int(node["step"]) + remaining_distance
        return float(obs_builder._deadline_slack(handle, total_distance))

    def _future_head_on_risk(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        planned_prefixes: dict[int, list[dict[str, Any]]],
    ) -> float:
        own_prefix = self._route_prefix_for_action(
            obs_builder,
            handle,
            action,
            self.FUTURE_RERANK_LOOKAHEAD_CELLS,
        )
        if not own_prefix:
            return 0.0

        own_priority = self._priority_key(obs_builder, handle)
        own_edges = self._prefix_edges(own_prefix)
        own_speed = self._agent_speed(obs_builder.env.agents[handle])
        best_risk = 0.0
        for other, other_prefix in planned_prefixes.items():
            if other == handle or not other_prefix:
                continue
            if self._priority_key(obs_builder, other) >= own_priority:
                continue
            other_speed = self._agent_speed(obs_builder.env.agents[other])
            for other_node in other_prefix:
                previous = other_node["prev_position"]
                if previous is None:
                    continue
                own_node = own_edges.get((other_node["position"], previous))
                if own_node is None:
                    continue
                risk = self._head_on_node_risk(own_node, other_node, own_speed, other_speed)
                if risk > best_risk:
                    best_risk = risk
        return best_risk

    def _head_on_node_risk(
        self,
        own_node: dict[str, Any],
        other_node: dict[str, Any],
        own_speed: float,
        other_speed: float,
    ) -> float:
        own_step = int(own_node["step"])
        other_step = int(other_node["step"])
        if own_step < 2 or own_step > self.FUTURE_RERANK_MAX_STEP:
            return 0.0
        eta_gap = abs(own_step / own_speed - other_step / other_speed)
        if eta_gap > self.FUTURE_RERANK_MAX_ETA_GAP:
            return 0.0
        step_risk = (self.FUTURE_RERANK_MAX_STEP - own_step + 1) * 2.0
        eta_risk = (self.FUTURE_RERANK_MAX_ETA_GAP - eta_gap + 1.0) * 15.0
        return 100.0 + step_risk + eta_risk

    def _route_prefix_for_action(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        lookahead: int,
    ) -> list[dict[str, Any]]:
        if handle >= len(obs_builder.env.agents):
            return []
        agent = obs_builder.env.agents[handle]
        state_name = getattr(agent.state, "name", str(agent.state))
        if state_name in {"DONE", "DONE_REMOVED"}:
            return []

        position, direction = self._route_start(obs_builder, handle)
        if position is None or direction is None:
            return []

        distance_map = obs_builder._get_distance_map(handle)
        if action == ReservationPolicy.DO_NOTHING and self._agent_is_moving(agent):
            target_position, target_direction = self._implicit_moving_target(
                position,
                direction,
                obs_builder,
            )
        elif action in (
            ReservationPolicy.MOVE_LEFT,
            ReservationPolicy.MOVE_FORWARD,
            ReservationPolicy.MOVE_RIGHT,
        ):
            target_position, target_direction = obs_builder._action_target(handle, action)
        elif agent.position is not None:
            target_position = agent.position
            target_direction = direction
        else:
            return []

        if (
            target_position is None
            or target_direction is None
            or not obs_builder._is_in_bounds(target_position)
        ):
            return []

        previous_position = position if action != ReservationPolicy.STOP_MOVING else None
        prefix = [
            {
                "step": 1,
                "prev_position": previous_position,
                "position": target_position,
                "direction": target_direction,
            }
        ]
        current_position = target_position
        current_direction = target_direction
        seen = {(current_position, current_direction)}
        for step in range(2, lookahead + 1):
            transitions = obs_builder.env.rail.get_transitions(
                (current_position, current_direction)
            )
            next_direction = self._best_progress_direction(
                transitions,
                current_position,
                distance_map,
            )
            if next_direction is None:
                break
            next_position = get_new_position(current_position, next_direction)
            if not obs_builder._is_in_bounds(next_position):
                break
            state = (next_position, next_direction)
            if state in seen:
                break
            seen.add(state)
            prefix.append(
                {
                    "step": step,
                    "prev_position": current_position,
                    "position": next_position,
                    "direction": next_direction,
                }
            )
            current_position = next_position
            current_direction = next_direction
        return prefix

    def _reserve_action_target(
        self,
        reserved_targets: set[tuple[int, int]],
        obs_builder: Any,
        handle: int,
        action: int,
    ) -> None:
        target, _ = obs_builder._action_target(handle, action)
        if target is not None:
            reserved_targets.add(target)

    @staticmethod
    def _prefix_edges(
        prefix: list[dict[str, Any]],
    ) -> dict[tuple[tuple[int, int], tuple[int, int]], dict[str, Any]]:
        edges = {}
        for node in prefix:
            previous = node["prev_position"]
            if previous is None or previous == node["position"]:
                continue
            edges.setdefault((previous, node["position"]), node)
        return edges

    @staticmethod
    def _route_start(
        obs_builder: Any,
        handle: int,
    ) -> tuple[tuple[int, int] | None, int | None]:
        agent = obs_builder.env.agents[handle]
        if agent.position is None:
            return agent.initial_position, agent.initial_direction
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        return agent.position, direction

    @staticmethod
    def _agent_speed(agent: Any) -> float:
        speed = float(agent.speed_counter.speed)
        return speed if speed > 0 else 1.0

    @staticmethod
    def _action_id(action: RailEnvActions | int) -> int:
        return int(action.value) if hasattr(action, "value") else int(action)


MyPolicy = RerankPolicy
