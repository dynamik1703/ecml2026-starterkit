from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_argmax
from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.my_policy import ActorCritic
from submission.reservation_policy import ReservationPolicy


class HybridPolicy:
    SIDE_DETOUR_MARGIN = 4.0
    ADJACENT_ESCAPE_MARGIN = 400.0
    LONG_LOOKAHEAD_CELLS = 45
    TEMPORAL_CORRIDOR_MIN_EDGES = 4
    TEMPORAL_CORRIDOR_MAX_AGE = 90
    TEMPORAL_CORRIDOR_MAX_CONFLICT_WAIT = 3
    TEMPORAL_CORRIDOR_STALE_PROGRESS_AGE = 8

    def __init__(self):
        self.rl_policy = ActorCritic()
        self.reservation_policy = ReservationPolicy()
        self._corridor_locks = {}
        self._corridor_lock_block_counts = {}
        self._corridor_lock_env_id = None
        self._corridor_lock_step = -1

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        return self.rl_policy.act(observation, **kwargs)

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
        return self._apply_temporal_corridor_locks(
            handles,
            observations,
            adjusted,
            context.obs_builder,
        )

    def _apply_temporal_corridor_locks(
        self,
        handles: List[int],
        observations: List[Any],
        actions: Dict[int, RailEnvActions],
        obs_builder: Any,
    ) -> Dict[int, RailEnvActions]:
        self._sync_corridor_locks(obs_builder)
        observations_by_handle = dict(zip(handles, observations))
        adjusted = dict(actions)
        planned_locks = []

        for handle in sorted(handles, key=lambda h: self._priority_key(obs_builder, h)):
            action = adjusted.get(handle, RailEnvActions.DO_NOTHING)
            action_id = int(action.value) if hasattr(action, "value") else int(action)
            edges = self._corridor_edges_for_action(obs_builder, handle, action_id)
            if len(edges) < self.TEMPORAL_CORRIDOR_MIN_EDGES:
                continue

            conflict_lock = self._conflicting_corridor_lock(
                handle,
                edges,
                planned_locks,
            )
            if conflict_lock is not None and self._can_progress_through_corridor_lock(
                obs_builder,
                handle,
                action_id,
                conflict_lock,
            ):
                continue

            if conflict_lock is not None and self._should_wait_for_corridor_lock(
                conflict_lock,
                handle,
            ):
                fallback = self._corridor_lock_fallback(
                    observations_by_handle.get(handle)
                )
                if fallback is not None:
                    adjusted[handle] = RailEnvActions(fallback)
                continue

            lock = self._make_corridor_lock(obs_builder, handle, edges)
            self._corridor_locks[handle] = lock
            planned_locks.append(lock)

        return adjusted

    def _sync_corridor_locks(self, obs_builder: Any) -> None:
        env = obs_builder.env
        step = int(env._elapsed_steps)
        env_id = id(env)
        if env_id != self._corridor_lock_env_id or step < self._corridor_lock_step:
            self._corridor_locks = {}
            self._corridor_lock_block_counts = {}
            self._corridor_lock_env_id = env_id

        active_locks = {}
        active_owners = set()
        for handle, lock in self._corridor_locks.items():
            if handle >= len(env.agents):
                continue
            agent = env.agents[handle]
            state_name = getattr(agent.state, "name", str(agent.state))
            if state_name in {"DONE", "DONE_REMOVED"}:
                continue
            if agent.position is None:
                continue
            if step - lock["step"] > self.TEMPORAL_CORRIDOR_MAX_AGE:
                continue
            if agent.position not in lock["cells"]:
                continue
            active_locks[handle] = lock
            active_owners.add(handle)

        self._corridor_locks = active_locks
        self._corridor_lock_block_counts = {
            key: count
            for key, count in self._corridor_lock_block_counts.items()
            if key[0] in active_owners
        }
        self._corridor_lock_step = step

    def _corridor_edges_for_action(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
    ) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
        if action not in (
            ReservationPolicy.MOVE_LEFT,
            ReservationPolicy.MOVE_FORWARD,
            ReservationPolicy.MOVE_RIGHT,
        ):
            return ()
        try:
            return tuple(obs_builder._corridor_edges_for_action(handle, action))
        except Exception:
            return ()

    def _make_corridor_lock(
        self,
        obs_builder: Any,
        handle: int,
        edges: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
    ) -> dict[str, Any]:
        cells = set()
        for source, target in edges:
            cells.add(source)
            cells.add(target)
        return {
            "owner": handle,
            "step": int(obs_builder.env._elapsed_steps),
            "edges": frozenset(edges),
            "cells": cells,
        }

    def _conflicting_corridor_lock(
        self,
        handle: int,
        edges: tuple[tuple[tuple[int, int], tuple[int, int]], ...],
        planned_locks: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        for lock in list(self._corridor_locks.values()) + planned_locks:
            if lock["owner"] == handle:
                continue
            lock_edges = lock["edges"]
            if any((target, source) in lock_edges for source, target in edges):
                return lock
        return None

    def _should_wait_for_corridor_lock(
        self,
        lock: dict[str, Any],
        handle: int,
    ) -> bool:
        key = (lock["owner"], handle)
        count = self._corridor_lock_block_counts.get(key, 0)
        if count >= self.TEMPORAL_CORRIDOR_MAX_CONFLICT_WAIT:
            return False
        self._corridor_lock_block_counts[key] = count + 1
        return True

    def _can_progress_through_corridor_lock(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        lock: dict[str, Any],
    ) -> bool:
        if action not in (
            ReservationPolicy.MOVE_LEFT,
            ReservationPolicy.MOVE_FORWARD,
            ReservationPolicy.MOVE_RIGHT,
        ):
            return False

        agent = obs_builder.env.agents[handle]
        if agent.position is None:
            return False

        target_position, target_direction = obs_builder._action_target(handle, action)
        if target_position is None or target_direction is None:
            return False

        owner = lock["owner"]
        if owner >= len(obs_builder.env.agents):
            return False

        current_direction = (
            agent.direction if agent.direction is not None else agent.initial_direction
        )
        distance_map = obs_builder._get_distance_map(handle)
        current_distance = distance_map[
            agent.position[0],
            agent.position[1],
            current_direction,
        ]
        new_distance = distance_map[
            target_position[0],
            target_position[1],
            target_direction,
        ]
        if not (
            np.isfinite(current_distance)
            and np.isfinite(new_distance)
            and new_distance < current_distance
        ):
            return False

        lock_age = int(obs_builder.env._elapsed_steps) - int(lock["step"])
        if lock_age >= self.TEMPORAL_CORRIDOR_STALE_PROGRESS_AGE:
            return True

        own_slack = obs_builder._deadline_slack(handle, current_distance)
        owner_distance = obs_builder._current_distance_to_waypoint(owner)
        owner_slack = obs_builder._deadline_slack(owner, owner_distance)
        if not (np.isfinite(own_slack) and np.isfinite(owner_slack)):
            return False
        return own_slack <= owner_slack

    def _corridor_lock_fallback(self, observation: Any) -> int | None:
        mask = self._mask_from_observation(observation)
        if mask[ReservationPolicy.STOP_MOVING] >= 0.5:
            return ReservationPolicy.STOP_MOVING
        if mask[ReservationPolicy.DO_NOTHING] >= 0.5:
            return ReservationPolicy.DO_NOTHING
        return None

    def _break_adjacent_deadlocks(
        self,
        actions: Dict[int, RailEnvActions],
        obs_builder: Any,
    ) -> Dict[int, RailEnvActions]:
        adjusted = dict(actions)
        reserved_targets = set()
        for handle in sorted(actions, key=lambda h: self._priority_key(obs_builder, h)):
            action = adjusted[handle]
            action_id = int(action.value) if hasattr(action, "value") else int(action)
            escape = self._adjacent_deadlock_escape(
                handle,
                action_id,
                reserved_targets,
                obs_builder,
            )
            if escape is not None:
                adjusted[handle] = RailEnvActions(escape)
                action_id = escape

            target, _ = obs_builder._action_target(handle, action_id)
            if target is not None:
                reserved_targets.add(target)
        return adjusted

    def _adjacent_deadlock_escape(
        self,
        handle: int,
        action: int,
        reserved_targets: set[tuple[int, int]],
        obs_builder: Any,
    ) -> int | None:
        agent = obs_builder.env.agents[handle]
        if agent.position is None:
            return None

        blocked_by = self._adjacent_mutual_blocker(handle, action, obs_builder)
        if blocked_by is None:
            return None

        mask = obs_builder._coordination_masks.get(
            handle,
            obs_builder._build_local_action_mask(handle),
        )
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        distance_map = obs_builder._get_distance_map(handle)
        current_distance = distance_map[agent.position[0], agent.position[1], direction]

        best_action = None
        best_distance = np.inf
        for candidate in (
            ReservationPolicy.MOVE_LEFT,
            ReservationPolicy.MOVE_RIGHT,
            ReservationPolicy.MOVE_FORWARD,
        ):
            if candidate == action or mask[candidate] < 0.5:
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
                and candidate_distance > current_distance + self.ADJACENT_ESCAPE_MARGIN
            ):
                continue
            if candidate_distance < best_distance:
                best_distance = candidate_distance
                best_action = candidate

        return best_action

    def _adjacent_mutual_blocker(
        self,
        handle: int,
        action: int,
        obs_builder: Any,
    ) -> int | None:
        if action in (
            ReservationPolicy.MOVE_LEFT,
            ReservationPolicy.MOVE_FORWARD,
            ReservationPolicy.MOVE_RIGHT,
        ):
            target, _ = obs_builder._action_target(handle, action)
            if target is not None:
                blocker = obs_builder._agent_at(target)
                if blocker != -1 and blocker != handle:
                    return (
                        int(blocker)
                        if self._targets_agent_position(blocker, handle, obs_builder)
                        else None
                    )

        for candidate in (
            ReservationPolicy.MOVE_LEFT,
            ReservationPolicy.MOVE_FORWARD,
            ReservationPolicy.MOVE_RIGHT,
        ):
            target, _ = obs_builder._action_target(handle, candidate)
            if target is None:
                continue
            blocker = obs_builder._agent_at(target)
            if blocker == -1 or blocker == handle:
                continue
            if self._targets_agent_position(blocker, handle, obs_builder):
                return int(blocker)
        return None

    def _targets_agent_position(
        self,
        handle: int,
        target_handle: int,
        obs_builder: Any,
    ) -> bool:
        if handle >= len(obs_builder.env.agents):
            return False
        target_agent = obs_builder.env.agents[target_handle]
        if target_agent.position is None:
            return False
        for action in (
            ReservationPolicy.MOVE_LEFT,
            ReservationPolicy.MOVE_FORWARD,
            ReservationPolicy.MOVE_RIGHT,
        ):
            target, _ = obs_builder._action_target(handle, action)
            if target == target_agent.position:
                return True
        return False

    def _detour_around_long_opposing(
        self,
        actions: Dict[int, RailEnvActions],
        obs_builder: Any,
    ) -> Dict[int, RailEnvActions]:
        adjusted = dict(actions)
        reserved_targets = set()
        planned_targets = {}
        for handle, action in actions.items():
            action_id = int(action.value) if hasattr(action, "value") else int(action)
            target, _ = obs_builder._action_target(handle, action_id)
            if target is not None:
                planned_targets[handle] = target

        for handle in sorted(actions, key=lambda h: self._priority_key(obs_builder, h)):
            action = adjusted[handle]
            action_id = int(action.value) if hasattr(action, "value") else int(action)
            if action_id != ReservationPolicy.MOVE_FORWARD:
                target, _ = obs_builder._action_target(handle, action_id)
                if target is not None:
                    reserved_targets.add(target)
                continue

            if not self._has_long_opposing_ahead(handle, action_id, obs_builder):
                target, _ = obs_builder._action_target(handle, action_id)
                if target is not None:
                    reserved_targets.add(target)
                continue

            other_planned_targets = {
                target
                for other_handle, target in planned_targets.items()
                if other_handle != handle
            }
            detour = self._best_side_detour(
                handle,
                reserved_targets | other_planned_targets,
                obs_builder,
            )
            if detour is not None:
                adjusted[handle] = RailEnvActions(detour)

            target, _ = obs_builder._action_target(
                handle,
                int(adjusted[handle].value)
                if hasattr(adjusted[handle], "value")
                else int(adjusted[handle]),
            )
            if target is not None:
                reserved_targets.add(target)
        return adjusted

    def _best_side_detour(
        self,
        handle: int,
        reserved_targets: set[tuple[int, int]],
        obs_builder: Any,
    ) -> int | None:
        env = obs_builder.env
        agent = env.agents[handle]
        if agent.position is None:
            return None
        direction = agent.direction if agent.direction is not None else agent.initial_direction
        transitions = env.rail.get_transitions((agent.position, direction))
        distance_map = obs_builder._get_distance_map(handle)
        current_dist = distance_map[agent.position[0], agent.position[1], direction]

        best_action = None
        best_dist = np.inf
        for action in (ReservationPolicy.MOVE_LEFT, ReservationPolicy.MOVE_RIGHT):
            new_direction = obs_builder._action_to_direction(action, direction)
            if not transitions[new_direction]:
                continue
            target = get_new_position(agent.position, new_direction)
            if target in reserved_targets or obs_builder._occupied_by_other(target, handle):
                continue
            if self._side_detour_creates_adjacent_trap(
                target,
                new_direction,
                reserved_targets,
                distance_map,
                obs_builder,
            ):
                continue
            new_dist = distance_map[target[0], target[1], new_direction]
            if not np.isfinite(new_dist):
                continue
            if np.isfinite(current_dist) and new_dist > current_dist + self.SIDE_DETOUR_MARGIN:
                continue
            if new_dist < best_dist:
                best_dist = new_dist
                best_action = action
        return best_action

    def _side_detour_creates_adjacent_trap(
        self,
        target: tuple[int, int],
        direction: int,
        reserved_targets: set[tuple[int, int]],
        distance_map: Any,
        obs_builder: Any,
    ) -> bool:
        if not reserved_targets:
            return False

        transitions = obs_builder.env.rail.get_transitions((target, direction))
        next_direction = self._best_progress_direction(
            transitions,
            target,
            distance_map,
        )
        if next_direction is None:
            return False

        next_position = get_new_position(target, next_direction)
        for reserved in reserved_targets:
            if self._manhattan(next_position, reserved) <= 1:
                return True
        return False

    def _has_long_opposing_ahead(
        self,
        handle: int,
        action: int,
        obs_builder: Any,
    ) -> bool:
        target_position, target_direction = obs_builder._action_target(handle, action)
        if target_position is None or target_direction is None:
            return False

        env = obs_builder.env
        distance_map = obs_builder._get_distance_map(handle)
        current_position = target_position
        current_direction = target_direction
        for _ in range(self.LONG_LOOKAHEAD_CELLS):
            if not obs_builder._is_in_bounds(current_position):
                return False
            other = obs_builder._agent_at(current_position)
            if other != -1 and other != handle:
                other_agent = env.agents[other]
                other_direction = (
                    other_agent.direction
                    if other_agent.direction is not None
                    else other_agent.initial_direction
                )
                return other_direction != current_direction

            transitions = env.rail.get_transitions((current_position, current_direction))
            next_direction = self._best_progress_direction(
                transitions,
                current_position,
                distance_map,
            )
            if next_direction is None:
                return False
            current_position = get_new_position(current_position, next_direction)
            current_direction = next_direction
        return False

    @staticmethod
    def _best_progress_direction(
        transitions: tuple[int, int, int, int],
        position: tuple[int, int],
        distance_map: Any,
    ) -> int | None:
        best_direction = None
        best_distance = np.inf
        for direction, is_open in enumerate(transitions):
            if not is_open:
                continue
            next_position = get_new_position(position, direction)
            distance = distance_map[next_position[0], next_position[1], direction]
            if np.isfinite(distance) and distance < best_distance:
                best_distance = distance
                best_direction = direction
        if best_direction is not None:
            return best_direction
        if any(transitions):
            return fast_argmax(transitions)
        return None

    @staticmethod
    def _manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
        return abs(left[0] - right[0]) + abs(left[1] - right[1])

    @staticmethod
    def _priority_key(obs_builder: Any, handle: int) -> tuple[float, float, int]:
        try:
            _, slack, distance, _ = obs_builder._priority_key(handle)
            return slack, distance, handle
        except Exception:
            return 1e9, 1e9, handle

    @staticmethod
    def _mask_from_observation(observation: Any) -> np.ndarray:
        if observation is None:
            return np.zeros(5, dtype=np.float32)
        values = np.asarray(observation, dtype=np.float32)
        if values.shape[0] < 5:
            return np.zeros(5, dtype=np.float32)
        return values[-5:]


MyPolicy = HybridPolicy
