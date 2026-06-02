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
    LONG_LOOKAHEAD_CELLS = 60

    def __init__(self):
        self.rl_policy = ActorCritic()
        self.reservation_policy = ReservationPolicy()

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
            or max(env.height, env.width) < 100
        ):
            return actions
        return self._detour_around_long_opposing(actions, context.obs_builder)

    def _detour_around_long_opposing(
        self,
        actions: Dict[int, RailEnvActions],
        obs_builder: Any,
    ) -> Dict[int, RailEnvActions]:
        adjusted = dict(actions)
        reserved_targets = set()
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

            detour = self._best_side_detour(handle, reserved_targets, obs_builder)
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
            new_dist = distance_map[target[0], target[1], new_direction]
            if not np.isfinite(new_dist):
                continue
            if np.isfinite(current_dist) and new_dist > current_dist + self.SIDE_DETOUR_MARGIN:
                continue
            if new_dist < best_dist:
                best_dist = new_dist
                best_action = action
        return best_action

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
    def _priority_key(obs_builder: Any, handle: int) -> tuple[float, float, int]:
        try:
            _, slack, distance, _ = obs_builder._priority_key(handle)
            return slack, distance, handle
        except Exception:
            return 1e9, 1e9, handle


MyPolicy = HybridPolicy
