from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_argmax, fast_count_nonzero
from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

from submission import runtime_context


class ReservationPolicy:
    DO_NOTHING = 0
    MOVE_LEFT = 1
    MOVE_FORWARD = 2
    MOVE_RIGHT = 3
    STOP_MOVING = 4

    OPPOSING_PENALTY = 100.0
    SAME_DIRECTION_PENALTY = 15.0
    NON_PROGRESS_PENALTY = 12.0
    WAIT_PENALTY = 8.0

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        mask = self._mask_from_observation(observation)
        return RailEnvActions(self._fallback_action(mask))

    def act_many(
        self, handles: List[int], observations: List[Any], **kwargs
    ) -> Dict[int, RailEnvActions]:
        context = runtime_context.get()
        if context.env is None or context.obs_builder is None:
            return {
                handle: RailEnvActions(self._fallback_action(self._mask_from_observation(obs)))
                for handle, obs in zip(handles, observations)
            }

        env = context.env
        obs_builder = context.obs_builder
        obs_by_handle = dict(zip(handles, observations))
        actions: dict[int, RailEnvActions] = {}
        reserved_positions = {
            agent.position for agent in env.agents if agent.position is not None
        }
        reserved_edges: set[tuple[tuple[int, int] | None, tuple[int, int]]] = set()
        planned_targets: dict[int, tuple[int, int] | None] = {}

        for handle in sorted(handles, key=lambda h: self._priority_key(obs_builder, h)):
            action = self._choose_action(
                handle,
                obs_by_handle.get(handle),
                reserved_positions,
                reserved_edges,
                planned_targets,
                obs_builder,
            )
            actions[handle] = RailEnvActions(action)

            source, target = self._movement_edge(obs_builder, handle, action)
            planned_targets[handle] = target
            if target is not None:
                reserved_positions.add(target)
            if source is not None and target is not None:
                reserved_edges.add((source, target))

        return actions

    def _choose_action(
        self,
        handle: int,
        observation: Any,
        reserved_positions: set[tuple[int, int]],
        reserved_edges: set[tuple[tuple[int, int] | None, tuple[int, int]]],
        planned_targets: dict[int, tuple[int, int] | None],
        obs_builder: Any,
    ) -> int:
        env = obs_builder.env
        agent = env.agents[handle]
        mask = self._mask_from_observation(observation)

        if self._state_matches(agent.state, "DONE", "DONE_REMOVED", "WAITING"):
            return self.DO_NOTHING

        if agent.position is None:
            if mask[self.MOVE_FORWARD] >= 0.5:
                return self.MOVE_FORWARD
            return self.DO_NOTHING

        direction = agent.direction if agent.direction is not None else agent.initial_direction
        possible_transitions = env.rail.get_transitions((agent.position, direction))
        candidates = []
        for action in (self.MOVE_LEFT, self.MOVE_FORWARD, self.MOVE_RIGHT):
            new_direction = obs_builder._action_to_direction(action, direction)
            if possible_transitions[new_direction]:
                candidates.append(action)

        best_action = None
        best_score = np.inf
        action_scores = []
        for action in candidates:
            score = self._score_movement_action(
                handle,
                action,
                reserved_positions,
                reserved_edges,
                planned_targets,
                obs_builder,
            )
            action_scores.append((score, action))
            if score < best_score:
                best_score = score
                best_action = action

        wait_action = self.STOP_MOVING
        wait_score = self._score_wait_action(handle, obs_builder)
        if (
            mask[self.DO_NOTHING] >= 0.5
            and self._state_matches(agent.state, "STOPPED")
        ):
            wait_action = self.DO_NOTHING

        forward_score = min(
            (
                score
                for score, action in action_scores
                if action == self.MOVE_FORWARD
            ),
            default=np.inf,
        )
        side_candidates = [
            (score, action)
            for score, action in action_scores
            if action != self.MOVE_FORWARD and np.isfinite(score)
        ]
        if forward_score >= self.OPPOSING_PENALTY and side_candidates:
            side_score, side_action = min(side_candidates)
            if side_score < self.OPPOSING_PENALTY * 2:
                return side_action

        if best_action is not None and (
            best_score <= wait_score or best_score < self.OPPOSING_PENALTY
        ):
            return best_action
        if mask[wait_action] >= 0.5:
            return wait_action
        return self._fallback_action(mask)

    def _score_movement_action(
        self,
        handle: int,
        action: int,
        reserved_positions: set[tuple[int, int]],
        reserved_edges: set[tuple[tuple[int, int] | None, tuple[int, int]]],
        planned_targets: dict[int, tuple[int, int] | None],
        obs_builder: Any,
    ) -> float:
        env = obs_builder.env
        agent = env.agents[handle]
        target_position, target_direction = obs_builder._action_target(handle, action)
        if target_position is None or target_direction is None:
            return np.inf

        source_position = agent.position
        if target_position in reserved_positions:
            return np.inf
        if (target_position, source_position) in reserved_edges:
            return np.inf

        other = obs_builder._agent_at(target_position)
        if other != -1 and other != handle:
            return np.inf

        distance_map = obs_builder._get_distance_map(handle)
        current_direction = (
            agent.direction if agent.direction is not None else agent.initial_direction
        )
        current_dist = distance_map[
            source_position[0], source_position[1], current_direction
        ]
        new_dist = distance_map[
            target_position[0], target_position[1], target_direction
        ]
        speed = self._speed(agent)
        score = new_dist / speed if np.isfinite(new_dist) else 1e6

        if np.isfinite(current_dist) and np.isfinite(new_dist) and new_dist >= current_dist:
            score += self.NON_PROGRESS_PENALTY + (new_dist - current_dist)

        opposing, same_direction = self._scan_ahead(
            handle,
            target_position,
            target_direction,
            planned_targets,
            obs_builder,
        )
        if opposing:
            score += self.OPPOSING_PENALTY
        if same_direction:
            score += self.SAME_DIRECTION_PENALTY

        return score

    def _score_wait_action(self, handle: int, obs_builder: Any) -> float:
        agent = obs_builder.env.agents[handle]
        distance = obs_builder._current_distance_to_waypoint(handle)
        speed = self._speed(agent)
        travel_time = distance / speed if np.isfinite(distance) else 1e6
        return travel_time + self.WAIT_PENALTY

    def _scan_ahead(
        self,
        handle: int,
        position: tuple[int, int],
        direction: int,
        planned_targets: dict[int, tuple[int, int] | None],
        obs_builder: Any,
        max_cells: int = 40,
    ) -> tuple[bool, bool]:
        env = obs_builder.env
        current_position = position
        current_direction = direction
        has_same_direction = False

        for index in range(max_cells):
            if not obs_builder._is_in_bounds(current_position):
                return False, has_same_direction

            other = obs_builder._agent_at(current_position)
            if other != -1 and other != handle:
                other_agent = env.agents[other]
                if (
                    other in planned_targets
                    and planned_targets[other] is not None
                    and planned_targets[other] != other_agent.position
                ):
                    current_direction = self._next_direction(
                        current_position,
                        current_direction,
                        obs_builder,
                    )
                    if current_direction is None:
                        return False, has_same_direction
                    current_position = get_new_position(
                        current_position, current_direction
                    )
                    continue
                other_direction = (
                    other_agent.direction
                    if other_agent.direction is not None
                    else other_agent.initial_direction
                )
                if other_direction != current_direction:
                    return True, has_same_direction
                has_same_direction = True

            agents_on_switch, _, _ = obs_builder.check_agent_decision(
                current_position, current_direction
            )
            if agents_on_switch and index > 0:
                return False, has_same_direction

            possible_transitions = env.rail.get_transitions(
                (current_position, current_direction)
            )
            if fast_count_nonzero(possible_transitions) != 1:
                return False, has_same_direction

            current_direction = fast_argmax(possible_transitions)
            current_position = get_new_position(current_position, current_direction)

        return False, has_same_direction

    @staticmethod
    def _next_direction(
        position: tuple[int, int],
        direction: int,
        obs_builder: Any,
    ) -> int | None:
        possible_transitions = obs_builder.env.rail.get_transitions((position, direction))
        if fast_count_nonzero(possible_transitions) != 1:
            return None
        return fast_argmax(possible_transitions)

    def _movement_edge(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
    ) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
        if action not in (self.MOVE_LEFT, self.MOVE_FORWARD, self.MOVE_RIGHT):
            agent = obs_builder.env.agents[handle]
            return agent.position, agent.position
        target_position, _ = obs_builder._action_target(handle, action)
        return obs_builder.env.agents[handle].position, target_position

    def _priority_key(self, obs_builder: Any, handle: int) -> tuple[float, float, int]:
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

    @staticmethod
    def _fallback_action(mask: np.ndarray) -> int:
        for action in (2, 1, 3, 4, 0):
            if mask[action] >= 0.5:
                return action
        return 0

    @staticmethod
    def _speed(agent: Any) -> float:
        speed = float(agent.speed_counter.speed)
        return speed if speed > 0 else 1.0

    @staticmethod
    def _state_matches(state: TrainState, *names: str) -> bool:
        for name in names:
            candidate = getattr(TrainState, name, None)
            if candidate is not None and state == candidate:
                return True
        return False


MyPolicy = ReservationPolicy
