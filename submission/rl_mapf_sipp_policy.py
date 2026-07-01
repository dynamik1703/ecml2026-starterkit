from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List

from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.dispatch_ranker import DispatchRanker
from submission.dla_first_hybrid_policy import DLAFirstHybridPolicy


@dataclass(frozen=True)
class PathNode:
    position: Any
    direction: Any


@dataclass
class MAPFCandidate:
    handle: int
    dla_action_id: int
    action_id: int
    wait_action: RailEnvActions
    source: Any
    target: Any
    direction: Any
    nodes: list[PathNode]
    slack: float
    path_len: float
    wait_streak: int
    rl_go_advantage: float
    occupied_target: bool
    conflict_degree: int = 0


class RLMAPFSIPPPolicy(DLAFirstHybridPolicy):
    """Rolling-horizon MAPF/SIPP layer guided by the learned action scores.

    DLA remains the path and safety prior. This policy extracts DLA path prefixes,
    builds a small time-expanded reservation table over cells and directed edges,
    and plans agents in a learned priority order. The neural policy does not get
    to emit unconstrained actions; it ranks timing decisions inside the planner.
    """

    def __init__(self, checkpoint_path: str | None = None):
        super().__init__(checkpoint_path=checkpoint_path)
        self.mapf_enabled = self._env_bool("ECML_MAPF_SIPP_ENABLED", True)
        self.mapf_disabled_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_MAPF_SIPP_DISABLED_SCENES",
                "scene_5",
            ).split(",")
            if scene.strip()
        }
        self.mapf_disabled_scene_override_min_agents = self._env_int(
            "ECML_MAPF_SIPP_DISABLED_SCENE_OVERRIDE_MIN_AGENTS",
            95,
        )
        self.mapf_min_agents = self._env_int("ECML_MAPF_SIPP_MIN_AGENTS", 35)
        self.mapf_min_steps = self._env_int("ECML_MAPF_SIPP_MIN_STEPS", 500)
        self.mapf_horizon = self._env_int("ECML_MAPF_SIPP_HORIZON", 24)
        self.mapf_max_inserted_wait = self._env_int(
            "ECML_MAPF_SIPP_MAX_INSERTED_WAIT",
            6,
        )
        self.mapf_max_waits_first_step = self._env_int(
            "ECML_MAPF_SIPP_MAX_WAITS_FIRST_STEP",
            10,
        )
        self.mapf_dense_wait_min_agents = self._env_int(
            "ECML_MAPF_SIPP_DENSE_WAIT_MIN_AGENTS",
            70,
        )
        self.mapf_dense_waits_first_step = self._env_int(
            "ECML_MAPF_SIPP_DENSE_WAITS_FIRST_STEP",
            15,
        )
        self.mapf_unlimited_wait_min_steps = self._env_int(
            "ECML_MAPF_SIPP_UNLIMITED_WAIT_MIN_STEPS",
            700,
        )
        self.mapf_allow_wait_release = self._env_bool(
            "ECML_MAPF_SIPP_ALLOW_WAIT_RELEASE",
            False,
        )
        self.mapf_release_min_wait = self._env_int(
            "ECML_MAPF_SIPP_RELEASE_MIN_WAIT",
            4,
        )
        self.mapf_release_max_slack = self._env_float(
            "ECML_MAPF_SIPP_RELEASE_MAX_SLACK",
            80.0,
        )
        self.mapf_slack_bucket = self._env_float("ECML_MAPF_SIPP_SLACK_BUCKET", 20.0)
        self.mapf_rl_weight = self._env_float("ECML_MAPF_SIPP_RL_WEIGHT", 1.0)
        self.mapf_wait_weight = self._env_float("ECML_MAPF_SIPP_WAIT_WEIGHT", 0.25)
        self.mapf_conflict_weight = self._env_float(
            "ECML_MAPF_SIPP_CONFLICT_WEIGHT",
            0.35,
        )
        self.mapf_occupied_target_penalty = self._env_float(
            "ECML_MAPF_SIPP_OCCUPIED_TARGET_PENALTY",
            0.75,
        )
        self.mapf_alternative_moves = self._env_bool(
            "ECML_MAPF_SIPP_ALTERNATIVE_MOVES",
            False,
        )
        self.mapf_preemptive_alternatives = self._env_bool(
            "ECML_MAPF_SIPP_PREEMPTIVE_ALTERNATIVES",
            False,
        )
        self.mapf_alternative_min_wait = self._env_int(
            "ECML_MAPF_SIPP_ALTERNATIVE_MIN_WAIT",
            2,
        )
        self.mapf_alternative_min_conflicts = self._env_int(
            "ECML_MAPF_SIPP_ALTERNATIVE_MIN_CONFLICTS",
            1,
        )
        self.mapf_alternative_max_distance_delta = self._env_float(
            "ECML_MAPF_SIPP_ALTERNATIVE_MAX_DISTANCE_DELTA",
            3.0,
        )
        self.mapf_alternative_min_rl_advantage = self._env_float(
            "ECML_MAPF_SIPP_ALTERNATIVE_MIN_RL_ADVANTAGE",
            -0.20,
        )
        self.mapf_alternative_max_per_step = self._env_int(
            "ECML_MAPF_SIPP_ALTERNATIVE_MAX_PER_STEP",
            3,
        )
        self.global_locks_enabled = self._env_bool(
            "ECML_MAPF_GLOBAL_LOCKS",
            False,
        )
        self.global_lock_min_edges = self._env_int(
            "ECML_MAPF_GLOBAL_LOCK_MIN_EDGES",
            4,
        )
        self.global_lock_horizon = self._env_int(
            "ECML_MAPF_GLOBAL_LOCK_HORIZON",
            12,
        )
        self.global_lock_create_horizon = self._env_int(
            "ECML_MAPF_GLOBAL_LOCK_CREATE_HORIZON",
            8,
        )
        self.global_lock_max_age = self._env_int(
            "ECML_MAPF_GLOBAL_LOCK_MAX_AGE",
            30,
        )
        self.global_lock_max_conflict_wait = self._env_int(
            "ECML_MAPF_GLOBAL_LOCK_MAX_CONFLICT_WAIT",
            5,
        )
        self.global_lock_stale_age = self._env_int(
            "ECML_MAPF_GLOBAL_LOCK_STALE_AGE",
            8,
        )
        self.global_lock_release_on_better_slack = self._env_bool(
            "ECML_MAPF_GLOBAL_LOCK_RELEASE_ON_BETTER_SLACK",
            True,
        )
        self.mapf_trace = self._env_bool("ECML_MAPF_SIPP_TRACE", False)
        self.dispatch_primary_weight = self._env_float(
            "ECML_DISPATCH_RANKER_PRIMARY_WEIGHT",
            0.0,
        )
        self.dispatch_ranker = DispatchRanker()
        self._dispatch_priority_scores: dict[int, float] = {}
        self._last_mapf_candidates: list[MAPFCandidate] = []
        self._global_locks: dict[int, dict[str, Any]] = {}
        self._global_lock_block_counts: dict[tuple[int, int], int] = {}
        self._global_lock_env_id: int | None = None
        self._global_lock_step: int | None = None

    def _mapf_should_run(self, env: Any) -> bool:
        if not self.mapf_enabled:
            return False
        try:
            num_agents = int(env.get_num_agents())
            max_steps = int(getattr(env, "_max_episode_steps", 0) or 0)
        except Exception:
            return False
        scene = runtime_context.get().scene
        if scene in self.mapf_disabled_scenes:
            if (
                self.mapf_disabled_scene_override_min_agents <= 0
                or num_agents < self.mapf_disabled_scene_override_min_agents
            ):
                return False
        if num_agents < self.mapf_min_agents:
            return False
        if max_steps < self.mapf_min_steps:
            return False
        return True

    def _path_nodes(self, env: Any, handle: int) -> list[PathNode]:
        dla_policy = self.dla_policy
        set_paths = getattr(dla_policy, "_set_paths", {}) if dla_policy is not None else {}
        path = set_paths.get(handle)
        if not path:
            return []

        nodes = [
            PathNode(
                position=getattr(waypoint, "position", None),
                direction=getattr(waypoint, "direction", None),
            )
            for waypoint in path
            if getattr(waypoint, "position", None) is not None
        ]
        if not nodes:
            return []

        agent = env.agents[handle]
        position, direction = self._agent_position_and_direction(agent)
        if position is None:
            return nodes[: self.mapf_horizon + 1]

        for index, node in enumerate(nodes):
            if node.position == position:
                trimmed = nodes[index:]
                if direction is not None and trimmed:
                    trimmed[0] = PathNode(position=position, direction=direction)
                return trimmed[: self.mapf_horizon + 1]
        return nodes[: self.mapf_horizon + 1]

    def _path_action_id(
        self,
        env: Any,
        handle: int,
        nodes: list[PathNode],
    ) -> int | None:
        agent = env.agents[handle]
        if agent.position is None:
            return 2 if nodes else None
        if len(nodes) < 2:
            return None
        next_node = nodes[1]
        for action_id in (1, 2, 3):
            source, target, direction = self._mini_lock_action_target(
                env,
                handle,
                action_id,
            )
            if (
                source == agent.position
                and target == next_node.position
                and direction == next_node.direction
            ):
                return action_id
        return None

    @staticmethod
    def _distance_at(distance_map: Any, position: Any, direction: Any) -> float:
        try:
            distance = float(distance_map[position[0], position[1], direction])
            return distance if math.isfinite(distance) else float("inf")
        except Exception:
            return float("inf")

    def _best_progress_direction(
        self,
        env: Any,
        distance_map: Any,
        position: Any,
        direction: Any,
    ) -> int | None:
        try:
            transitions = env.rail.get_transitions((position, direction))
        except Exception:
            return None

        best_direction = None
        best_distance = float("inf")
        for next_direction, is_open in enumerate(transitions):
            if not is_open:
                continue
            try:
                next_position = get_new_position(position, next_direction)
            except Exception:
                continue
            distance = self._distance_at(distance_map, next_position, next_direction)
            if distance < best_distance:
                best_distance = distance
                best_direction = next_direction
        if best_direction is not None:
            return best_direction
        for next_direction, is_open in enumerate(transitions):
            if is_open:
                return int(next_direction)
        return None

    def _greedy_nodes_from_action(
        self,
        env: Any,
        handle: int,
        action_id: int,
    ) -> list[PathNode]:
        source, target, direction = self._mini_lock_action_target(
            env,
            handle,
            action_id,
        )
        if source is None or target is None or direction is None:
            return []

        nodes = [
            PathNode(position=source, direction=direction),
            PathNode(position=target, direction=direction),
        ]
        obs_builder = runtime_context.get().obs_builder
        if obs_builder is None:
            return nodes
        try:
            distance_map = obs_builder._get_distance_map(handle)
        except Exception:
            return nodes

        current_position = target
        current_direction = direction
        seen = {(current_position, current_direction)}
        for _ in range(max(0, self.mapf_horizon - 1)):
            next_direction = self._best_progress_direction(
                env,
                distance_map,
                current_position,
                current_direction,
            )
            if next_direction is None:
                break
            try:
                next_position = get_new_position(current_position, next_direction)
            except Exception:
                break
            state = (next_position, next_direction)
            if state in seen:
                break
            seen.add(state)
            nodes.append(PathNode(position=next_position, direction=next_direction))
            current_position = next_position
            current_direction = next_direction
        return nodes[: self.mapf_horizon + 1]

    def _action_distance_delta(
        self,
        handle: int,
        source: Any,
        target: Any,
        direction: Any,
    ) -> float:
        obs_builder = runtime_context.get().obs_builder
        if obs_builder is None:
            return float("inf")
        try:
            distance_map = obs_builder._get_distance_map(handle)
        except Exception:
            return float("inf")
        current_distance = self._distance_at(
            distance_map,
            source,
            getattr(runtime_context.get().env.agents[handle], "direction", direction)
            if runtime_context.get().env is not None
            else direction,
        )
        next_distance = self._distance_at(distance_map, target, direction)
        if not math.isfinite(next_distance):
            return float("inf")
        if not math.isfinite(current_distance):
            return 0.0
        return next_distance - current_distance

    def _alternative_candidate(
        self,
        env: Any,
        candidate: MAPFCandidate,
        action_id: int,
        timing_scores: dict[int, dict[str, float]],
        reserved_cells: dict[int, set[Any]],
        reserved_edges: dict[int, set[tuple[Any, Any]]],
    ) -> MAPFCandidate | None:
        if action_id == candidate.action_id or action_id not in {1, 2, 3}:
            return None
        if not self._action_is_allowed_by_mask(candidate.handle, action_id, None):
            return None

        source, target, direction = self._mini_lock_action_target(
            env,
            candidate.handle,
            action_id,
        )
        if source is None or target is None or direction is None or source == target:
            return None
        if target != source and self._cell_reserved(reserved_cells, 0, target):
            return None
        if not self._can_reserve_move(
            reserved_cells,
            reserved_edges,
            1,
            source,
            target,
        ):
            return None

        distance_delta = self._action_distance_delta(
            candidate.handle,
            source,
            target,
            direction,
        )
        if distance_delta > self.mapf_alternative_max_distance_delta:
            return None

        rl_go_advantage = self._go_advantage(
            env,
            candidate.handle,
            action_id,
            timing_scores,
        )
        if rl_go_advantage < self.mapf_alternative_min_rl_advantage:
            return None

        nodes = self._greedy_nodes_from_action(env, candidate.handle, action_id)
        if len(nodes) < 2:
            return None

        return MAPFCandidate(
            handle=candidate.handle,
            dla_action_id=candidate.dla_action_id,
            action_id=action_id,
            wait_action=candidate.wait_action,
            source=source,
            target=target,
            direction=direction,
            nodes=nodes,
            slack=candidate.slack,
            path_len=max(0.0, candidate.path_len + distance_delta),
            wait_streak=candidate.wait_streak,
            rl_go_advantage=rl_go_advantage,
            occupied_target=False,
            conflict_degree=candidate.conflict_degree,
        )

    def _should_try_alternative(self, candidate: MAPFCandidate) -> bool:
        if not self.mapf_alternative_moves:
            return False
        if candidate.occupied_target:
            return True
        if (
            self.mapf_alternative_min_conflicts > 0
            and candidate.conflict_degree >= self.mapf_alternative_min_conflicts
        ):
            return True
        return candidate.wait_streak >= self.mapf_alternative_min_wait

    def _try_alternative_action(
        self,
        env: Any,
        candidate: MAPFCandidate,
        candidates: list[MAPFCandidate],
        timing_scores: dict[int, dict[str, float]],
        reserved_cells: dict[int, set[Any]],
        reserved_edges: dict[int, set[tuple[Any, Any]]],
        preemptive: bool = False,
    ) -> RailEnvActions | None:
        if not self._should_try_alternative(candidate):
            return None

        baseline_conflicts = self._future_conflict_count(candidate, candidates)
        if preemptive and baseline_conflicts <= 0:
            return None

        alternatives = []
        for action_id in (1, 2, 3):
            alternative = self._alternative_candidate(
                env,
                candidate,
                action_id,
                timing_scores,
                reserved_cells,
                reserved_edges,
            )
            if alternative is None:
                continue
            distance_delta = self._action_distance_delta(
                alternative.handle,
                alternative.source,
                alternative.target,
                alternative.direction,
            )
            future_conflicts = self._future_conflict_count(alternative, candidates)
            if preemptive and future_conflicts >= baseline_conflicts:
                continue
            alternatives.append(
                (
                    future_conflicts,
                    distance_delta,
                    -alternative.rl_go_advantage,
                    alternative.action_id,
                    alternative,
                )
            )
        if not alternatives:
            return None

        _, _, _, _, alternative = min(alternatives)
        return self._plan_candidate(
            env,
            alternative,
            candidates,
            reserved_cells,
            reserved_edges,
            timing_scores,
            allow_alternative=False,
        )

    def _current_occupancy(self, env: Any) -> dict[Any, int]:
        occupancy = {}
        for other, agent in enumerate(env.agents):
            if agent.position is not None and not self._is_done_agent(agent):
                occupancy[agent.position] = other
        return occupancy

    def _action_is_allowed_by_mask(
        self,
        handle: int,
        action_id: int,
        observation: Any | None,
    ) -> bool:
        return self._mask_allows_action(handle, action_id, observation)

    def _rl_scores(
        self,
        handles: List[int],
        observations: List[Any],
    ) -> dict[int, dict[str, float]]:
        if len(handles) != len(observations):
            return {}
        try:
            return self._rl_timing_scores(handles, observations)
        except Exception:
            return {}

    def _go_advantage(
        self,
        env: Any,
        handle: int,
        action_id: int,
        timing_scores: dict[int, dict[str, float]],
    ) -> float:
        scores = timing_scores.get(handle)
        if not scores:
            return 0.0
        wait_id = self._action_id(self._wait_action_for_agent(env.agents[handle]))
        return scores.get(f"score_{action_id}", 0.0) - scores.get(
            f"score_{wait_id}",
            0.0,
        )

    def _candidate_from_dla_action(
        self,
        env: Any,
        handle: int,
        observation: Any | None,
        actions: Dict[int, RailEnvActions],
        timing_scores: dict[int, dict[str, float]],
        current_occupancy: dict[Any, int],
    ) -> MAPFCandidate | None:
        agent = env.agents[handle]
        if self._is_done_agent(agent):
            return None

        nodes = self._path_nodes(env, handle)
        dla_action_id = self._action_id(actions.get(handle, RailEnvActions.DO_NOTHING))
        action_id = dla_action_id
        source, target, direction = self._mini_lock_action_target(
            env,
            handle,
            action_id,
        )
        if action_id not in {1, 2, 3} or target is None or source == target:
            if not self.mapf_allow_wait_release:
                return None
            if self.wait_streaks.get(handle, 0) < self.mapf_release_min_wait:
                return None
            slack, _, _ = self._mini_lock_priority(env, handle)
            if slack > self.mapf_release_max_slack:
                return None
            released_action = self._path_action_id(env, handle, nodes)
            if released_action not in {1, 2, 3}:
                return None
            if not self._action_is_allowed_by_mask(handle, released_action, observation):
                return None
            source, target, direction = self._mini_lock_action_target(
                env,
                handle,
                released_action,
            )
            if target is None or source == target:
                return None
            action_id = int(released_action)

        if agent.position is not None and source is None:
            return None
        if agent.position is None:
            source = getattr(agent, "initial_position", None)
            direction = getattr(agent, "initial_direction", None)
            target = source
        if source is None or target is None:
            return None
        if agent.position is not None and not self._action_is_allowed_by_mask(
            handle,
            action_id,
            observation,
        ):
            return None

        if not nodes:
            nodes = [
                PathNode(position=source, direction=direction),
                PathNode(position=target, direction=direction),
            ]
        elif nodes[0].position != source:
            nodes = [PathNode(position=source, direction=direction)] + nodes
        if len(nodes) == 1 and target != source:
            nodes.append(PathNode(position=target, direction=direction))

        slack, path_len, _ = self._mini_lock_priority(env, handle)
        occupant = current_occupancy.get(target)
        occupied_target = occupant is not None and occupant != handle
        return MAPFCandidate(
            handle=handle,
            dla_action_id=dla_action_id,
            action_id=action_id,
            wait_action=self._wait_action_for_agent(agent),
            source=source,
            target=target,
            direction=direction,
            nodes=nodes[: self.mapf_horizon + 1],
            slack=slack,
            path_len=path_len,
            wait_streak=self.wait_streaks.get(handle, 0),
            rl_go_advantage=self._go_advantage(
                env,
                handle,
                action_id,
                timing_scores,
            ),
            occupied_target=occupied_target,
        )

    @staticmethod
    def _edge(source: Any, target: Any) -> tuple[Any, Any]:
        return (source, target)

    @staticmethod
    def _reverse_edge(edge: tuple[Any, Any]) -> tuple[Any, Any]:
        return (edge[1], edge[0])

    def _candidate_edges(
        self,
        candidate: MAPFCandidate,
        max_edges: int | None = None,
    ) -> tuple[tuple[Any, Any], ...]:
        limit = max_edges if max_edges is not None else self.global_lock_horizon
        edges = []
        for source, target in zip(candidate.nodes, candidate.nodes[1:]):
            if source.position is None or target.position is None:
                continue
            if source.position == target.position:
                continue
            edges.append(self._edge(source.position, target.position))
            if limit > 0 and len(edges) >= limit:
                break
        return tuple(edges)

    @staticmethod
    def _edge_cells(edges: tuple[tuple[Any, Any], ...]) -> set[Any]:
        cells = set()
        for source, target in edges:
            cells.add(source)
            cells.add(target)
        return cells

    def _annotate_conflict_degree(
        self,
        candidates: list[MAPFCandidate],
    ) -> None:
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                if left.target == right.target:
                    left.conflict_degree += 1
                    right.conflict_degree += 1
                    continue
                if left.target == right.source and right.target == left.source:
                    left.conflict_degree += 1
                    right.conflict_degree += 1
                    continue
                left_edges = {
                    self._edge(a.position, b.position)
                    for a, b in zip(left.nodes, left.nodes[1:])
                }
                right_edges = {
                    self._edge(a.position, b.position)
                    for a, b in zip(right.nodes, right.nodes[1:])
                }
                if any(self._reverse_edge(edge) in right_edges for edge in left_edges):
                    left.conflict_degree += 1
                    right.conflict_degree += 1

    def _priority_key(self, candidate: MAPFCandidate) -> tuple[float, float, float, int]:
        slack_key = (
            candidate.slack
            if self.mapf_slack_bucket <= 0.0
            else candidate.slack // self.mapf_slack_bucket
        )
        dispatch_score = self._dispatch_priority_scores.get(candidate.handle, 0.0)
        score = (
            self.mapf_rl_weight * candidate.rl_go_advantage
            + self.mapf_wait_weight * float(candidate.wait_streak)
            + self.mapf_conflict_weight * float(candidate.conflict_degree)
            - self.mapf_occupied_target_penalty * float(candidate.occupied_target)
            + dispatch_score
        )
        slack_key = slack_key - self.dispatch_primary_weight * dispatch_score
        path_len = candidate.path_len if math.isfinite(candidate.path_len) else 1.0e9
        return (slack_key, -score, path_len, candidate.handle)

    def _reserve_initial_state(
        self,
        env: Any,
    ) -> tuple[dict[int, set[Any]], dict[int, set[tuple[Any, Any]]]]:
        reserved_cells: dict[int, set[Any]] = {0: set()}
        reserved_edges: dict[int, set[tuple[Any, Any]]] = {}
        for agent in env.agents:
            if agent.position is not None and not self._is_done_agent(agent):
                reserved_cells[0].add(agent.position)
        return reserved_cells, reserved_edges

    def _sync_global_locks(self, env: Any) -> None:
        if not self.global_locks_enabled:
            self._global_locks = {}
            self._global_lock_block_counts = {}
            return

        step = int(getattr(env, "_elapsed_steps", 0) or 0)
        env_id = id(env)
        if (
            self._global_lock_env_id != env_id
            or self._global_lock_step is None
            or step < self._global_lock_step
        ):
            self._global_locks = {}
            self._global_lock_block_counts = {}
            self._global_lock_env_id = env_id
        self._global_lock_step = step

        active_locks = {}
        active_owners = set()
        for owner, lock in self._global_locks.items():
            if owner >= len(env.agents):
                continue
            agent = env.agents[owner]
            if self._is_done_agent(agent):
                continue
            if agent.position is None:
                continue
            if step - int(lock["step"]) > self.global_lock_max_age:
                continue
            if agent.position not in lock["cells"]:
                continue
            active_locks[owner] = lock
            active_owners.add(owner)

        self._global_locks = active_locks
        self._global_lock_block_counts = {
            key: count
            for key, count in self._global_lock_block_counts.items()
            if key[0] in active_owners
        }

    def _candidate_conflicts_lock(
        self,
        candidate: MAPFCandidate,
        lock: dict[str, Any],
    ) -> bool:
        edges = self._candidate_edges(candidate)
        if not edges:
            return False
        lock_edges = lock["edges"]
        if any(self._reverse_edge(edge) in lock_edges for edge in edges):
            return True
        return False

    def _lock_age(self, env: Any, lock: dict[str, Any]) -> int:
        return int(getattr(env, "_elapsed_steps", 0) or 0) - int(lock["step"])

    def _should_yield_to_global_lock(
        self,
        env: Any,
        candidate: MAPFCandidate,
        lock: dict[str, Any],
    ) -> bool:
        owner = int(lock["owner"])
        if owner == candidate.handle:
            return False

        age = self._lock_age(env, lock)
        if age >= self.global_lock_stale_age:
            return False

        if self.global_lock_release_on_better_slack:
            lock_priority = lock.get("priority")
            if lock_priority is not None:
                candidate_priority = self._priority_key(candidate)
                if candidate_priority < lock_priority and candidate.slack + 1e-6 < lock.get(
                    "slack",
                    1.0e9,
                ):
                    return False

        key = (owner, candidate.handle)
        wait_count = self._global_lock_block_counts.get(key, 0)
        if wait_count >= self.global_lock_max_conflict_wait:
            return False
        self._global_lock_block_counts[key] = wait_count + 1
        return True

    def _global_lock_waits(
        self,
        env: Any,
        candidates: list[MAPFCandidate],
    ) -> set[int]:
        if not self.global_locks_enabled or not self._global_locks:
            return set()

        hard_waits = set()
        for candidate in sorted(candidates, key=self._priority_key):
            for lock in list(self._global_locks.values()):
                if not self._candidate_conflicts_lock(candidate, lock):
                    continue
                if self._should_yield_to_global_lock(env, candidate, lock):
                    hard_waits.add(candidate.handle)
                    break
        return hard_waits

    def _candidate_has_opposing_future_conflict(
        self,
        candidate: MAPFCandidate,
        candidates: list[MAPFCandidate],
    ) -> bool:
        edges = self._candidate_edges(candidate, self.global_lock_create_horizon)
        if len(edges) < self.global_lock_min_edges:
            return False
        edge_set = set(edges)
        for other in candidates:
            if other.handle == candidate.handle:
                continue
            other_edges = self._candidate_edges(other, self.global_lock_create_horizon)
            if any(self._reverse_edge(edge) in edge_set for edge in other_edges):
                return True
        for lock in self._global_locks.values():
            if lock["owner"] == candidate.handle:
                continue
            if any(self._reverse_edge(edge) in lock["edges"] for edge in edges):
                return True
        return False

    def _future_conflict_count(
        self,
        candidate: MAPFCandidate,
        candidates: list[MAPFCandidate],
    ) -> int:
        edges = self._candidate_edges(candidate, self.global_lock_create_horizon)
        if not edges:
            return 0

        edge_set = set(edges)
        count = 0
        for other in candidates:
            if other.handle == candidate.handle:
                continue
            other_edges = self._candidate_edges(other, self.global_lock_create_horizon)
            if any(self._reverse_edge(edge) in edge_set for edge in other_edges):
                count += 1
            elif any(edge in other_edges for edge in edge_set):
                count += 1
        return count

    def _make_global_lock(
        self,
        env: Any,
        candidate: MAPFCandidate,
    ) -> dict[str, Any] | None:
        edges = self._candidate_edges(candidate, self.global_lock_horizon)
        if len(edges) < self.global_lock_min_edges:
            return None
        return {
            "owner": candidate.handle,
            "step": int(getattr(env, "_elapsed_steps", 0) or 0),
            "edges": frozenset(edges),
            "cells": self._edge_cells(edges),
            "priority": self._priority_key(candidate),
            "slack": candidate.slack,
        }

    def _update_global_locks(
        self,
        env: Any,
        candidates: list[MAPFCandidate],
        planned: Dict[int, RailEnvActions],
        hard_waits: set[int],
    ) -> None:
        if not self.global_locks_enabled:
            return

        for candidate in sorted(candidates, key=self._priority_key):
            if candidate.handle in hard_waits:
                continue
            planned_action_id = self._action_id(
                planned.get(candidate.handle, RailEnvActions.DO_NOTHING)
            )
            if planned_action_id not in {1, 2, 3}:
                continue
            if not self._candidate_has_opposing_future_conflict(candidate, candidates):
                continue
            lock = self._make_global_lock(env, candidate)
            if lock is not None:
                self._global_locks[candidate.handle] = lock

    def _first_step_wait_limit(self, env: Any) -> int:
        wait_limit = self.mapf_max_waits_first_step
        try:
            num_agents = int(env.get_num_agents())
        except Exception:
            num_agents = 0
        if (
            self.mapf_dense_wait_min_agents > 0
            and num_agents >= self.mapf_dense_wait_min_agents
        ):
            wait_limit = self.mapf_dense_waits_first_step
        if self.mapf_unlimited_wait_min_steps <= 0:
            return wait_limit
        try:
            max_steps = int(getattr(env, "_max_episode_steps", 0) or 0)
        except Exception:
            return wait_limit
        if max_steps >= self.mapf_unlimited_wait_min_steps:
            return 0
        return wait_limit

    @staticmethod
    def _cell_reserved(
        reserved_cells: dict[int, set[Any]],
        time_index: int,
        position: Any,
    ) -> bool:
        return position in reserved_cells.get(time_index, set())

    @staticmethod
    def _edge_reserved(
        reserved_edges: dict[int, set[tuple[Any, Any]]],
        time_index: int,
        edge: tuple[Any, Any],
    ) -> bool:
        return edge in reserved_edges.get(time_index, set())

    def _can_reserve_move(
        self,
        reserved_cells: dict[int, set[Any]],
        reserved_edges: dict[int, set[tuple[Any, Any]]],
        time_index: int,
        source: Any,
        target: Any,
    ) -> bool:
        if self._cell_reserved(reserved_cells, time_index, target):
            return False
        edge = self._edge(source, target)
        if self._edge_reserved(reserved_edges, time_index, self._reverse_edge(edge)):
            return False
        return True

    def _reserve_wait(
        self,
        reserved_cells: dict[int, set[Any]],
        time_index: int,
        position: Any,
    ) -> None:
        reserved_cells.setdefault(time_index, set()).add(position)

    def _reserve_move(
        self,
        reserved_cells: dict[int, set[Any]],
        reserved_edges: dict[int, set[tuple[Any, Any]]],
        time_index: int,
        source: Any,
        target: Any,
    ) -> None:
        reserved_cells.setdefault(time_index, set()).add(target)
        reserved_edges.setdefault(time_index, set()).add(self._edge(source, target))

    def _plan_candidate(
        self,
        env: Any,
        candidate: MAPFCandidate,
        candidates: list[MAPFCandidate],
        reserved_cells: dict[int, set[Any]],
        reserved_edges: dict[int, set[tuple[Any, Any]]],
        timing_scores: dict[int, dict[str, float]],
        allow_alternative: bool = True,
    ) -> RailEnvActions:
        if (
            allow_alternative
            and self.mapf_preemptive_alternatives
            and not candidate.occupied_target
        ):
            alternative = self._try_alternative_action(
                env,
                candidate,
                candidates,
                timing_scores,
                reserved_cells,
                reserved_edges,
                preemptive=True,
            )
            if alternative is not None:
                return alternative

        if candidate.occupied_target:
            if allow_alternative:
                alternative = self._try_alternative_action(
                    env,
                    candidate,
                    candidates,
                    timing_scores,
                    reserved_cells,
                    reserved_edges,
                )
                if alternative is not None:
                    return alternative
            self._reserve_wait(reserved_cells, 1, candidate.source)
            return candidate.wait_action

        if len(candidate.nodes) < 2:
            self._reserve_wait(reserved_cells, 1, candidate.source)
            return candidate.wait_action

        current = candidate.nodes[0].position
        node_index = 1
        time_index = 1
        inserted_waits = 0
        first_action: RailEnvActions | None = None

        while time_index <= max(1, self.mapf_horizon) and node_index < len(candidate.nodes):
            target = candidate.nodes[node_index].position
            if self._can_reserve_move(
                reserved_cells,
                reserved_edges,
                time_index,
                current,
                target,
            ):
                self._reserve_move(
                    reserved_cells,
                    reserved_edges,
                    time_index,
                    current,
                    target,
                )
                if first_action is None:
                    first_action = RailEnvActions(candidate.action_id)
                current = target
                node_index += 1
                time_index += 1
                continue

            if inserted_waits >= self.mapf_max_inserted_wait:
                if first_action is None:
                    if allow_alternative:
                        alternative = self._try_alternative_action(
                            env,
                            candidate,
                            candidates,
                            timing_scores,
                            reserved_cells,
                            reserved_edges,
                        )
                        if alternative is not None:
                            return alternative
                    self._reserve_wait(reserved_cells, 1, candidate.source)
                    return candidate.wait_action
                break

            if self._cell_reserved(reserved_cells, time_index, current):
                if first_action is None:
                    if allow_alternative:
                        alternative = self._try_alternative_action(
                            env,
                            candidate,
                            candidates,
                            timing_scores,
                            reserved_cells,
                            reserved_edges,
                        )
                        if alternative is not None:
                            return alternative
                    return candidate.wait_action
                break
            self._reserve_wait(reserved_cells, time_index, current)
            if first_action is None:
                first_action = candidate.wait_action
            inserted_waits += 1
            time_index += 1

        return first_action or candidate.wait_action

    def _apply_mapf_sipp(
        self,
        env: Any,
        handles: List[int],
        observations: List[Any],
        actions: Dict[int, RailEnvActions],
    ) -> Dict[int, RailEnvActions]:
        self._dispatch_priority_scores = {}
        self._last_mapf_candidates = []
        if not self._mapf_should_run(env):
            return actions

        self._sync_global_locks(env)
        timing_scores = self._rl_scores(handles, observations)
        observation_by_handle = {
            handle: observation
            for handle, observation in zip(handles, observations)
        }
        current_occupancy = self._current_occupancy(env)
        candidates = []
        for handle in handles:
            candidate = self._candidate_from_dla_action(
                env,
                handle,
                observation_by_handle.get(handle),
                actions,
                timing_scores,
                current_occupancy,
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            return actions

        self._annotate_conflict_degree(candidates)
        self._last_mapf_candidates = list(candidates)
        self._dispatch_priority_scores = self.dispatch_ranker.priority_scores(
            env,
            candidates,
        )
        hard_waits = self._global_lock_waits(env, candidates)
        reserved_cells, reserved_edges = self._reserve_initial_state(env)
        planned = dict(actions)
        waits_added = 0
        first_step_wait_limit = self._first_step_wait_limit(env)
        for candidate in sorted(candidates, key=self._priority_key):
            if candidate.handle in hard_waits:
                self._reserve_wait(reserved_cells, 1, candidate.source)
                planned_action = candidate.wait_action
            else:
                planned_action = self._plan_candidate(
                    env,
                    candidate,
                    candidates,
                    reserved_cells,
                    reserved_edges,
                    timing_scores,
                )
            if (
                self._action_id(planned_action) in {0, 4}
                and candidate.dla_action_id in {1, 2, 3}
            ):
                waits_added += 1
                if (
                    candidate.handle not in hard_waits
                    and first_step_wait_limit > 0
                    and waits_added > first_step_wait_limit
                ):
                    planned_action = RailEnvActions(candidate.dla_action_id)
            planned[candidate.handle] = planned_action

        self._update_global_locks(env, candidates, planned, hard_waits)
        return planned

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
                actions = self._apply_mapf_sipp(
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


MyPolicy = RLMAPFSIPPPolicy
