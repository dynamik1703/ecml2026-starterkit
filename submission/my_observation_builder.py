from collections import deque

import numpy as np
from flatland.core.env_observation_builder import ObservationBuilder
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.step_utils.states import TrainState
from flatland.envs.fast_methods import fast_count_nonzero, fast_argmax

from submission import runtime_context

"""
Based on the FastTreeObs implementation by Adrian Egli (adrian.egli@gmail.com)
Extended by the Flatland Association team for environments with intermediate waypoints and additional observation features.
"""


class FastTreeObsBuilder(ObservationBuilder):

    # Fixed observation length. Indices 0..35 are documented in `get()`.
    BASE_OBSERVATION_DIM = 36
    ROUTE_OCCUPANCY_FEATURE_DIM = 8
    ROUTE_INTERSECTION_FEATURE_DIM = 8
    ROUTE_CONFLICT_FEATURE_DIM = (
        ROUTE_OCCUPANCY_FEATURE_DIM + ROUTE_INTERSECTION_FEATURE_DIM
    )
    TRAJECTORY_PRIORITY_FEATURE_DIM = 12
    ROUTE_CONFLICT_LOOKAHEAD_CELLS = 45
    SIDE_DETOUR_MARGIN = 4.0
    NEAR_TARGET_PRIORITY_DISTANCE = 20.0
    NEAR_TARGET_PRIORITY_BONUS = 200.0
    OBSERVATION_DIM = BASE_OBSERVATION_DIM

    # RailEnvActions: 0 DO_NOTHING, 1 LEFT, 2 FORWARD, 3 RIGHT, 4 STOP.
    # Mask is appended to every observation when with_action_mask=True.
    ACTION_MASK_SIZE = 5
    DO_NOTHING = 0
    MOVE_LEFT = 1
    MOVE_FORWARD = 2
    MOVE_RIGHT = 3
    STOP_MOVING = 4

    def __init__(
        self,
        max_depth=3,
        with_action_mask=True,
        with_route_conflict_features=False,
        with_trajectory_priority_features=False,
    ):
        self.max_depth = max_depth
        if with_trajectory_priority_features:
            with_route_conflict_features = True
        # Append an action mask after the feature block. Useful at inference
        # (no env access in the policy); training computes its own mask from
        # the env, so set with_action_mask=False there.
        self.with_action_mask = with_action_mask
        self.with_route_conflict_features = with_route_conflict_features
        self.with_trajectory_priority_features = with_trajectory_priority_features
        self.feature_dim = self.BASE_OBSERVATION_DIM + (
            self.ROUTE_CONFLICT_FEATURE_DIM
            if with_route_conflict_features
            else 0
        ) + (
            self.TRAJECTORY_PRIORITY_FEATURE_DIM
            if with_trajectory_priority_features
            else 0
        )
        # handle -> index into agent.waypoints of last stop visited
        self.waypoint_index = {}
        # handle -> np.ndarray (height, width, 4) BFS distance to next waypoint
        self._waypoint_distance_maps = {}
        self._switches_built = False
        self._coordination_step = None
        self._coordination_masks = {}
        self._route_prefix_cache_step = None
        self._route_prefix_cache = {}

    # ------------------------------------------------------------------
    # Topology pre-computation: switches and their neighbours.
    # ------------------------------------------------------------------

    def build_data(self):
        if self.env is not None:
            self.env.dev_obs_dict = {}
        self.switches = {}
        self.switches_neighbours = {}
        self._switches_built = False
        if self.env is not None and self.env.rail is not None:
            self.find_all_cell_where_agent_can_choose()
            self._switches_built = True

    def find_all_cell_where_agent_can_choose(self):
        """Scan the grid once and record:
          - switches: cells with >1 outgoing transition for some incoming direction.
          - switches_neighbours: cells one step before a switch.
        Both are dicts: position -> list of incoming directions where the
        property holds.
        """
        switches = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                pos = (h, w)
                for dir in range(4):
                    possible_transitions = self.env.rail.get_transitions((pos, dir))
                    if fast_count_nonzero(possible_transitions) > 1:
                        switches.setdefault(pos, []).append(dir)

        switches_neighbours = {}
        for h in range(self.env.height):
            for w in range(self.env.width):
                pos = (h, w)
                for dir in range(4):
                    possible_transitions = self.env.rail.get_transitions((pos, dir))
                    for d in range(4):
                        if possible_transitions[d] == 1:
                            new_cell = get_new_position(pos, d)
                            if new_cell in switches and pos not in switches:
                                switches_neighbours.setdefault(pos, []).append(dir)

        self.switches = switches
        self.switches_neighbours = switches_neighbours

    def check_agent_decision(self, position, direction):
        """Classify (position, direction) wrt switch topology.

        Returns:
          agents_on_switch          - on a switch and the incoming direction
                                      actually has a choice
          agents_near_to_switch     - one step before a switch (and the next
                                      cell is not itself a no-choice cell for
                                      this direction)
          agents_near_to_switch_all - one step before a switch, regardless
        """
        switches = self.switches
        switches_neighbours = self.switches_neighbours
        agents_on_switch = False
        agents_near_to_switch = False
        agents_near_to_switch_all = False

        if position in switches:
            agents_on_switch = direction in switches[position]

        if position in switches_neighbours:
            new_cell = get_new_position(position, direction)
            if new_cell in switches:
                # Only flag "near switch" if that direction does NOT itself
                # have a choice on the next cell (otherwise we'd double-count).
                if direction not in switches[new_cell]:
                    agents_near_to_switch = direction in switches_neighbours[position]
            else:
                agents_near_to_switch = direction in switches_neighbours[position]

            agents_near_to_switch_all = direction in switches_neighbours[position]

        return agents_on_switch, agents_near_to_switch, agents_near_to_switch_all

    # ------------------------------------------------------------------
    # Reset / per-handle state.
    # ------------------------------------------------------------------

    def reset(self):
        self.build_data()
        self.waypoint_index = {}
        self._waypoint_distance_maps = {}
        self._coordination_step = None
        self._coordination_masks = {}
        self._route_prefix_cache_step = None
        self._route_prefix_cache = {}
        if self.env is not None and self.env.rail is not None:
            for handle in range(self.env.get_num_agents()):
                self.waypoint_index[handle] = 0
                self._recompute_distance_map(handle)

    def _ensure_handle_initialized(self, handle):
        """Lazy init for a handle. Covers the case where reset() ran before
        env.rail was populated."""
        if not self._switches_built and self.env.rail is not None:
            self.find_all_cell_where_agent_can_choose()
            self._switches_built = True
        if handle not in self.waypoint_index:
            self.waypoint_index[handle] = 0
            self._recompute_distance_map(handle)

    @staticmethod
    def _state_matches(state, *names):
        for name in names:
            candidate = getattr(TrainState, name, None)
            if candidate is not None and state == candidate:
                return True
        return False

    # ------------------------------------------------------------------
    # Waypoint tracking and distance maps.
    # ------------------------------------------------------------------

    def _get_next_waypoint_target(self, handle):
        """Position of the next waypoint stop. waypoint_index tracks the last
        stop visited (0 = not departed). Falls back to final target once all
        intermediate stops are done."""
        agent = self.env.agents[handle]
        wps = agent.waypoints
        next_idx = self.waypoint_index[handle] + 1
        if next_idx < len(wps):
            # Take the first alternative of the waypoint group.
            return wps[next_idx][0].position
        return agent.target

    def _compute_bfs_distance_map(self, target_position):
        """BFS backward from target_position over the directed rail graph.
        Returns dist[row, col, direction], inf where unreachable."""
        rail = self.env.rail
        height, width = self.env.height, self.env.width
        dist = np.full((height, width, 4), np.inf)

        # Seed: target reachable in any direction with cost 0, but only for
        # directions that correspond to a real rail configuration.
        queue = deque()
        for d in range(4):
            if rail.get_transitions((target_position, d)) != (0, 0, 0, 0):
                dist[target_position[0], target_position[1], d] = 0
                queue.append((target_position, d, 0))

        while queue:
            pos, direction, current_dist = queue.popleft()
            # Predecessor cell = one step opposite to current direction.
            prev_pos = get_new_position(pos, (direction + 2) % 4)
            if not (0 <= prev_pos[0] < height and 0 <= prev_pos[1] < width):
                continue
            for agent_dir in range(4):
                if rail.get_transition((prev_pos, agent_dir), direction):
                    new_dist = current_dist + 1
                    if new_dist < dist[prev_pos[0], prev_pos[1], agent_dir]:
                        dist[prev_pos[0], prev_pos[1], agent_dir] = new_dist
                        queue.append((prev_pos, agent_dir, new_dist))

        return dist

    def _recompute_distance_map(self, handle):
        target_pos = self._get_next_waypoint_target(handle)
        self._waypoint_distance_maps[handle] = self._compute_bfs_distance_map(target_pos)

    def _get_distance_map(self, handle):
        """Distance map for the agent's current next waypoint. If the agent
        only has start+target (no intermediate stops), reuse the env's
        built-in distance map."""
        agent = self.env.agents[handle]
        if len(agent.waypoints) <= 2:
            return self.env.distance_map.get()[handle]
        return self._waypoint_distance_maps[handle]

    def _check_and_advance_waypoint(self, handle):
        """If the agent's current cell matches the next waypoint, advance
        the index and recompute the distance map for the new target."""
        self._ensure_handle_initialized(handle)
        agent = self.env.agents[handle]
        wps = agent.waypoints
        if len(wps) <= 2:
            return  # no intermediate stops

        next_idx = self.waypoint_index[handle] + 1
        if next_idx >= len(wps):
            return  # already at or past final target

        pos = agent.position
        if pos is None:
            return

        for wp in wps[next_idx]:
            if wp.position == pos:
                self.waypoint_index[handle] = next_idx
                self._recompute_distance_map(handle)
                break

    # ------------------------------------------------------------------
    # Forward exploration along a branch (used to detect blocking agents
    # and downstream switches).
    # ------------------------------------------------------------------

    def _explore(self, handle, new_position, new_direction, depth=0):
        has_opp_agent = 0
        has_same_agent = 0
        has_switch = 0
        visited = []

        if depth >= self.max_depth:
            return has_opp_agent, has_same_agent, has_switch, visited

        # Walk forward up to 100 cells, stopping at: a blocking agent, a
        # near-switch cell, or a switch (where we recurse on each branch).
        cnt = 0
        while cnt < 100:
            cnt += 1
            visited.append(new_position)

            # Detect another agent in this cell.
            opp_a = self._agent_at(new_position)
            if opp_a != -1 and opp_a != handle:
                if self.env.agents[opp_a].direction != new_direction:
                    has_opp_agent = 1
                else:
                    has_same_agent = 1
                return has_opp_agent, has_same_agent, has_switch, visited

            agents_on_switch, agents_near_to_switch, _ = self.check_agent_decision(
                new_position, new_direction
            )
            if agents_near_to_switch:
                return has_opp_agent, has_same_agent, has_switch, visited

            possible_transitions = self.env.rail.get_transitions(
                (new_position, new_direction)
            )
            if agents_on_switch:
                # Recurse on each outgoing branch, average results.
                f = 0
                for dir_loop in range(4):
                    if possible_transitions[dir_loop] == 1:
                        f += 1
                        hoa, hsa, hs, v = self._explore(
                            handle,
                            get_new_position(new_position, dir_loop),
                            dir_loop,
                            depth + 1,
                        )
                        visited.append(v)
                        has_opp_agent += hoa
                        has_same_agent += hsa
                        has_switch += hs
                f = max(f, 1.0)
                return has_opp_agent / f, has_same_agent / f, has_switch / f, visited
            else:
                # Straight track: follow the single transition.
                new_direction = fast_argmax(possible_transitions)
                new_position = get_new_position(new_position, new_direction)

        return has_opp_agent, has_same_agent, has_switch, visited

    # ------------------------------------------------------------------
    # Tactical action masking.
    # ------------------------------------------------------------------

    def _is_in_bounds(self, position):
        return (
            position is not None
            and 0 <= position[0] < self.env.height
            and 0 <= position[1] < self.env.width
        )

    def _agent_at(self, position):
        if not self._is_in_bounds(position):
            return -1
        try:
            handle = self.env.agent_positions[position]
            if handle != -1:
                return handle
        except (IndexError, KeyError, TypeError):
            pass
        for handle, agent in enumerate(self.env.agents):
            if agent.position == position:
                return handle
        return -1

    def _occupied_by_other(self, position, handle):
        other = self._agent_at(position)
        return other != -1 and other != handle

    def _action_to_direction(self, action, direction):
        if action == self.MOVE_LEFT:
            return (direction - 1) % 4
        if action == self.MOVE_FORWARD:
            return direction
        if action == self.MOVE_RIGHT:
            return (direction + 1) % 4
        return None

    def _scan_branch_conflict(self, handle, position, direction, max_cells=30):
        """Look ahead on a single-track branch until the next decision point."""
        current_position = position
        current_direction = direction

        for _ in range(max_cells):
            if not self._is_in_bounds(current_position):
                return False, False

            other = self._agent_at(current_position)
            if other != -1 and other != handle:
                other_agent = self.env.agents[other]
                other_direction = (
                    other_agent.direction
                    if other_agent.direction is not None
                    else other_agent.initial_direction
                )
                return other_direction != current_direction, other_direction == current_direction

            agents_on_switch, agents_near_to_switch, _ = self.check_agent_decision(
                current_position, current_direction
            )
            if agents_on_switch or agents_near_to_switch:
                return False, False

            possible_transitions = self.env.rail.get_transitions(
                (current_position, current_direction)
            )
            if fast_count_nonzero(possible_transitions) != 1:
                return False, False

            current_direction = fast_argmax(possible_transitions)
            current_position = get_new_position(current_position, current_direction)

        return False, False

    def _build_local_action_mask(self, handle):
        """Build a conservative mask from rail validity, target progress and local conflicts."""
        mask = np.zeros(self.ACTION_MASK_SIZE, dtype=np.float32)
        mask[self.DO_NOTHING] = 1.0

        self._ensure_handle_initialized(handle)
        agent = self.env.agents[handle]
        if self._state_matches(agent.state, "DONE", "DONE_REMOVED", "WAITING"):
            return mask

        if self._state_matches(agent.state, "MALFUNCTION"):
            mask[self.STOP_MOVING] = 1.0
            return mask

        direction = (
            agent.direction if agent.direction is not None else agent.initial_direction
        )

        if agent.position is None:
            if not self._occupied_by_other(agent.initial_position, handle):
                mask[self.DO_NOTHING] = 0.0
                mask[self.MOVE_FORWARD] = 1.0
            return mask

        distance_map = self._get_distance_map(handle)
        current_dist = distance_map[agent.position[0], agent.position[1], direction]
        possible_transitions = self.env.rail.get_transitions((agent.position, direction))
        agents_on_switch, agents_near_to_switch, _ = self.check_agent_decision(
            agent.position, direction
        )
        entering_corridor = agents_on_switch or agents_near_to_switch

        candidates = []
        progress_actions = []
        non_conflict_actions = []
        progress_non_conflict_actions = []
        opposite_conflict_actions = []
        conflict_seen = False

        for action in (self.MOVE_LEFT, self.MOVE_FORWARD, self.MOVE_RIGHT):
            new_direction = self._action_to_direction(action, direction)
            if not possible_transitions[new_direction]:
                continue

            new_position = get_new_position(agent.position, new_direction)
            if self._occupied_by_other(new_position, handle):
                conflict_seen = True
                continue

            candidates.append(action)

            new_dist = distance_map[new_position[0], new_position[1], new_direction]
            is_progress = (
                np.isfinite(new_dist)
                and (not np.isfinite(current_dist) or new_dist < current_dist)
            )
            has_opposite, has_same = self._scan_branch_conflict(
                handle, new_position, new_direction
            )
            is_non_conflict = not has_opposite and not has_same
            conflict_seen = conflict_seen or has_opposite or has_same

            if has_opposite:
                opposite_conflict_actions.append(action)
            if is_progress:
                progress_actions.append(action)
            if is_non_conflict:
                non_conflict_actions.append(action)
            if is_progress and is_non_conflict:
                progress_non_conflict_actions.append(action)

        if entering_corridor and opposite_conflict_actions:
            blocked = set(opposite_conflict_actions)
            candidates = [action for action in candidates if action not in blocked]
            progress_actions = [
                action for action in progress_actions if action not in blocked
            ]
            progress_non_conflict_actions = [
                action for action in progress_non_conflict_actions
                if action not in blocked
            ]
            if not candidates:
                mask[:] = 0.0
                mask[self.STOP_MOVING] = 1.0
                return mask

        preferred_actions = (
            progress_non_conflict_actions
            or non_conflict_actions
            or progress_actions
            or candidates
        )

        if preferred_actions:
            mask[:] = 0.0
            for action in preferred_actions:
                mask[action] = 1.0

            if conflict_seen or agents_on_switch or agents_near_to_switch:
                mask[self.STOP_MOVING] = 1.0
            return mask

        mask[self.DO_NOTHING] = 1.0
        mask[self.STOP_MOVING] = 1.0
        return mask

    def _current_distance_to_waypoint(self, handle):
        agent = self.env.agents[handle]
        if agent.position is None:
            position = agent.initial_position
            direction = agent.initial_direction
        else:
            position = agent.position
            direction = agent.direction

        if position is None or direction is None:
            return np.inf

        distance_map = self._get_distance_map(handle)
        distance = distance_map[position[0], position[1], direction]
        return distance if np.isfinite(distance) else np.inf

    def _deadline_slack(self, handle, distance):
        agent = self.env.agents[handle]
        next_idx = self.waypoint_index.get(handle, 0) + 1
        deadline = None
        if next_idx < len(agent.waypoints_latest_arrival):
            deadline = agent.waypoints_latest_arrival[next_idx]
        if deadline is None:
            deadline = agent.latest_arrival
        if deadline is None:
            return np.inf
        speed = float(agent.speed_counter.speed)
        travel_time = distance / speed if speed > 0 and np.isfinite(distance) else distance
        return deadline - self.env._elapsed_steps - travel_time

    def _priority_key(self, handle):
        agent = self.env.agents[handle]
        distance = self._current_distance_to_waypoint(handle)
        slack = self._deadline_slack(handle, distance)

        if self._state_matches(agent.state, "MOVING", "STOPPED", "MALFUNCTION"):
            state_priority = 0
        elif self._state_matches(agent.state, "READY_TO_DEPART"):
            state_priority = 1
        elif self._state_matches(agent.state, "WAITING"):
            state_priority = 2
        else:
            state_priority = 3

        priority_slack = slack
        if np.isfinite(distance) and distance <= self.NEAR_TARGET_PRIORITY_DISTANCE:
            priority_slack -= self.NEAR_TARGET_PRIORITY_BONUS

        return (
            state_priority,
            priority_slack if np.isfinite(priority_slack) else 1e9,
            distance if np.isfinite(distance) else 1e9,
            handle,
        )

    def _route_conflict_features(self, handle, position, direction):
        """Compact route-centric features for the next relevant train on our path.

        Layout:
          0  distance to first other train on best route, normalised by lookahead
          1  first train is opposing-direction
          2  first train is same-direction
          3  relative slack in [0,1], 0.5 neutral, lower means we are tighter
          4  other train has tighter slack / should likely have priority
          5  immediate side detour is currently available
          6  first train is stopped or malfunctioning
          7  count of occupied route cells within lookahead, clipped to 3
        """
        features = np.zeros(self.ROUTE_OCCUPANCY_FEATURE_DIM, dtype=np.float32)
        if position is None or direction is None or not self._is_in_bounds(position):
            features[0] = 1.0
            return features

        distance_map = self._get_distance_map(handle)
        current_dist = distance_map[position[0], position[1], direction]
        features[5] = float(
            self._has_immediate_side_detour(
                handle,
                position,
                direction,
                distance_map,
                current_dist,
            )
        )

        current_position = position
        current_direction = direction
        first_conflict_seen = False
        occupied_count = 0
        for cells in range(1, self.ROUTE_CONFLICT_LOOKAHEAD_CELLS + 1):
            transitions = self.env.rail.get_transitions(
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
            if not self._is_in_bounds(next_position):
                break

            other = self._agent_at(next_position)
            if other != -1 and other != handle:
                occupied_count += 1
                if not first_conflict_seen:
                    first_conflict_seen = True
                    other_agent = self.env.agents[other]
                    other_direction = (
                        other_agent.direction
                        if other_agent.direction is not None
                        else other_agent.initial_direction
                    )
                    is_opposing = other_direction != next_direction
                    _, slack, _, _ = self._priority_key(handle)
                    _, other_slack, _, _ = self._priority_key(other)
                    if np.isfinite(slack) and np.isfinite(other_slack):
                        relative_slack = np.clip(
                            (slack - other_slack)
                            / max(1, self.env._max_episode_steps),
                            -1.0,
                            1.0,
                        )
                        features[3] = 0.5 + 0.5 * relative_slack
                        features[4] = float(other_slack < slack)
                    else:
                        features[3] = 0.5

                    features[0] = cells / self.ROUTE_CONFLICT_LOOKAHEAD_CELLS
                    features[1] = float(is_opposing)
                    features[2] = float(not is_opposing)
                    features[6] = float(
                        self._state_matches(
                            other_agent.state,
                            "STOPPED",
                            "MALFUNCTION",
                        )
                    )

            current_position = next_position
            current_direction = next_direction

        if not first_conflict_seen:
            features[0] = 1.0
            features[3] = 0.5
        features[7] = min(occupied_count, 3) / 3.0
        return features

    def _route_intersection_features(self, handle):
        """Features for medium-term intersections between greedy route prefixes.

        Layout:
          0  our distance to first route intersection, normalised by lookahead
          1  other train distance to same conflict, normalised by lookahead
          2  ETA-overlap risk, 1 means both arrive at similar time
          3  other train reaches the conflict first
          4  reverse-edge / head-on route conflict
          5  crossing route conflict
          6  other train has tighter slack / should likely have priority
          7  number of other route prefixes intersecting ours, clipped to 3
        """
        features = np.zeros(self.ROUTE_INTERSECTION_FEATURE_DIM, dtype=np.float32)
        features[0] = 1.0
        features[1] = 1.0

        own_prefix = self._route_prefix(handle)
        if not own_prefix:
            return features

        own_positions = {}
        own_edges = {}
        for node in own_prefix:
            own_positions.setdefault(node["position"], node)
            own_edges.setdefault((node["prev_position"], node["position"]), node)

        best_candidate = None
        intersecting_agents = 0
        for other in self.env.get_agent_handles():
            if other == handle:
                continue
            other_prefix = self._route_prefix(other)
            if not other_prefix:
                continue

            candidate = self._best_route_intersection_candidate(
                own_positions,
                own_edges,
                other_prefix,
            )
            if candidate is None:
                continue
            intersecting_agents += 1
            if best_candidate is None or candidate["sort_key"] < best_candidate["sort_key"]:
                best_candidate = {
                    **candidate,
                    "other": other,
                }

        if best_candidate is None:
            return features

        own_step = best_candidate["own"]["step"]
        other_step = best_candidate["other_node"]["step"]
        own_eta = own_step / self._speed(self.env.agents[handle])
        other_eta = other_step / self._speed(self.env.agents[best_candidate["other"]])
        eta_gap = abs(own_eta - other_eta)
        _, slack, _, _ = self._priority_key(handle)
        _, other_slack, _, _ = self._priority_key(best_candidate["other"])

        own_direction = best_candidate["own"]["direction"]
        other_direction = best_candidate["other_node"]["direction"]
        reverse_direction = (own_direction + 2) % 4
        is_head_on = bool(best_candidate["head_on"])
        is_crossing = (
            not is_head_on
            and other_direction != own_direction
            and other_direction != reverse_direction
        )

        features[0] = own_step / self.ROUTE_CONFLICT_LOOKAHEAD_CELLS
        features[1] = other_step / self.ROUTE_CONFLICT_LOOKAHEAD_CELLS
        features[2] = max(0.0, 1.0 - eta_gap / self.ROUTE_CONFLICT_LOOKAHEAD_CELLS)
        features[3] = float(other_eta < own_eta)
        features[4] = float(is_head_on)
        features[5] = float(is_crossing)
        if np.isfinite(slack) and np.isfinite(other_slack):
            features[6] = float(other_slack < slack)
        features[7] = min(intersecting_agents, 3) / 3.0
        return features

    def _trajectory_priority_features(self, handle):
        """Decision-time trajectory context for experimental RL checkpoints.

        Layout:
          0  fraction of agents with higher priority
          1  fraction of agents with tighter effective slack
          2  fraction of agents in the same state-priority bucket
          3  own effective slack, clipped around the episode horizon
          4  valid side detour exists
          5  best side detour rejoins the forward greedy prefix
          6  best side detour divergence length
          7  best side detour prefix overlap with forward
          8  best side detour target-distance delta versus forward
          9  best side detour conflict-count delta versus forward
         10  best side detour head-on-conflict delta versus forward
         11  best side detour opposing-intersection delta versus forward
        """
        features = np.zeros(self.TRAJECTORY_PRIORITY_FEATURE_DIM, dtype=np.float32)
        try:
            own_key = self._priority_key(handle)
        except Exception:
            return features

        other_count = max(1, self.env.get_num_agents() - 1)
        higher_priority = 0
        tighter_slack = 0
        same_state_priority = 0
        for other in self.env.get_agent_handles():
            if other == handle:
                continue
            try:
                other_key = self._priority_key(other)
            except Exception:
                continue
            higher_priority += int(other_key < own_key)
            tighter_slack += int(other_key[1] < own_key[1])
            same_state_priority += int(other_key[0] == own_key[0])

        features[0] = higher_priority / other_count
        features[1] = tighter_slack / other_count
        features[2] = same_state_priority / other_count
        max_steps = max(1, self.env._max_episode_steps)
        effective_slack = own_key[1]
        features[3] = (
            0.5 + 0.5 * np.clip(effective_slack / max_steps, -1.0, 1.0)
            if np.isfinite(effective_slack)
            else 1.0
        )

        forward_prefix = self._route_prefix_for_action(self.MOVE_FORWARD, handle)
        if not forward_prefix:
            return features
        forward_summary = self._prefix_conflict_summary(forward_prefix, handle)
        forward_target_distance = self._target_distance(handle, self.MOVE_FORWARD)

        best_candidate = None
        for action in (self.MOVE_LEFT, self.MOVE_RIGHT):
            if not self._local_action_valid(handle, action):
                continue
            candidate_prefix = self._route_prefix_for_action(action, handle)
            if not candidate_prefix:
                continue
            rejoin = self._prefix_rejoin_summary(forward_prefix, candidate_prefix)
            candidate_summary = self._prefix_conflict_summary(candidate_prefix, handle)
            target_distance = self._target_distance(handle, action)
            distance_delta = (
                target_distance - forward_target_distance
                if np.isfinite(target_distance) and np.isfinite(forward_target_distance)
                else np.inf
            )
            sort_key = (
                candidate_summary["conflict_count"] - forward_summary["conflict_count"],
                candidate_summary["head_on_count"] - forward_summary["head_on_count"],
                rejoin["divergence_len"],
                distance_delta if np.isfinite(distance_delta) else 1e9,
            )
            candidate = {
                "sort_key": sort_key,
                "rejoin": rejoin,
                "summary": candidate_summary,
                "distance_delta": distance_delta,
            }
            if best_candidate is None or sort_key < best_candidate["sort_key"]:
                best_candidate = candidate

        if best_candidate is None:
            return features

        features[4] = 1.0
        rejoin = best_candidate["rejoin"]
        candidate_summary = best_candidate["summary"]
        features[5] = float(rejoin["found"])
        features[6] = min(
            rejoin["divergence_len"],
            self.ROUTE_CONFLICT_LOOKAHEAD_CELLS,
        ) / self.ROUTE_CONFLICT_LOOKAHEAD_CELLS
        features[7] = rejoin["overlap_ratio"]
        distance_delta = best_candidate["distance_delta"]
        features[8] = (
            0.5
            + 0.5
            * np.clip(
                distance_delta / max(1.0, self.SIDE_DETOUR_MARGIN * 2.0),
                -1.0,
                1.0,
            )
            if np.isfinite(distance_delta)
            else 1.0
        )
        features[9] = 0.5 + 0.5 * np.clip(
            (candidate_summary["conflict_count"] - forward_summary["conflict_count"])
            / 3.0,
            -1.0,
            1.0,
        )
        features[10] = 0.5 + 0.5 * np.clip(
            (candidate_summary["head_on_count"] - forward_summary["head_on_count"])
            / 3.0,
            -1.0,
            1.0,
        )
        features[11] = 0.5 + 0.5 * np.clip(
            (
                candidate_summary["opposing_count"]
                - forward_summary["opposing_count"]
            )
            / 3.0,
            -1.0,
            1.0,
        )
        return features

    def _route_prefix_for_action(self, action, handle):
        if action not in (self.MOVE_LEFT, self.MOVE_FORWARD, self.MOVE_RIGHT):
            return []
        agent = self.env.agents[handle]
        if self._state_matches(agent.state, "DONE", "DONE_REMOVED"):
            return []
        target_position, target_direction = self._action_target(handle, action)
        if (
            target_position is None
            or target_direction is None
            or not self._is_in_bounds(target_position)
        ):
            return []

        start_position, _ = self._agent_route_start(handle)
        if start_position is None:
            return []
        distance_map = self._get_distance_map(handle)
        prefix = [
            {
                "step": 1,
                "prev_position": start_position,
                "position": target_position,
                "direction": target_direction,
            }
        ]
        current_position = target_position
        current_direction = target_direction
        seen = {(current_position, current_direction)}
        for step_index in range(2, self.ROUTE_CONFLICT_LOOKAHEAD_CELLS + 1):
            transitions = self.env.rail.get_transitions(
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
            if not self._is_in_bounds(next_position):
                break
            state = (next_position, next_direction)
            if state in seen:
                break
            seen.add(state)
            prefix.append(
                {
                    "step": step_index,
                    "prev_position": current_position,
                    "position": next_position,
                    "direction": next_direction,
                }
            )
            current_position = next_position
            current_direction = next_direction
        return prefix

    def _local_action_valid(self, handle, action):
        try:
            mask = self._build_local_action_mask(handle)
            return bool(mask[action] >= 0.5)
        except Exception:
            return False

    def _target_distance(self, handle, action):
        try:
            target, target_direction = self._action_target(handle, action)
            if target is None or target_direction is None:
                return np.inf
            distance_map = self._get_distance_map(handle)
            distance = distance_map[target[0], target[1], target_direction]
            return float(distance) if np.isfinite(distance) else np.inf
        except Exception:
            return np.inf

    @staticmethod
    def _prefix_rejoin_summary(forward_prefix, candidate_prefix):
        forward_positions = {}
        for node in forward_prefix:
            position = node["position"]
            if position not in forward_positions:
                forward_positions[position] = node
        candidate_positions = {node["position"] for node in candidate_prefix}
        rejoin_step = 0
        for node in candidate_prefix:
            if node["position"] in forward_positions:
                rejoin_step = int(node["step"])
                break
        shared = len(set(forward_positions).intersection(candidate_positions))
        union = len(set(forward_positions).union(candidate_positions))
        return {
            "found": bool(rejoin_step),
            "divergence_len": max(0, rejoin_step - 1) if rejoin_step else len(candidate_prefix),
            "overlap_ratio": shared / union if union else 0.0,
        }

    def _prefix_conflict_summary(self, prefix, handle):
        own_positions = {}
        own_edges = {}
        for node in prefix:
            own_positions.setdefault(node["position"], node)
            own_edges.setdefault((node["prev_position"], node["position"]), node)

        conflict_count = 0
        head_on_count = 0
        opposing_count = 0
        for other in self.env.get_agent_handles():
            if other == handle:
                continue
            other_prefix = self._route_prefix(other)
            if not other_prefix:
                continue
            for other_node in other_prefix:
                own_node = own_positions.get(other_node["position"])
                if own_node is not None:
                    conflict_count += 1
                    if self._direction_relation(
                        own_node.get("direction"),
                        other_node.get("direction"),
                    ) == "opposing":
                        opposing_count += 1
                previous = other_node.get("prev_position")
                if previous is None:
                    continue
                own_node = own_edges.get((other_node["position"], previous))
                if own_node is not None:
                    conflict_count += 1
                    head_on_count += 1
        return {
            "conflict_count": conflict_count,
            "head_on_count": head_on_count,
            "opposing_count": opposing_count,
        }

    @staticmethod
    def _direction_relation(own_direction, other_direction):
        if own_direction is None or other_direction is None:
            return "crossing"
        own = int(own_direction)
        other = int(other_direction)
        if own == other:
            return "same"
        if (own + 2) % 4 == other:
            return "opposing"
        return "crossing"

    @staticmethod
    def _best_route_intersection_candidate(own_positions, own_edges, other_prefix):
        best = None
        for other_node in other_prefix:
            own_node = own_positions.get(other_node["position"])
            if own_node is not None:
                candidate = {
                    "own": own_node,
                    "other_node": other_node,
                    "head_on": False,
                    "sort_key": (
                        own_node["step"],
                        abs(own_node["step"] - other_node["step"]),
                        other_node["step"],
                        1,
                    ),
                }
                if best is None or candidate["sort_key"] < best["sort_key"]:
                    best = candidate

            reverse_edge = (other_node["position"], other_node["prev_position"])
            own_node = own_edges.get(reverse_edge)
            if own_node is not None:
                candidate = {
                    "own": own_node,
                    "other_node": other_node,
                    "head_on": True,
                    "sort_key": (
                        own_node["step"],
                        abs(own_node["step"] - other_node["step"]),
                        other_node["step"],
                        0,
                    ),
                }
                if best is None or candidate["sort_key"] < best["sort_key"]:
                    best = candidate
        return best

    def _route_prefix(self, handle):
        step = self.env._elapsed_steps
        if self._route_prefix_cache_step != step:
            self._route_prefix_cache_step = step
            self._route_prefix_cache = {}
        if handle in self._route_prefix_cache:
            return self._route_prefix_cache[handle]

        self._check_and_advance_waypoint(handle)
        position, direction = self._agent_route_start(handle)
        if position is None or direction is None or not self._is_in_bounds(position):
            self._route_prefix_cache[handle] = []
            return []

        distance_map = self._get_distance_map(handle)
        current_position = position
        current_direction = direction
        prefix = []
        for step_index in range(1, self.ROUTE_CONFLICT_LOOKAHEAD_CELLS + 1):
            transitions = self.env.rail.get_transitions(
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
            if not self._is_in_bounds(next_position):
                break
            prefix.append(
                {
                    "step": step_index,
                    "prev_position": current_position,
                    "position": next_position,
                    "direction": next_direction,
                }
            )
            current_position = next_position
            current_direction = next_direction

        self._route_prefix_cache[handle] = prefix
        return prefix

    def _agent_route_start(self, handle):
        agent = self.env.agents[handle]
        if self._state_matches(agent.state, "READY_TO_DEPART"):
            return agent.initial_position, agent.initial_direction
        if self._state_matches(agent.state, "MOVING", "STOPPED", "MALFUNCTION"):
            direction = (
                agent.direction
                if agent.direction is not None
                else agent.initial_direction
            )
            return agent.position, direction
        return None, None

    @staticmethod
    def _speed(agent):
        speed = float(agent.speed_counter.speed)
        return speed if speed > 0 else 1.0

    def _has_immediate_side_detour(
        self,
        handle,
        position,
        direction,
        distance_map,
        current_dist,
    ):
        transitions = self.env.rail.get_transitions((position, direction))
        for action in (self.MOVE_LEFT, self.MOVE_RIGHT):
            new_direction = self._action_to_direction(action, direction)
            if new_direction is None or not transitions[new_direction]:
                continue
            target = get_new_position(position, new_direction)
            if not self._is_in_bounds(target):
                continue
            if self._occupied_by_other(target, handle):
                continue
            new_dist = distance_map[target[0], target[1], new_direction]
            if not np.isfinite(new_dist):
                continue
            if np.isfinite(current_dist) and new_dist > current_dist + self.SIDE_DETOUR_MARGIN:
                continue
            return True
        return False

    @staticmethod
    def _best_progress_direction(possible_transitions, position, distance_map):
        best_direction = None
        best_distance = np.inf
        for direction, is_open in enumerate(possible_transitions):
            if not is_open:
                continue
            next_position = get_new_position(position, direction)
            if not (
                0 <= next_position[0] < distance_map.shape[0]
                and 0 <= next_position[1] < distance_map.shape[1]
            ):
                continue
            distance = distance_map[next_position[0], next_position[1], direction]
            if np.isfinite(distance) and distance < best_distance:
                best_distance = distance
                best_direction = direction
        if best_direction is not None:
            return best_direction
        if any(possible_transitions):
            return fast_argmax(possible_transitions)
        return None

    def _action_target(self, handle, action):
        agent = self.env.agents[handle]
        if action in (self.DO_NOTHING, self.STOP_MOVING):
            return agent.position, agent.direction

        direction = (
            agent.direction if agent.direction is not None else agent.initial_direction
        )
        if agent.position is None:
            return agent.initial_position, direction

        new_direction = self._action_to_direction(action, direction)
        if new_direction is None:
            return None, None
        return get_new_position(agent.position, new_direction), new_direction

    def _corridor_edges_for_action(self, handle, action, max_cells=60):
        if action not in (self.MOVE_LEFT, self.MOVE_FORWARD, self.MOVE_RIGHT):
            return []

        agent = self.env.agents[handle]
        source_position = agent.position
        target_position, target_direction = self._action_target(handle, action)
        if (
            source_position is None
            or target_position is None
            or target_direction is None
        ):
            return []

        edges = []
        previous_position = source_position
        current_position = target_position
        current_direction = target_direction

        for _ in range(max_cells):
            if not self._is_in_bounds(current_position):
                break
            edges.append((previous_position, current_position))

            agents_on_switch, agents_near_to_switch, _ = self.check_agent_decision(
                current_position, current_direction
            )
            if agents_on_switch or agents_near_to_switch:
                break

            possible_transitions = self.env.rail.get_transitions(
                (current_position, current_direction)
            )
            if fast_count_nonzero(possible_transitions) != 1:
                break

            next_direction = fast_argmax(possible_transitions)
            next_position = get_new_position(current_position, next_direction)
            previous_position = current_position
            current_position = next_position
            current_direction = next_direction

        return edges

    def _reserve_current_positions(self):
        reserved_positions = set()
        for agent in self.env.agents:
            if agent.position is not None:
                reserved_positions.add(agent.position)
        return reserved_positions

    def _ensure_step_coordination(self):
        step = self.env._elapsed_steps
        if self._coordination_step == step:
            return

        handles = list(self.env.get_agent_handles())
        for handle in handles:
            self._check_and_advance_waypoint(handle)

        local_masks = {
            handle: self._build_local_action_mask(handle)
            for handle in handles
        }
        coordinated_masks = {}
        reserved_positions = self._reserve_current_positions()
        reserved_edges = set()
        reserved_corridor_edges = set()

        for handle in sorted(handles, key=self._priority_key):
            agent = self.env.agents[handle]
            local_mask = local_masks[handle]
            mask = np.zeros(self.ACTION_MASK_SIZE, dtype=np.float32)

            for action in (self.MOVE_LEFT, self.MOVE_FORWARD, self.MOVE_RIGHT):
                if local_mask[action] < 0.5:
                    continue
                target_position, _ = self._action_target(handle, action)
                if target_position is None:
                    continue
                source_position = agent.position
                if target_position in reserved_positions:
                    continue
                edge = (source_position, target_position)
                reverse_edge = (target_position, source_position)
                if reverse_edge in reserved_edges:
                    continue
                corridor_edges = self._corridor_edges_for_action(handle, action)
                if any(
                    (target, source) in reserved_corridor_edges
                    for source, target in corridor_edges
                ):
                    continue
                mask[action] = 1.0

            if local_mask[self.STOP_MOVING] >= 0.5:
                mask[self.STOP_MOVING] = 1.0
            if local_mask[self.DO_NOTHING] >= 0.5 and not np.any(mask[1:4]):
                mask[self.DO_NOTHING] = 1.0

            if not np.any(mask):
                if local_mask[self.STOP_MOVING] >= 0.5:
                    mask[self.STOP_MOVING] = 1.0
                else:
                    mask[self.DO_NOTHING] = 1.0

            for action in (self.MOVE_LEFT, self.MOVE_FORWARD, self.MOVE_RIGHT):
                if mask[action] < 0.5:
                    continue
                target_position, _ = self._action_target(handle, action)
                if target_position is None:
                    continue
                reserved_positions.add(target_position)
                reserved_edges.add((agent.position, target_position))
                reserved_corridor_edges.update(
                    self._corridor_edges_for_action(handle, action)
                )

            if (
                mask[self.DO_NOTHING] >= 0.5
                or mask[self.STOP_MOVING] >= 0.5
            ) and agent.position is not None:
                reserved_positions.add(agent.position)

            coordinated_masks[handle] = mask

        self._coordination_step = step
        self._coordination_masks = coordinated_masks

    def _build_action_mask(self, handle):
        self._ensure_step_coordination()
        runtime_context.update(self.env, self, self._coordination_masks)
        return self._coordination_masks.get(handle, self._build_local_action_mask(handle))

    # ------------------------------------------------------------------
    # Main observation.
    # ------------------------------------------------------------------

    def get(self, handle):
        # Observation layout (all values in [0,1]):
        #  0..3   per-direction flag: this branch leads closer to the next waypoint
        #  4      state == READY_TO_DEPART
        #  5      state in {MOVING, STOPPED, MALFUNCTION}
        #  6      state in {DONE, DONE_REMOVED}
        #  7      on a switch (routing decision required)
        #  8      one step before a switch (stop-or-go decision)
        #  9      one step before/after a switch (regardless of direction choice)
        # 10..13  per-direction flag: a transition exists in that direction
        # 14..17  per-direction: opposite-direction agent ahead on this branch
        # 18..21  per-direction: same-direction agent ahead on this branch
        # 22..25  per-direction: a switch (not usable by us) lies ahead
        # 26..29  per-direction: normalised distance to next waypoint (1.0 = unreachable)
        # 30      priority signal: normalised distance from current cell to next waypoint
        # 31      elapsed-time fraction
        # 32      fraction of agents currently active
        # 33      proximity to next stop (1 at stop, decays with distance)
        # 34      fraction of intermediate waypoints already served
        # 35      time slack: fraction of remaining steps until latest arrival
        #
        # Optional route-conflict layout when with_route_conflict_features=True:
        # 36      distance to first other train on best route
        # 37      first train is opposing-direction
        # 38      first train is same-direction
        # 39      relative slack, 0.5 neutral
        # 40      other train has tighter slack
        # 41      immediate side detour is available
        # 42      first train is stopped or malfunctioning
        # 43      occupied cells on route within lookahead
        # 44      our distance to first future route intersection
        # 45      other train distance to same future route intersection
        # 46      ETA-overlap risk at future route intersection
        # 47      other train reaches future conflict first
        # 48      reverse-edge / head-on future route conflict
        # 49      crossing future route conflict
        # 50      other train has tighter slack at future conflict
        # 51      count of other route prefixes intersecting ours
        #
        # Optional trajectory-priority layout when
        # with_trajectory_priority_features=True:
        # 52      fraction of agents with higher priority
        # 53      fraction of agents with tighter effective slack
        # 54      fraction of agents in the same state-priority bucket
        # 55      own effective slack, clipped around the episode horizon
        # 56      valid side detour exists
        # 57      best side detour rejoins the forward greedy prefix
        # 58      best side detour divergence length
        # 59      best side detour prefix overlap with forward
        # 60      best side detour target-distance delta versus forward
        # 61      best side detour conflict-count delta versus forward
        # 62      best side detour head-on-conflict delta versus forward
        # 63      best side detour opposing-intersection delta versus forward

        observation = np.zeros(self.BASE_OBSERVATION_DIM, dtype=np.float32)
        visited = []
        agent = self.env.agents[handle]

        # Advance waypoint index if agent reached the next stop.
        self._check_and_advance_waypoint(handle)

        # --- State flags (obs[4..6]) and current virtual position. ---
        agent_done = False
        if agent.state == TrainState.READY_TO_DEPART:
            agent_virtual_position = agent.initial_position
            observation[4] = 1
        elif self._state_matches(agent.state, "MOVING", "STOPPED", "MALFUNCTION"):
            agent_virtual_position = agent.position
            observation[5] = 1
        else:
            observation[6] = 1
            agent_virtual_position = (-1, -1)
            agent_done = True

        # In flatland 4.2.x, READY_TO_DEPART agents have direction=None until
        # they actually enter the grid; fall back to initial_direction.
        direction = (
            agent.direction if agent.direction is not None else agent.initial_direction
        )

        max_dist = self.env.width + self.env.height

        # --- Per-branch features (obs[0..3], [10..29]). ---
        if not agent_done:
            visited.append(agent_virtual_position)
            distance_map = self._get_distance_map(handle)
            current_cell_dist = distance_map[
                agent_virtual_position[0], agent_virtual_position[1], direction
            ]
            possible_transitions = self.env.rail.get_transitions(
                (agent_virtual_position, direction)
            )
            # In a single-transition cell the "orientation" is the forced
            # outgoing direction; otherwise it equals agent direction. This
            # is the reference frame used to map (left, forward, right, back)
            # onto absolute directions.
            orientation = direction
            if fast_count_nonzero(possible_transitions) == 1:
                orientation = fast_argmax(possible_transitions)

            for dir_loop, branch_direction in enumerate(
                [(orientation + dir_loop) % 4 for dir_loop in range(-1, 3)]
            ):
                if possible_transitions[branch_direction]:
                    new_position = get_new_position(
                        agent_virtual_position, branch_direction
                    )
                    new_cell_dist = distance_map[
                        new_position[0], new_position[1], branch_direction
                    ]
                    # Binary "is this branch closer to the goal".
                    if not (np.isinf(new_cell_dist) and np.isinf(current_cell_dist)):
                        observation[dir_loop] = int(new_cell_dist < current_cell_dist)

                    # Continuous normalised distance per direction.
                    observation[26 + dir_loop] = (
                        new_cell_dist / max_dist if np.isfinite(new_cell_dist) else 1.0
                    )

                    # Walk this branch to find blockers and downstream switches.
                    has_opp_agent, has_same_agent, has_switch, v = self._explore(
                        handle, new_position, branch_direction
                    )
                    visited.append(v)

                    observation[10 + dir_loop] = 1
                    observation[14 + dir_loop] = has_opp_agent
                    observation[18 + dir_loop] = has_same_agent
                    observation[22 + dir_loop] = has_switch

        # --- Switch-context flags (obs[7..9]). ---
        agents_on_switch, agents_near_to_switch, agents_near_to_switch_all = (
            self.check_agent_decision(agent_virtual_position, direction)
        )
        observation[7] = int(agents_on_switch)
        observation[8] = int(agents_near_to_switch)
        observation[9] = int(agents_near_to_switch_all)

        # --- Global / scalar features (obs[30..35]). ---
        nb_active_agents = sum(
            1
            for h in self.env.get_agent_handles()
            if self._state_matches(
                self.env.agents[h].state, "MOVING", "STOPPED", "MALFUNCTION"
            )
        )

        # Priority signal: closer agents get lower values, letting the
        # network learn right-of-way without depending on agent IDs.
        if not agent_done:
            observation[30] = (
                current_cell_dist / max_dist if np.isfinite(current_cell_dist) else 1.0
            )
        else:
            observation[30] = 0.0

        observation[31] = (
            self.env._elapsed_steps / self.env._max_episode_steps
            if self.env._max_episode_steps > 0
            else 0.0
        )
        observation[32] = nb_active_agents / self.env.get_num_agents()

        # Proximity to next stop: 1.0 at the stop, decays linearly with distance.
        if not agent_done and np.isfinite(current_cell_dist):
            observation[33] = max(0.0, 1.0 - current_cell_dist / max(1, max_dist))
        else:
            observation[33] = 0.0

        # Fraction of intermediate waypoints served (excludes start + final target).
        wps = agent.waypoints
        n_intermediate = max(len(wps) - 2, 0)
        if n_intermediate > 0:
            served = min(max(0, self.waypoint_index[handle]), n_intermediate)
            observation[34] = served / n_intermediate
        else:
            observation[34] = 1.0

        # Time slack: fraction of episode remaining until the next stop's
        # latest-arrival deadline (or the agent's overall deadline).
        next_idx = self.waypoint_index[handle] + 1
        wps_la = agent.waypoints_latest_arrival
        deadline = None
        if next_idx < len(wps_la) and wps_la[next_idx] is not None:
            deadline = wps_la[next_idx]
        elif agent.latest_arrival is not None:
            deadline = agent.latest_arrival
        if deadline is not None:
            remaining = deadline - self.env._elapsed_steps
            max_steps = self.env._max_episode_steps
            observation[35] = max(0.0, min(1.0, remaining / max(1, max_steps)))
        else:
            observation[35] = 1.0

        self.env.dev_obs_dict.update({handle: visited})

        if self.with_route_conflict_features:
            observation = np.concatenate(
                [
                    observation,
                    self._route_conflict_features(
                        handle,
                        agent_virtual_position,
                        direction,
                    ),
                    self._route_intersection_features(handle),
                ]
            )

        if self.with_trajectory_priority_features:
            observation = np.concatenate(
                [
                    observation,
                    self._trajectory_priority_features(handle),
                ]
            )

        if not self.with_action_mask:
            return observation

        return np.concatenate([observation, self._build_action_mask(handle)])

    @staticmethod
    def agent_can_choose(observation):
        return observation[7] == 1 or observation[8] == 1


MyObservationBuilder = FastTreeObsBuilder


class RouteConflictObsBuilder(FastTreeObsBuilder):
    OBSERVATION_DIM = (
        FastTreeObsBuilder.BASE_OBSERVATION_DIM
        + FastTreeObsBuilder.ROUTE_CONFLICT_FEATURE_DIM
    )

    def __init__(self, max_depth=3, with_action_mask=True):
        super().__init__(
            max_depth=max_depth,
            with_action_mask=with_action_mask,
            with_route_conflict_features=True,
        )


MyRouteConflictObservationBuilder = RouteConflictObsBuilder


class TrajectoryConflictObsBuilder(FastTreeObsBuilder):
    OBSERVATION_DIM = (
        FastTreeObsBuilder.BASE_OBSERVATION_DIM
        + FastTreeObsBuilder.ROUTE_CONFLICT_FEATURE_DIM
        + FastTreeObsBuilder.TRAJECTORY_PRIORITY_FEATURE_DIM
    )

    def __init__(self, max_depth=3, with_action_mask=True):
        super().__init__(
            max_depth=max_depth,
            with_action_mask=with_action_mask,
            with_route_conflict_features=True,
            with_trajectory_priority_features=True,
        )


MyTrajectoryConflictObservationBuilder = TrajectoryConflictObsBuilder
