from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context


DEFAULT_CHECKPOINT = "submission/models/ecml_rescue_bc_v4b_s7300.pt"
ACTION_MASK_SIZE = 5


class DLAFirstHybridPolicy:
    """Deadlock-avoidance-first policy with an RL fallback.

    The official leaderboard shows that robust deadlock avoidance dominates
    pure PPO on dense level-1 instances. This wrapper therefore makes the
    deterministic DLA planner the primary decision maker and keeps the learned
    ActorCritic only as a fallback if DLA cannot initialize or raises.
    """

    def __init__(self, checkpoint_path: str | None = None):
        checkpoint = (
            checkpoint_path
            or os.environ.get("ECML_DLA_FIRST_CHECKPOINT", "").strip()
            or os.environ.get("ECML_ULTRA_FAST_CHECKPOINT", "").strip()
            or DEFAULT_CHECKPOINT
        )
        self.checkpoint = checkpoint
        self.rl_policy = None
        self.dla_policy = None
        self.dla_failed_steps = 0
        self.max_failures = self._env_int("ECML_DLA_FIRST_MAX_FAILURES", 2)
        self.start_time = time.monotonic()
        self.max_seconds = self._env_float("ECML_DLA_FIRST_MAX_SECONDS", 1700.0)
        self.last_env_id = None
        self.last_step = None
        self.late_rl_rescue_enabled = self._env_bool(
            "ECML_DLA_FIRST_LATE_RL_RESCUE",
            False,
        )
        self.late_rl_rescue_progress = self._env_float(
            "ECML_DLA_FIRST_LATE_RL_RESCUE_PROGRESS",
            0.75,
        )
        self.mini_locks_enabled = self._env_bool(
            "ECML_DLA_FIRST_MINI_LOCKS",
            False,
        )
        self.mini_lock_lookahead = self._env_int(
            "ECML_DLA_FIRST_MINI_LOCK_LOOKAHEAD",
            5,
        )
        self.mini_lock_min_agents = self._env_int(
            "ECML_DLA_FIRST_MINI_LOCK_MIN_AGENTS",
            20,
        )
        self.mini_lock_min_progress = self._env_float(
            "ECML_DLA_FIRST_MINI_LOCK_MIN_PROGRESS",
            0.0,
        )
        self.mini_lock_max_done_fraction = self._env_float(
            "ECML_DLA_FIRST_MINI_LOCK_MAX_DONE_FRACTION",
            1.0,
        )

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
    def _env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _action_id(action: Any) -> int:
        try:
            return int(action)
        except Exception:
            return int(getattr(action, "value", action))

    def _reset_for_env(self, env: Any) -> None:
        step = int(getattr(env, "_elapsed_steps", 0) or 0)
        env_id = id(env)
        if env_id != self.last_env_id or (
            self.last_step is not None and step < self.last_step
        ):
            self.dla_policy = None
            self.dla_failed_steps = 0
            self.last_env_id = env_id
        self.last_step = step

    def _dla(self):
        if self.dla_policy is None:
            from submission.dla_vendor.deadlock_avoidance_policy import (
                DeadLockAvoidancePolicy,
            )

            always_first = self._env_bool("ECML_DLA_FIRST_ALWAYS_FIRST", False)
            k_alternatives = self._env_int("ECML_DLA_FIRST_K_ALTERNATIVES", 0)
            strategy = 1 if always_first and k_alternatives <= 0 else k_alternatives
            self.dla_policy = DeadLockAvoidancePolicy(
                min_free_cell=self._env_int("ECML_DLA_FIRST_MIN_FREE_CELL", 1),
                show_debug_plot=False,
                count_num_opp_agents_towards_min_free_cell=self._env_bool(
                    "ECML_DLA_FIRST_COUNT_OPP_AGENTS",
                    True,
                ),
                use_switches_heuristic=self._env_bool(
                    "ECML_DLA_FIRST_USE_SWITCHES",
                    True,
                ),
                use_entering_prevention=self._env_bool(
                    "ECML_DLA_FIRST_ENTERING_PREVENTION",
                    False,
                ),
                use_alternative_at_first_intermediate_and_then_always_first_strategy=strategy,
                drop_next_threshold=(
                    self._env_int("ECML_DLA_FIRST_DROP_NEXT_THRESHOLD", -1)
                    if self._env_int("ECML_DLA_FIRST_DROP_NEXT_THRESHOLD", -1) >= 0
                    else None
                ),
                k_shortest_path_cutoff=self._env_int(
                    "ECML_DLA_FIRST_K_SHORTEST_PATH_CUTOFF",
                    500,
                ),
                seed=self._env_int("ECML_DLA_FIRST_SEED", 17),
                verbose=False,
            )
            if always_first and k_alternatives <= 0:
                self.dla_policy.use_k_alternatives_at_first_intermediate_and_then_always_first_strategy = 0
        return self.dla_policy

    def _dla_actions(
        self,
        handles: List[int],
        env: Any,
    ) -> Dict[int, RailEnvActions] | None:
        if self.dla_failed_steps >= self.max_failures:
            return None
        if self.max_seconds > 0.0 and (time.monotonic() - self.start_time) > self.max_seconds:
            return None
        try:
            env_observations = [env for _ in range(max(handles, default=-1) + 1)]
            actions = self._dla().act_many(handles, env_observations)
            self.dla_failed_steps = 0
            return {
                handle: RailEnvActions(self._action_id(action))
                for handle, action in actions.items()
            }
        except Exception:
            self.dla_failed_steps += 1
            self.dla_policy = None
            return None

    @staticmethod
    def _split_features_and_mask(
        observations: List[Any],
    ) -> tuple[Any, Any | None]:
        import numpy as np

        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim == 1:
            obs = obs[None, :]
        if obs.shape[1] <= ACTION_MASK_SIZE:
            return obs, None
        return obs[:, :-ACTION_MASK_SIZE], obs[:, -ACTION_MASK_SIZE:]

    def _ensure_rl_policy(self):
        if self.rl_policy is None:
            from submission.my_policy import ActorCritic

            self.rl_policy = ActorCritic(checkpoint_path=self.checkpoint)
            self.rl_policy.eval()
        return self.rl_policy

    def _rl_action_ids(
        self,
        observations: List[Any],
    ) -> list[int]:
        import torch

        policy = self._ensure_rl_policy()
        features, mask = self._split_features_and_mask(observations)
        with torch.no_grad():
            logits = policy.masked_logits(features, mask)
        return [int(action) for action in logits.argmax(dim=-1).cpu().numpy()]

    def _late_rl_rescue_actions(
        self,
        env: Any,
        handles: List[int],
        observations: List[Any],
        dla_actions: Dict[int, RailEnvActions],
    ) -> Dict[int, RailEnvActions]:
        if not self.late_rl_rescue_enabled:
            return dla_actions
        max_steps = max(1, int(getattr(env, "_max_episode_steps", 1) or 1))
        step = int(getattr(env, "_elapsed_steps", 0) or 0)
        if step / max_steps < self.late_rl_rescue_progress:
            return dla_actions

        rescued = dict(dla_actions)
        for handle, rl_id in zip(handles, self._rl_action_ids(observations)):
            dla_id = self._action_id(dla_actions.get(handle, RailEnvActions.DO_NOTHING))
            if dla_id in {0, 4} and rl_id in {1, 2, 3}:
                rescued[handle] = RailEnvActions(rl_id)
        return rescued

    @staticmethod
    def _agent_position_and_direction(agent: Any) -> tuple[Any | None, Any | None]:
        position = agent.position
        direction = agent.direction
        if position is None:
            position = getattr(agent, "initial_position", None)
            direction = getattr(agent, "initial_direction", None)
        return position, direction

    @staticmethod
    def _is_active_agent(agent: Any) -> bool:
        return agent.position is not None

    def _wait_action_for_agent(self, agent: Any) -> RailEnvActions:
        if self._is_active_agent(agent):
            return RailEnvActions.STOP_MOVING
        return RailEnvActions.DO_NOTHING

    @staticmethod
    def _is_done_agent(agent: Any) -> bool:
        state = getattr(agent, "state", None)
        state_name = getattr(state, "name", str(state))
        return state_name in {"DONE", "DONE_REMOVED"} or state == 6

    def _mini_locks_should_run(self, env: Any) -> bool:
        try:
            num_agents = int(env.get_num_agents())
        except Exception:
            return False
        if num_agents < self.mini_lock_min_agents:
            return False

        max_steps = max(1, int(getattr(env, "_max_episode_steps", 1) or 1))
        step = int(getattr(env, "_elapsed_steps", 0) or 0)
        if step / max_steps < self.mini_lock_min_progress:
            return False

        if self.mini_lock_max_done_fraction < 1.0:
            done = sum(int(self._is_done_agent(agent)) for agent in env.agents)
            if done / max(1, num_agents) > self.mini_lock_max_done_fraction:
                return False
        return True

    def _mini_lock_priority(self, env: Any, handle: int) -> tuple[float, float, int]:
        agent = env.agents[handle]
        step = int(getattr(env, "_elapsed_steps", 0) or 0)
        latest_arrival = getattr(agent, "latest_arrival", None)
        if latest_arrival is None:
            slack = 1e9
        else:
            slack = float(latest_arrival - step)
        path_len = 1e9
        dla_policy = self.dla_policy
        set_paths = getattr(dla_policy, "_set_paths", {}) if dla_policy is not None else {}
        path = set_paths.get(handle)
        if path is not None:
            path_len = float(len(path))
        return (slack, path_len, handle)

    def _mini_lock_action_target(
        self,
        env: Any,
        handle: int,
        action_id: int,
    ) -> tuple[Any | None, Any | None, Any | None]:
        agent = env.agents[handle]
        source, direction = self._agent_position_and_direction(agent)
        if source is None or direction is None:
            return None, None, None
        if action_id in {0, 4}:
            return source, source, direction
        if agent.position is None:
            return None, source, direction

        try:
            result = env.rail._check_action_on_agent(
                RailEnvActions(action_id),
                (source, direction),
            )
            if len(result) >= 3:
                valid = bool(result[0]) and bool(result[2])
                new_position, new_direction = result[1]
                if valid:
                    return source, new_position, new_direction
        except Exception:
            pass

        try:
            result = env.rail.check_action_on_agent(
                RailEnvActions(action_id),
                (source, direction),
            )
            if len(result) >= 3:
                valid = bool(result[0]) and bool(result[2])
                new_position, new_direction = result[1]
                if valid:
                    return source, new_position, new_direction
        except Exception:
            pass

        return None, None, None

    def _mini_lock_corridor_edges(
        self,
        env: Any,
        source: Any | None,
        target: Any | None,
        direction: Any | None,
    ) -> list[tuple[Any, Any]]:
        if source is None or target is None or direction is None or target == source:
            return []

        edges = [(source, target)]
        previous = target
        current_direction = direction
        for _ in range(max(0, self.mini_lock_lookahead - 1)):
            try:
                transitions = env.rail.get_transitions((previous, current_direction))
            except Exception:
                break
            outgoing = [
                new_direction
                for new_direction, value in enumerate(transitions)
                if value
            ]
            if len(outgoing) != 1:
                break
            next_direction = outgoing[0]
            try:
                from flatland.core.grid.grid4_utils import get_new_position

                current = get_new_position(previous, next_direction)
            except Exception:
                break
            edges.append((previous, current))
            previous = current
            current_direction = next_direction
        return edges

    def _apply_mini_sipp_locks(
        self,
        env: Any,
        handles: List[int],
        actions: Dict[int, RailEnvActions],
    ) -> Dict[int, RailEnvActions]:
        if not self.mini_locks_enabled:
            return actions
        if not self._mini_locks_should_run(env):
            return actions

        adjusted = dict(actions)
        reserved_edges: set[tuple[Any, Any]] = set()
        reserved_corridor_edges: set[tuple[Any, Any]] = set()

        for handle in sorted(handles, key=lambda h: self._mini_lock_priority(env, h)):
            agent = env.agents[handle]
            action_id = self._action_id(adjusted.get(handle, RailEnvActions.DO_NOTHING))
            source, target, direction = self._mini_lock_action_target(
                env,
                handle,
                action_id,
            )

            if action_id in {1, 2, 3} and source is not None and target is not None:
                edge = (source, target)
                corridor_edges = self._mini_lock_corridor_edges(
                    env,
                    source,
                    target,
                    direction,
                )
                reverse_conflict = (target, source) in reserved_edges or any(
                    (edge_target, edge_source) in reserved_corridor_edges
                    for edge_source, edge_target in corridor_edges
                )
                if reverse_conflict:
                    adjusted[handle] = self._wait_action_for_agent(agent)
                else:
                    reserved_edges.add(edge)
                    reserved_corridor_edges.update(corridor_edges)

        return adjusted

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        return RailEnvActions(self._ensure_rl_policy().act(observation, **kwargs))

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
                actions = self._apply_mini_sipp_locks(env, handles, actions)
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


MyPolicy = DLAFirstHybridPolicy
