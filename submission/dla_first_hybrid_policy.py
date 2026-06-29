from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.my_policy import ActorCritic


DEFAULT_CHECKPOINT = "submission/models/ecml_rescue_bc_v4b_s7300.pt"


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
        self.rl_policy = ActorCritic(checkpoint_path=checkpoint)
        self.rl_policy.eval()
        self.dla_policy = None
        self.dla_failed_steps = 0
        self.max_failures = self._env_int("ECML_DLA_FIRST_MAX_FAILURES", 2)
        self.start_time = time.monotonic()
        self.max_seconds = self._env_float("ECML_DLA_FIRST_MAX_SECONDS", 1700.0)
        self.last_env_id = None
        self.last_step = None

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

            self.dla_policy = DeadLockAvoidancePolicy(
                min_free_cell=self._env_int("ECML_DLA_FIRST_MIN_FREE_CELL", 1),
                show_debug_plot=False,
                count_num_opp_agents_towards_min_free_cell=True,
                use_switches_heuristic=True,
                use_entering_prevention=False,
                use_alternative_at_first_intermediate_and_then_always_first_strategy=(
                    self._env_int("ECML_DLA_FIRST_K_ALTERNATIVES", 0)
                ),
                drop_next_threshold=None,
                k_shortest_path_cutoff=self._env_int(
                    "ECML_DLA_FIRST_K_SHORTEST_PATH_CUTOFF",
                    500,
                ),
                seed=self._env_int("ECML_DLA_FIRST_SEED", 17),
                verbose=False,
            )
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

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        return RailEnvActions(self.rl_policy.act(observation, **kwargs))

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
                return actions

        return {
            handle: RailEnvActions(action)
            for handle, action in self.rl_policy.act_many(
                handles,
                observations,
                **kwargs,
            ).items()
        }


MyPolicy = DLAFirstHybridPolicy
