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
