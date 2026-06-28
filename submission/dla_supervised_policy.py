from __future__ import annotations

import os
import time
from typing import Any, Dict, List

from flatland.envs.rail_env_action import RailEnvActions
from flatland.envs.step_utils.states import TrainState

from submission import runtime_context
from submission.fast_competition_policy import FastCompetitionPolicy


class DLASupervisedPolicy:
    """RL/Rerank policy with a gated DLA safety supervisor.

    The base policy remains the learned/RL path. DLA is only consulted for dense
    environments and can override risky moves with a yield/stop. It can also
    rescue agents that have been stationary for several steps when DLA proposes a
    legal movement. This keeps the submission RL-centered while adding a proven
    deadlock-avoidance prior for high-density cases.
    """

    MOVING_ACTIONS = {1, 2, 3}
    YIELD_ACTIONS = {0, 4}

    def __init__(self, checkpoint_path: str | None = None):
        self.base_policy = FastCompetitionPolicy(checkpoint_path=checkpoint_path)
        self.dla_policy = None
        self.dla_failed_steps = 0
        self.start_time = time.monotonic()
        self.last_env_id = None
        self.last_step = None
        self.last_positions: dict[int, Any] = {}
        self.stationary_steps: dict[int, int] = {}
        self.min_agents = self._env_int("ECML_DLA_MIN_AGENTS", 16)
        self.min_progress = self._env_float("ECML_DLA_MIN_PROGRESS", 0.55)
        self.min_stationary_fraction = self._env_float(
            "ECML_DLA_MIN_STATIONARY_FRACTION",
            0.25,
        )
        self.max_seconds = self._env_float("ECML_DLA_MAX_SECONDS", 520.0)
        self.period = max(1, self._env_int("ECML_DLA_PERIOD", 1))
        self.max_failures = self._env_int("ECML_DLA_MAX_FAILURES", 3)
        self.rescue_stationary_steps = self._env_int(
            "ECML_DLA_RESCUE_STATIONARY_STEPS",
            4,
        )
        self.allow_rescue_moves = self._env_bool("ECML_DLA_ALLOW_RESCUE_MOVES", True)
        self.override_mode = os.environ.get(
            "ECML_DLA_OVERRIDE_MODE",
            "veto_and_rescue",
        ).strip().lower()
        self.allow_full_takeover = self._env_bool(
            "ECML_DLA_ALLOW_FULL_TAKEOVER",
            True,
        )
        self.expert_min_agents = self._env_int("ECML_DLA_EXPERT_MIN_AGENTS", 20)
        self.expert_max_seconds = self._env_float(
            "ECML_DLA_EXPERT_MAX_SECONDS",
            480.0,
        )
        self.full_takeover_progress = self._env_float(
            "ECML_DLA_FULL_TAKEOVER_PROGRESS",
            0.70,
        )
        self.full_takeover_stationary_fraction = self._env_float(
            "ECML_DLA_FULL_TAKEOVER_STATIONARY_FRACTION",
            0.40,
        )

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

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

    @staticmethod
    def _is_active_state(state: Any) -> bool:
        return state in (
            getattr(TrainState, "MOVING", None),
            getattr(TrainState, "STOPPED", None),
            getattr(TrainState, "MALFUNCTION", None),
        )

    def _reset_dla(self) -> None:
        self.dla_policy = None

    def _dla(self):
        if self.dla_policy is None:
            from submission.dla_vendor.deadlock_avoidance_policy import (
                DeadLockAvoidancePolicy,
            )

            self.dla_policy = DeadLockAvoidancePolicy(
                min_free_cell=1,
                show_debug_plot=False,
                count_num_opp_agents_towards_min_free_cell=True,
                use_switches_heuristic=True,
                use_entering_prevention=False,
                use_alternative_at_first_intermediate_and_then_always_first_strategy=3,
                k_shortest_path_cutoff=500,
                seed=17,
                verbose=False,
            )
        return self.dla_policy

    def _update_stationary_state(self, env: Any) -> None:
        step = int(getattr(env, "_elapsed_steps", 0) or 0)
        env_id = id(env)
        if env_id != self.last_env_id or (
            self.last_step is not None and step < self.last_step
        ):
            self.last_env_id = env_id
            self.last_positions = {}
            self.stationary_steps = {}
            self._reset_dla()

        for agent in env.agents:
            handle = int(agent.handle)
            position = agent.position
            previous = self.last_positions.get(handle)
            if position is not None and previous == position and self._is_active_state(agent.state):
                self.stationary_steps[handle] = self.stationary_steps.get(handle, 0) + 1
            else:
                self.stationary_steps[handle] = 0
            self.last_positions[handle] = position
        self.last_step = step

    def _stationary_fraction(self, env: Any) -> float:
        stationary = 0
        active = 0
        for agent in env.agents:
            if not self._is_active_state(agent.state):
                continue
            active += 1
            if self.stationary_steps.get(int(agent.handle), 0) >= 2:
                stationary += 1
        return stationary / max(1, active)

    def _gate_enabled(self, env: Any) -> bool:
        if env is None:
            return False
        if self.dla_failed_steps >= self.max_failures:
            return False
        if self.max_seconds > 0.0 and (time.monotonic() - self.start_time) >= self.max_seconds:
            return False
        try:
            step = int(getattr(env, "_elapsed_steps", 0) or 0)
            if step % self.period != 0:
                return False
            num_agents = int(env.get_num_agents())
            if num_agents < self.min_agents:
                return False
            max_steps = max(1, int(getattr(env, "_max_episode_steps", 1) or 1))
            progress = step / max_steps
            stationary_fraction = self._stationary_fraction(env)
            if (
                progress < self.min_progress
                and stationary_fraction < self.min_stationary_fraction
            ):
                return False
        except Exception:
            return False
        return True

    def _expert_enabled(self, env: Any) -> bool:
        if env is None:
            return False
        if self.dla_failed_steps >= self.max_failures:
            return False
        if (
            self.expert_max_seconds > 0.0
            and (time.monotonic() - self.start_time) >= self.expert_max_seconds
        ):
            return False
        try:
            return int(env.get_num_agents()) >= self.expert_min_agents
        except Exception:
            return False

    def _full_takeover_enabled(self, env: Any) -> bool:
        if not self.allow_full_takeover:
            return False
        try:
            step = int(getattr(env, "_elapsed_steps", 0) or 0)
            max_steps = max(1, int(getattr(env, "_max_episode_steps", 1) or 1))
            progress = step / max_steps
            if progress >= self.full_takeover_progress:
                return True
            return (
                self._stationary_fraction(env)
                >= self.full_takeover_stationary_fraction
            )
        except Exception:
            return False

    def _dla_actions(self, handles: List[int], env: Any) -> dict[int, int]:
        try:
            actions = self._dla().act_many(handles, [env for _ in handles])
            self.dla_failed_steps = 0
            return {handle: self._action_id(action) for handle, action in actions.items()}
        except Exception:
            self.dla_failed_steps += 1
            self._reset_dla()
            return {}

    def _merge_actions(
        self,
        env: Any,
        base_actions: Dict[int, RailEnvActions],
        dla_actions: dict[int, int],
    ) -> Dict[int, RailEnvActions]:
        if not dla_actions:
            return base_actions

        if self._full_takeover_enabled(env):
            output = dict(base_actions)
            for handle, dla_id in dla_actions.items():
                output[handle] = RailEnvActions(dla_id)
            return output

        output = dict(base_actions)
        for handle, base_action in base_actions.items():
            base_id = self._action_id(base_action)
            dla_id = dla_actions.get(handle)
            if dla_id is None:
                continue

            if base_id in self.MOVING_ACTIONS and dla_id in self.YIELD_ACTIONS:
                output[handle] = RailEnvActions(dla_id)
                continue

            if (
                self.override_mode == "veto_and_rescue"
                and self.allow_rescue_moves
                and base_id in self.YIELD_ACTIONS
                and dla_id in self.MOVING_ACTIONS
                and self.stationary_steps.get(int(handle), 0)
                >= self.rescue_stationary_steps
            ):
                output[handle] = RailEnvActions(dla_id)
        return output

    def act(self, observation: Any, **kwargs) -> RailEnvActions:
        return self.base_policy.act(observation, **kwargs)

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        context = runtime_context.get()
        env = context.env
        if env is None:
            return self.base_policy.act_many(handles, observations, **kwargs)

        self._update_stationary_state(env)
        if self._expert_enabled(env):
            dla_actions = self._dla_actions(handles, env)
            if dla_actions:
                return {
                    handle: RailEnvActions(action)
                    for handle, action in dla_actions.items()
                }

        base_actions = self.base_policy.act_many(handles, observations, **kwargs)
        if not self._gate_enabled(env):
            return base_actions

        return self._merge_actions(env, base_actions, self._dla_actions(handles, env))


MyPolicy = DLASupervisedPolicy
