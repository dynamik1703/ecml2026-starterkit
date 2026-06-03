from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.rerank_policy import RerankPolicy
from submission.reservation_policy import ReservationPolicy


class BCGatedPolicy(RerankPolicy):
    """Experimental policy that accepts only narrow BC checkpoint deviations.

    The baseline remains the guarded rerank policy using the repository
    checkpoint. A candidate BC checkpoint is queried in parallel, but its action
    is accepted only when first-difference diagnostics showed a low-risk pattern.
    This class is intentionally not the Docker default.
    """

    BC_FORWARD_MIN_LOGIT_DELTA = 1.35
    BC_FORWARD_MIN_SLACK = 100.0
    BC_LEFT_MIN_SLACK = 100.0
    BC_STOP_LEFT_MAX_DISTANCE = 30.0
    BC_STOP_LEFT_MIN_LOGIT_DELTA = 0.4

    def __init__(self, checkpoint_path: str | None = None):
        super().__init__(checkpoint_path="./submission/checkpoint.pt")
        self.bc_policy = (
            RerankPolicy(checkpoint_path=checkpoint_path)
            if checkpoint_path is not None
            else None
        )

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        baseline_actions = super().act_many(handles, observations, **kwargs)
        if self.bc_policy is None:
            return baseline_actions

        candidate_actions = self.bc_policy.act_many(handles, observations, **kwargs)
        context = runtime_context.get()
        if context.obs_builder is None:
            return baseline_actions

        adjusted = dict(baseline_actions)
        for handle, observation in zip(handles, observations):
            baseline_action = self._action_id(
                baseline_actions.get(handle, RailEnvActions.DO_NOTHING)
            )
            candidate_action = self._action_id(
                candidate_actions.get(handle, RailEnvActions.DO_NOTHING)
            )
            if baseline_action == candidate_action:
                continue
            if self._accept_bc_action(
                context.obs_builder,
                handle,
                observation,
                baseline_action,
                candidate_action,
            ):
                adjusted[handle] = RailEnvActions(candidate_action)
        return adjusted

    def _accept_bc_action(
        self,
        obs_builder: Any,
        handle: int,
        observation: Any,
        baseline_action: int,
        candidate_action: int,
    ) -> bool:
        if self.bc_policy is None:
            return False

        distance, slack = self._distance_and_slack(obs_builder, handle)
        logit_delta = self._candidate_logit_delta(
            observation,
            baseline_action,
            candidate_action,
        )

        if (
            baseline_action == ReservationPolicy.MOVE_RIGHT
            and candidate_action == ReservationPolicy.MOVE_FORWARD
        ):
            return (
                slack >= self.BC_FORWARD_MIN_SLACK
                and logit_delta >= self.BC_FORWARD_MIN_LOGIT_DELTA
                and not self._candidate_target_occupied(obs_builder, handle, candidate_action)
            )

        if (
            baseline_action == ReservationPolicy.MOVE_FORWARD
            and candidate_action == ReservationPolicy.MOVE_LEFT
        ):
            return (
                slack >= self.BC_LEFT_MIN_SLACK
                and self._candidate_branch_has_opposing(
                    obs_builder,
                    handle,
                    observation,
                    candidate_action,
                )
                and not self._candidate_target_occupied(obs_builder, handle, candidate_action)
            )

        if (
            baseline_action == ReservationPolicy.STOP_MOVING
            and candidate_action == ReservationPolicy.MOVE_LEFT
        ):
            return (
                distance <= self.BC_STOP_LEFT_MAX_DISTANCE
                and slack >= self.BC_LEFT_MIN_SLACK
                and logit_delta >= self.BC_STOP_LEFT_MIN_LOGIT_DELTA
                and not self._candidate_target_occupied(obs_builder, handle, candidate_action)
            )

        return False

    def _candidate_logit_delta(
        self,
        observation: Any,
        baseline_action: int,
        candidate_action: int,
    ) -> float:
        if self.bc_policy is None:
            return float("-inf")
        try:
            logits = self.bc_policy.rl_policy.masked_logits(
                np.asarray([observation], dtype=np.float32)
            )[0]
            return float((logits[candidate_action] - logits[baseline_action]).item())
        except Exception:
            return float("-inf")

    @staticmethod
    def _distance_and_slack(obs_builder: Any, handle: int) -> tuple[float, float]:
        try:
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = float(obs_builder._deadline_slack(handle, distance))
            return distance, slack
        except Exception:
            return float("inf"), float("-inf")

    @staticmethod
    def _candidate_target_occupied(
        obs_builder: Any,
        handle: int,
        action: int,
    ) -> bool:
        try:
            target, _ = obs_builder._action_target(handle, action)
            return target is not None and bool(obs_builder._occupied_by_other(target, handle))
        except Exception:
            return True

    @staticmethod
    def _candidate_branch_has_opposing(
        obs_builder: Any,
        handle: int,
        observation: Any,
        action: int,
    ) -> bool:
        try:
            _, target_direction = obs_builder._action_target(handle, action)
            if target_direction is None:
                return False
            values = np.asarray(observation, dtype=np.float32)
            if values.shape[0] < 18:
                return False
            return float(values[14 + target_direction]) >= 0.5
        except Exception:
            return False


MyPolicy = BCGatedPolicy
