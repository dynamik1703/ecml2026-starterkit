from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.my_policy import ActorCritic
from submission.rerank_policy import RerankPolicy
from submission.sequence_success_policy import (
    DEFAULT_CANDIDATE_CHECKPOINT_PATHS,
    SequenceSuccessPolicy,
)


class RiskVetoPolicy:
    """Direct RL proposal policy with a learned action-risk veto.

    The safe Sequence policy remains the baseline. A direct RL/Rerank candidate
    may override it only when the risk head considers the candidate action safe.
    This lets us test stronger RL proposal policies without deploying them
    unfiltered.
    """

    def __init__(self, checkpoint_path: str | None = None):
        candidate_checkpoint = (
            checkpoint_path
            or os.environ.get("ECML_RISK_VETO_CANDIDATE_CHECKPOINT", "").strip()
            or DEFAULT_CANDIDATE_CHECKPOINT_PATHS[-1]
        )
        risk_checkpoint = os.environ.get(
            "ECML_RISK_VETO_RISK_CHECKPOINT",
            "",
        ).strip()
        self.baseline_policy = SequenceSuccessPolicy()
        self.candidate_policy = RerankPolicy(checkpoint_path=candidate_checkpoint)
        self.top_n_candidate_actions = max(
            1,
            int(os.environ.get("ECML_RISK_VETO_TOP_N_CANDIDATE_ACTIONS", "1") or "1"),
        )
        self.top_n_allowed_transitions = self._env_transition_set(
            "ECML_RISK_VETO_TOP_N_ALLOWED_TRANSITIONS",
        )
        self.risk_policy = (
            ActorCritic(checkpoint_path=risk_checkpoint)
            if risk_checkpoint and Path(risk_checkpoint).exists()
            else None
        )
        if self.risk_policy is not None:
            self.risk_policy.eval()
        reward_risk_checkpoint = os.environ.get(
            "ECML_RISK_VETO_REWARD_RISK_CHECKPOINT",
            "",
        ).strip()
        self.reward_risk_policy = (
            ActorCritic(checkpoint_path=reward_risk_checkpoint)
            if reward_risk_checkpoint and Path(reward_risk_checkpoint).exists()
            else None
        )
        if self.reward_risk_policy is not None:
            self.reward_risk_policy.eval()
        value_checkpoint = os.environ.get(
            "ECML_RISK_VETO_VALUE_CHECKPOINT",
            "",
        ).strip()
        self.value_policy = (
            ActorCritic(checkpoint_path=value_checkpoint)
            if value_checkpoint and Path(value_checkpoint).exists()
            else None
        )
        if self.value_policy is not None:
            self.value_policy.eval()
        self.max_candidate_risk = self._env_float(
            "ECML_RISK_VETO_MAX_CANDIDATE",
            0.50,
        )
        self.max_candidate_minus_baseline = self._env_float(
            "ECML_RISK_VETO_MAX_CANDIDATE_MINUS_BASELINE",
            0.00,
        )
        self.min_baseline_minus_candidate = self._env_float(
            "ECML_RISK_VETO_MIN_BASELINE_MINUS_CANDIDATE",
            float("-inf"),
        )
        self.max_reward_risk = self._env_float(
            "ECML_RISK_VETO_MAX_REWARD_RISK",
            0.50,
        )
        self.max_stop_right_reward_risk = self._env_float(
            "ECML_RISK_VETO_MAX_STOP_RIGHT_REWARD_RISK",
            self.max_reward_risk,
        )
        self.max_stop_left_reward_risk = self._env_float(
            "ECML_RISK_VETO_MAX_STOP_LEFT_REWARD_RISK",
            self.max_reward_risk,
        )
        self.max_reward_risk_candidate_minus_baseline = self._env_float(
            "ECML_RISK_VETO_MAX_REWARD_RISK_CANDIDATE_MINUS_BASELINE",
            0.00,
        )
        self.min_reward_risk_baseline_minus_candidate = self._env_float(
            "ECML_RISK_VETO_MIN_REWARD_RISK_BASELINE_MINUS_CANDIDATE",
            float("-inf"),
        )
        self.min_value_delta = self._env_float(
            "ECML_RISK_VETO_MIN_VALUE_DELTA",
            float("-inf"),
        )
        self.min_candidate_value = self._env_float(
            "ECML_RISK_VETO_MIN_CANDIDATE_VALUE",
            float("-inf"),
        )
        self.max_candidate_distance_delta = self._env_float(
            "ECML_RISK_VETO_MAX_CANDIDATE_DISTANCE_DELTA",
            float("inf"),
        )
        self.max_action_distance_delta = {
            1: self._env_float(
                "ECML_RISK_VETO_MAX_LEFT_DISTANCE_DELTA",
                float("inf"),
            ),
            2: self._env_float(
                "ECML_RISK_VETO_MAX_FORWARD_DISTANCE_DELTA",
                float("inf"),
            ),
            3: self._env_float(
                "ECML_RISK_VETO_MAX_RIGHT_DISTANCE_DELTA",
                float("inf"),
            ),
        }
        self.require_stop_left_conflict = bool(
            int(os.environ.get("ECML_RISK_VETO_REQUIRE_STOP_LEFT_CONFLICT", "0") or "0")
        )
        self.stop_left_min_slack_for_unconflicted = self._env_float(
            "ECML_RISK_VETO_STOP_LEFT_MIN_SLACK_FOR_UNCONFLICTED",
            float("-inf"),
        )
        self.prefix_relax_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_PREFIX_RELAX_ENABLED", "0") or "0")
        )
        self.prefix_relax_max_future_head_on_risk = self._env_float(
            "ECML_RISK_VETO_PREFIX_RELAX_MAX_FUTURE_HEAD_ON_RISK",
            0.0,
        )
        self.prefix_relax_max_deadline_conflict_penalty = self._env_float(
            "ECML_RISK_VETO_PREFIX_RELAX_MAX_DEADLINE_CONFLICT_PENALTY",
            0.0,
        )
        allowed_prefix_relax_reasons = os.environ.get(
            "ECML_RISK_VETO_PREFIX_RELAX_ALLOWED_REJECT_REASONS",
            (
                "insufficient_risk_improvement,"
                "insufficient_reward_risk_improvement,"
                "reward_risk_too_high,"
                "reward_risk_regression"
            ),
        ).strip()
        self.prefix_relax_allowed_reject_reasons = {
            reason.strip()
            for reason in allowed_prefix_relax_reasons.split(",")
            if reason.strip()
        }
        self.prefix_relax_allow_all_reject_reasons = (
            allowed_prefix_relax_reasons == "*"
        )
        self.trace_path = os.environ.get("ECML_RISK_VETO_TRACE_PATH", "").strip()

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except Exception:
            return default

    @staticmethod
    def _env_transition_set(name: str) -> set[tuple[int, int]]:
        raw_value = os.environ.get(name, "").strip()
        if not raw_value:
            return set()
        transitions: set[tuple[int, int]] = set()
        for item in raw_value.split(","):
            if not item.strip():
                continue
            try:
                baseline, candidate = item.split(":", maxsplit=1)
                transitions.add((int(baseline), int(candidate)))
            except Exception:
                continue
        return transitions

    @staticmethod
    def _action_id(action: Any) -> int:
        if hasattr(action, "value"):
            return int(action.value)
        return int(action)

    @staticmethod
    def _action_name(action: int) -> str:
        try:
            return RailEnvActions(action).name
        except Exception:
            return str(action)

    def _max_reward_risk_for(
        self,
        baseline_action: int,
        candidate_action: int,
    ) -> float:
        if baseline_action == 4 and candidate_action == 3:
            return self.max_stop_right_reward_risk
        if baseline_action == 4 and candidate_action == 1:
            return self.max_stop_left_reward_risk
        return self.max_reward_risk

    def _action_risk_scores(
        self,
        policy: ActorCritic,
        observation: Any,
        baseline_action: int,
        candidate_action: int,
        prefix: str,
    ) -> dict[str, float]:
        with torch.no_grad():
            logits = policy.risk_logits(np.asarray(observation, dtype=np.float32))
            probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        baseline_risk = float(probs[baseline_action])
        candidate_risk = float(probs[candidate_action])
        return {
            f"{prefix}_baseline": baseline_risk,
            f"{prefix}_candidate": candidate_risk,
            f"{prefix}_candidate_minus_baseline": candidate_risk - baseline_risk,
            f"{prefix}_baseline_minus_candidate": baseline_risk - candidate_risk,
        }

    def _risk_scores(
        self,
        observation: Any,
        baseline_action: int,
        candidate_action: int,
    ) -> tuple[bool, dict[str, Any]]:
        if self.risk_policy is None:
            return False, {"reject_reason": "missing_risk_head"}
        try:
            scores = self._action_risk_scores(
                self.risk_policy,
                observation,
                baseline_action,
                candidate_action,
                "risk_head",
            )
        except Exception:
            return False, {"reject_reason": "risk_score_error"}

        candidate_risk = float(scores["risk_head_candidate"])
        candidate_minus_baseline = float(scores["risk_head_candidate_minus_baseline"])
        baseline_minus_candidate = float(scores["risk_head_baseline_minus_candidate"])
        accepted = True
        if candidate_risk > self.max_candidate_risk:
            scores["reject_reason"] = "candidate_risk_too_high"
            accepted = False
        elif candidate_minus_baseline > self.max_candidate_minus_baseline:
            scores["reject_reason"] = "candidate_risk_regression"
            accepted = False
        elif baseline_minus_candidate < self.min_baseline_minus_candidate:
            scores["reject_reason"] = "insufficient_risk_improvement"
            accepted = False
        if accepted and self.reward_risk_policy is not None:
            try:
                scores.update(
                    self._action_risk_scores(
                        self.reward_risk_policy,
                        observation,
                        baseline_action,
                        candidate_action,
                        "reward_risk_head",
                    )
                )
            except Exception:
                scores["reject_reason"] = "reward_risk_score_error"
                accepted = False
        if accepted and self.reward_risk_policy is not None:
            reward_candidate = float(scores["reward_risk_head_candidate"])
            reward_risk_limit = self._max_reward_risk_for(
                baseline_action,
                candidate_action,
            )
            scores["reward_risk_head_candidate_limit"] = float(reward_risk_limit)
            reward_candidate_minus_baseline = float(
                scores["reward_risk_head_candidate_minus_baseline"]
            )
            reward_baseline_minus_candidate = float(
                scores["reward_risk_head_baseline_minus_candidate"]
            )
            if reward_candidate > reward_risk_limit:
                scores["reject_reason"] = "reward_risk_too_high"
                accepted = False
            elif (
                reward_candidate_minus_baseline
                > self.max_reward_risk_candidate_minus_baseline
            ):
                scores["reject_reason"] = "reward_risk_regression"
                accepted = False
            elif (
                reward_baseline_minus_candidate
                < self.min_reward_risk_baseline_minus_candidate
            ):
                scores["reject_reason"] = "insufficient_reward_risk_improvement"
                accepted = False
        if accepted and self.value_policy is not None:
            try:
                with torch.no_grad():
                    values = self.value_policy.action_value_scores(
                        np.asarray(observation, dtype=np.float32)
                    ).squeeze(0).cpu().numpy()
                baseline_value = float(values[baseline_action])
                candidate_value = float(values[candidate_action])
                value_delta = candidate_value - baseline_value
                scores.update(
                    {
                        "value_head_baseline": baseline_value,
                        "value_head_candidate": candidate_value,
                        "value_head_candidate_minus_baseline": value_delta,
                    }
                )
            except Exception:
                scores["reject_reason"] = "value_score_error"
                accepted = False
        if accepted and self.value_policy is not None:
            candidate_value = float(scores["value_head_candidate"])
            value_delta = float(scores["value_head_candidate_minus_baseline"])
            if candidate_value < self.min_candidate_value:
                scores["reject_reason"] = "candidate_value_too_low"
                accepted = False
            elif value_delta < self.min_value_delta:
                scores["reject_reason"] = "insufficient_value_delta"
                accepted = False
        return accepted, scores

    def _baseline_prefixes(
        self,
        obs_builder: Any,
        baseline_actions: dict[int, int],
    ) -> dict[int, list[dict[str, Any]]]:
        lookahead = getattr(self.baseline_policy, "FUTURE_RERANK_LOOKAHEAD_CELLS", 45)
        prefixes: dict[int, list[dict[str, Any]]] = {}
        for handle, action in baseline_actions.items():
            try:
                prefixes[handle] = self.baseline_policy._route_prefix_for_action(
                    obs_builder,
                    handle,
                    int(action),
                    lookahead,
                )
            except Exception:
                prefixes[handle] = []
        return prefixes

    def _prefix_relax(
        self,
        obs_builder: Any | None,
        planned_prefixes: dict[int, list[dict[str, Any]]] | None,
        handle: int,
        candidate_action: int,
        scores: dict[str, Any],
    ) -> tuple[bool, dict[str, Any]]:
        if (
            not self.prefix_relax_enabled
            or obs_builder is None
            or planned_prefixes is None
        ):
            return False, scores
        original_reject_reason = str(scores.get("reject_reason", ""))
        scores["prefix_relax_original_reject_reason"] = original_reject_reason
        if (
            not self.prefix_relax_allow_all_reject_reasons
            and original_reject_reason
            not in self.prefix_relax_allowed_reject_reasons
        ):
            scores["prefix_relax_blocked_reject_reason"] = original_reject_reason
            return False, scores
        try:
            future_head_on_risk = float(
                self.baseline_policy._future_head_on_risk(
                    obs_builder,
                    handle,
                    candidate_action,
                    planned_prefixes,
                )
            )
            deadline_conflict_penalty = float(
                self.baseline_policy._deadline_conflict_penalty(
                    obs_builder,
                    handle,
                    candidate_action,
                    planned_prefixes,
                )
            )
        except Exception:
            scores["prefix_relax_error"] = 1.0
            return False, scores

        scores["prefix_relax_candidate_future_head_on_risk"] = future_head_on_risk
        scores["prefix_relax_candidate_deadline_conflict_penalty"] = (
            deadline_conflict_penalty
        )
        if future_head_on_risk > self.prefix_relax_max_future_head_on_risk:
            return False, scores
        if deadline_conflict_penalty > self.prefix_relax_max_deadline_conflict_penalty:
            return False, scores

        scores["selector_source"] = "prefix_relax"
        scores.pop("reject_reason", None)
        return True, scores

    def _candidate_distance_delta(
        self,
        obs_builder: Any | None,
        handle: int,
        candidate_action: int,
    ) -> float | None:
        if obs_builder is None:
            return None
        try:
            agent = obs_builder.env.agents[handle]
            position = agent.position
            if position is None:
                return None
            direction = (
                agent.direction
                if agent.direction is not None
                else agent.initial_direction
            )
            target, target_direction = obs_builder._action_target(
                handle,
                candidate_action,
            )
            if target is None or target_direction is None:
                return None
            distance_map = obs_builder._get_distance_map(handle)
            current_distance = float(distance_map[position[0], position[1], direction])
            candidate_distance = float(
                distance_map[target[0], target[1], target_direction]
            )
            if not np.isfinite(current_distance) or not np.isfinite(candidate_distance):
                return None
            return candidate_distance - current_distance
        except Exception:
            return None

    def _distance_delta_veto(
        self,
        obs_builder: Any | None,
        handle: int,
        candidate_action: int,
        scores: dict[str, Any],
    ) -> bool:
        action_limit = self.max_action_distance_delta.get(
            int(candidate_action),
            float("inf"),
        )
        max_distance_delta = min(self.max_candidate_distance_delta, action_limit)
        if not np.isfinite(max_distance_delta):
            return False
        distance_delta = self._candidate_distance_delta(
            obs_builder,
            handle,
            candidate_action,
        )
        if distance_delta is None:
            return False
        scores["candidate_distance_delta"] = float(distance_delta)
        scores["candidate_distance_delta_limit"] = float(max_distance_delta)
        if distance_delta > max_distance_delta:
            scores["reject_reason"] = "candidate_distance_delta_too_high"
            return True
        return False

    def _candidate_prefix_cell_intersections(
        self,
        obs_builder: Any | None,
        planned_prefixes: dict[int, list[dict[str, Any]]] | None,
        handle: int,
        candidate_action: int,
    ) -> float | None:
        if obs_builder is None or planned_prefixes is None:
            return None
        try:
            lookahead = getattr(
                self.baseline_policy,
                "FUTURE_RERANK_LOOKAHEAD_CELLS",
                45,
            )
            candidate_prefix = self.baseline_policy._route_prefix_for_action(
                obs_builder,
                handle,
                candidate_action,
                lookahead,
            )
        except Exception:
            return None
        candidate_positions = [
            node.get("position")
            for node in candidate_prefix
            if node.get("position") is not None
        ]
        if not candidate_positions:
            return 0.0
        other_positions = set()
        for other, other_prefix in planned_prefixes.items():
            if other == handle:
                continue
            for node in other_prefix:
                position = node.get("position")
                if position is not None:
                    other_positions.add(position)
        return float(sum(position in other_positions for position in candidate_positions))

    def _stop_left_conflict_veto(
        self,
        obs_builder: Any | None,
        planned_prefixes: dict[int, list[dict[str, Any]]] | None,
        handle: int,
        baseline_action: int,
        candidate_action: int,
        scores: dict[str, Any],
    ) -> bool:
        slack_threshold_enabled = np.isfinite(
            self.stop_left_min_slack_for_unconflicted
        )
        if not self.require_stop_left_conflict and not slack_threshold_enabled:
            return False
        if baseline_action != 4 or candidate_action != 1:
            return False
        prefix_intersections = self._candidate_prefix_cell_intersections(
            obs_builder,
            planned_prefixes,
            handle,
            candidate_action,
        )
        if prefix_intersections is None:
            return False
        try:
            deadline_penalty = float(
                self.baseline_policy._deadline_conflict_penalty(
                    obs_builder,
                    handle,
                    candidate_action,
                    planned_prefixes,
                )
            )
        except Exception:
            deadline_penalty = 0.0
        scores["candidate_prefix_cell_intersections"] = float(prefix_intersections)
        scores["candidate_deadline_conflict_penalty"] = float(deadline_penalty)
        if prefix_intersections <= 0.0 and deadline_penalty <= 0.0:
            if slack_threshold_enabled:
                try:
                    distance = obs_builder._current_distance_to_waypoint(handle)
                    slack = float(obs_builder._deadline_slack(handle, distance))
                except Exception:
                    slack = float("inf")
                scores["stop_left_current_slack"] = float(slack)
                scores["stop_left_min_slack_for_unconflicted"] = float(
                    self.stop_left_min_slack_for_unconflicted
                )
                if slack >= self.stop_left_min_slack_for_unconflicted:
                    return False
            scores["reject_reason"] = "stop_left_without_prefix_conflict"
            return True
        return False

    def _trace(
        self,
        *,
        handle: int,
        seed: int | None,
        step: int | None,
        baseline_action: int,
        candidate_action: int,
        accepted: bool,
        scores: dict[str, Any],
    ) -> None:
        if not self.trace_path:
            return
        context = runtime_context.get()
        row = {
            "seed": seed,
            "scene": context.scene,
            "env_time": step,
            "agent_id": int(handle),
            "baseline_action": int(baseline_action),
            "baseline_action_name": self._action_name(int(baseline_action)),
            "candidate_action": int(candidate_action),
            "candidate_action_name": self._action_name(int(candidate_action)),
            "accepted": bool(accepted),
            **scores,
        }
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle_out:
            handle_out.write(json.dumps(row, sort_keys=True) + "\n")

    def _candidate_action_lists(
        self,
        handles: List[int],
        observations: List[Any],
        baseline_action_ids: dict[int, int],
        candidate_actions: Dict[int, RailEnvActions],
    ) -> dict[int, list[tuple[int, dict[str, Any]]]]:
        result: dict[int, list[tuple[int, dict[str, Any]]]] = {}
        for handle in handles:
            result[handle] = []
            if handle in candidate_actions:
                result[handle].append(
                    (
                        self._action_id(candidate_actions[handle]),
                        {"candidate_source": "rerank"},
                    )
                )

        if self.top_n_candidate_actions <= 1:
            return result

        try:
            obs_array = np.asarray(observations, dtype=np.float32)
            with torch.no_grad():
                logits = (
                    self.candidate_policy.rl_policy.masked_logits(obs_array)
                    .cpu()
                    .numpy()
                )
        except Exception:
            return result

        for row_index, handle in enumerate(handles):
            existing = {action for action, _ in result.get(handle, [])}
            baseline_action = baseline_action_ids.get(handle)
            ranked_actions = [
                (int(action), float(logit))
                for action, logit in enumerate(logits[row_index])
                if np.isfinite(logit)
            ]
            ranked_actions.sort(key=lambda item: item[1], reverse=True)
            for rank, (action, logit) in enumerate(ranked_actions, start=1):
                if action == baseline_action or action in existing:
                    continue
                if self.top_n_allowed_transitions and (
                    baseline_action,
                    action,
                ) not in self.top_n_allowed_transitions:
                    continue
                result.setdefault(handle, []).append(
                    (
                        action,
                        {
                            "candidate_source": "raw_topn",
                            "candidate_policy_rank": int(rank),
                            "candidate_policy_logit": float(logit),
                        },
                    )
                )
                existing.add(action)
                if len(result[handle]) >= self.top_n_candidate_actions:
                    break
        return result

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        baseline_actions = self.baseline_policy.act_many(handles, observations, **kwargs)
        candidate_actions = self.candidate_policy.act_many(handles, observations, **kwargs)
        observations_by_handle = dict(zip(handles, observations))
        output = dict(baseline_actions)
        seed = None
        step = None
        obs_builder = None
        try:
            from submission import runtime_context

            context = runtime_context.get()
            seed = context.seed
            step = context.step
            obs_builder = context.obs_builder
        except Exception:
            pass
        baseline_action_ids = {
            handle: self._action_id(action)
            for handle, action in baseline_actions.items()
        }
        planned_prefixes = None
        if self.prefix_relax_enabled and obs_builder is not None:
            planned_prefixes = self._baseline_prefixes(obs_builder, baseline_action_ids)
        candidate_action_lists = self._candidate_action_lists(
            handles,
            observations,
            baseline_action_ids,
            candidate_actions,
        )

        for handle in handles:
            if handle not in baseline_actions:
                continue
            baseline_action = baseline_action_ids[handle]
            for candidate_action, candidate_metadata in candidate_action_lists.get(
                handle,
                [],
            ):
                if candidate_action == baseline_action:
                    continue
                accepted, scores = self._risk_scores(
                    observations_by_handle.get(handle),
                    baseline_action,
                    candidate_action,
                )
                scores.update(candidate_metadata)
                if not accepted:
                    accepted, scores = self._prefix_relax(
                        obs_builder,
                        planned_prefixes,
                        int(handle),
                        candidate_action,
                        scores,
                    )
                if accepted and self._distance_delta_veto(
                    obs_builder,
                    int(handle),
                    candidate_action,
                    scores,
                ):
                    accepted = False
                if accepted and self._stop_left_conflict_veto(
                    obs_builder,
                    planned_prefixes,
                    int(handle),
                    baseline_action,
                    candidate_action,
                    scores,
                ):
                    accepted = False
                self._trace(
                    handle=handle,
                    seed=seed,
                    step=step,
                    baseline_action=baseline_action,
                    candidate_action=candidate_action,
                    accepted=accepted,
                    scores=scores,
                )
                if accepted:
                    output[handle] = RailEnvActions(candidate_action)
                    break
        return output


MyPolicy = RiskVetoPolicy
