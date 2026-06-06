from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
from submission.rerank_policy import RerankPolicy
from submission.reservation_policy import ReservationPolicy

try:
    from tools.analyze_policy_action_diffs import diff_row
    from tools.mine_diff_prefix_dataset import (
        add_delta_features,
        aggregate_event_features,
    )
except Exception:  # pragma: no cover - submission fallback for stripped packages.
    diff_row = None
    add_delta_features = None
    aggregate_event_features = None


SUBMISSION_DIR = Path(__file__).resolve().parent
DEFAULT_SEQUENCE_MODEL_PATH = str(
    SUBMISSION_DIR / "models" / "ecml_success_only_sequence_fasttrain.pt"
)
DEFAULT_CANDIDATE_CHECKPOINT_PATHS = (
    str(SUBMISSION_DIR / "models" / "ecml_ppo_trajectory_successdiv_seed610_u4.pt"),
    str(SUBMISSION_DIR / "models" / "ecml_ppo_trajectory_terminal_stronger_u3.pt"),
    str(SUBMISSION_DIR / "models" / "ecml_ppo_trajectory_targeted_seed901_u4.pt"),
)


class SequenceValueRiskMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int):
        super().__init__()
        if hidden_size <= 0:
            self.shared = nn.Identity()
            shared_dim = input_dim
        else:
            self.shared = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            shared_dim = hidden_size
        self.value_head = nn.Linear(shared_dim, 1)
        self.bad_head = nn.Linear(shared_dim, 1)
        self.success_head = nn.Linear(shared_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.shared(features)
        return (
            self.value_head(hidden).squeeze(-1),
            self.bad_head(hidden).squeeze(-1),
            self.success_head(hidden).squeeze(-1),
        )


class SequenceEnsembleScorer:
    def __init__(
        self,
        checkpoint_path: str,
        min_utility: float,
        max_bad_probability: float,
        min_success_probability: float,
        value_std_coef: float,
        bad_std_coef: float,
        success_std_coef: float,
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.feature_columns = list(checkpoint["feature_columns"])
        self.feature_mean = checkpoint["feature_mean"].detach().cpu().numpy()
        self.feature_std = checkpoint["feature_std"].detach().cpu().numpy()
        self.min_utility = float(min_utility)
        self.max_bad_probability = float(max_bad_probability)
        self.min_success_probability = float(min_success_probability)
        self.value_std_coef = float(value_std_coef)
        self.bad_std_coef = float(bad_std_coef)
        self.success_std_coef = float(success_std_coef)
        self.models = []
        for state_dict in checkpoint["model_state_dicts"]:
            model = SequenceValueRiskMLP(
                input_dim=int(checkpoint["input_dim"]),
                hidden_size=int(checkpoint["hidden_size"]),
            )
            model.load_state_dict(state_dict)
            model.eval()
            self.models.append(model)

    @staticmethod
    def _safe_float(value: Any) -> float:
        try:
            result = float(value)
        except Exception:
            return 0.0
        return result if np.isfinite(result) else 0.0

    def _feature_vector(self, row: dict[str, Any]) -> np.ndarray:
        values = [
            self._safe_float(row.get(column, 0.0))
            for column in self.feature_columns
        ]
        raw = np.asarray(values, dtype=np.float32)
        return (raw - self.feature_mean) / self.feature_std

    def score(self, row: dict[str, Any]) -> tuple[bool, dict[str, float]]:
        features = torch.as_tensor(
            self._feature_vector(row)[None, :],
            dtype=torch.float32,
        )
        values = []
        bad_probs = []
        success_probs = []
        with torch.no_grad():
            for model in self.models:
                value_pred, bad_logits, success_logits = model(features)
                values.append(float(value_pred.item()))
                bad_probs.append(float(torch.sigmoid(bad_logits).item()))
                success_probs.append(float(torch.sigmoid(success_logits).item()))

        value_mean = float(np.mean(values))
        value_std = float(np.std(values))
        bad_mean = float(np.mean(bad_probs))
        bad_std = float(np.std(bad_probs))
        success_mean = float(np.mean(success_probs))
        success_std = float(np.std(success_probs))
        value_lcb = value_mean - self.value_std_coef * value_std
        bad_ucb = bad_mean + self.bad_std_coef * bad_std
        success_lcb = success_mean - self.success_std_coef * success_std
        accepted = (
            bad_ucb <= self.max_bad_probability
            and (
                value_lcb >= self.min_utility
                or success_lcb >= self.min_success_probability
            )
        )
        return accepted, {
            "value_lcb": value_lcb,
            "bad_ucb": bad_ucb,
            "success_lcb": success_lcb,
        }


class SequenceSuccessPolicy(RerankPolicy):
    """Experimental online wrapper for the exported sequence Success gate.

    The guarded rerank policy remains the baseline. A PPO candidate checkpoint
    proposes deviations, and the exported sequence ensemble can accept a
    deviation only if the cumulative accepted-diff prefix scores as low-risk
    Success rescue. This policy is intentionally not the default submission
    policy yet.
    """

    def __init__(self, checkpoint_path: str | None = None):
        super().__init__(checkpoint_path="./submission/checkpoint.pt")
        sequence_model_path = (
            checkpoint_path
            or os.environ.get("ECML_SEQUENCE_SUCCESS_MODEL")
            or DEFAULT_SEQUENCE_MODEL_PATH
        )
        candidate_checkpoint_path = (
            os.environ.get("ECML_SEQUENCE_SUCCESS_CANDIDATE_CHECKPOINT")
            or ",".join(DEFAULT_CANDIDATE_CHECKPOINT_PATHS)
        )
        candidate_checkpoint_paths = self._candidate_checkpoint_paths(
            candidate_checkpoint_path
        )
        existing_candidate_paths = [
            path for path in candidate_checkpoint_paths if Path(path).exists()
        ]
        self.candidate_policies = [
            RerankPolicy(checkpoint_path=path)
            for path in existing_candidate_paths
        ]
        self.candidate_policy_paths = {
            id(policy): path
            for policy, path in zip(self.candidate_policies, existing_candidate_paths)
        }
        self.sequence_scorer = (
            self._load_scorer(sequence_model_path)
            if Path(sequence_model_path).exists()
            else None
        )
        self.max_accepted_events = self._env_int(
            "ECML_SEQUENCE_MAX_ACCEPTED_EVENTS",
            1,
        )
        self.left_max_slack = self._env_float(
            "ECML_SEQUENCE_LEFT_MAX_SLACK",
            self.FUTURE_RERANK_LEFT_MAX_SLACK,
        )
        self.trace_path = os.environ.get("ECML_SEQUENCE_TRACE_PATH", "").strip()
        self.trace_all = bool(self._env_int("ECML_SEQUENCE_TRACE_ALL", 0))
        self.first_diff_only = bool(
            self._env_int("ECML_SEQUENCE_FIRST_DIFF_ONLY", 1)
        )
        self.rejected_transitions = self._env_transition_set(
            "ECML_SEQUENCE_REJECT_TRANSITIONS",
            default="MOVE_RIGHT->MOVE_FORWARD",
        )
        self.max_head_on_edge_conflicts = self._env_int(
            "ECML_SEQUENCE_MAX_HEAD_ON_EDGE_CONFLICTS",
            8,
        )
        self.same_edge_value_guard_min_conflicts = self._env_int(
            "ECML_SEQUENCE_SAME_EDGE_VALUE_GUARD_MIN_CONFLICTS",
            8,
        )
        self.same_edge_value_guard_min_value = self._env_float(
            "ECML_SEQUENCE_SAME_EDGE_VALUE_GUARD_MIN_VALUE_LCB",
            -0.5,
        )
        self._accepted_event_details: list[dict[str, Any]] = []
        self._seen_candidate_diff_policy_ids: set[int] = set()
        self._last_step: int | None = None

    @staticmethod
    def _candidate_checkpoint_paths(default_path: str) -> list[str]:
        value = os.environ.get("ECML_SEQUENCE_SUCCESS_CANDIDATE_CHECKPOINTS")
        if value is None:
            value = default_path
        return [
            item.strip()
            for item in value.split(",")
            if item.strip()
        ]

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except Exception:
            return default

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        try:
            return int(os.environ.get(name, default))
        except Exception:
            return default

    @classmethod
    def _env_transition_set(
        cls,
        name: str,
        default: str = "",
    ) -> set[tuple[int, int]]:
        value = os.environ.get(name, default)
        transitions: set[tuple[int, int]] = set()
        for item in value.split(","):
            token = item.strip()
            if not token or "->" not in token:
                continue
            left, right = (part.strip() for part in token.split("->", maxsplit=1))
            baseline_action = cls._parse_action_token(left)
            candidate_action = cls._parse_action_token(right)
            if baseline_action is None or candidate_action is None:
                continue
            transitions.add((baseline_action, candidate_action))
        return transitions

    @staticmethod
    def _parse_action_token(token: str) -> int | None:
        if not token:
            return None
        try:
            return int(token)
        except ValueError:
            pass
        name = token.upper()
        if not name.startswith("MOVE_") and name in {"LEFT", "FORWARD", "RIGHT"}:
            name = f"MOVE_{name}"
        try:
            return int(RailEnvActions[name].value)
        except Exception:
            return None

    def _load_scorer(self, sequence_model_path: str) -> SequenceEnsembleScorer:
        return SequenceEnsembleScorer(
            checkpoint_path=sequence_model_path,
            min_utility=self._env_float("ECML_SEQUENCE_MIN_UTILITY", 999.0),
            max_bad_probability=self._env_float(
                "ECML_SEQUENCE_MAX_BAD_PROBABILITY",
                0.0025,
            ),
            min_success_probability=self._env_float(
                "ECML_SEQUENCE_MIN_SUCCESS_PROBABILITY",
                0.02,
            ),
            value_std_coef=self._env_float("ECML_SEQUENCE_VALUE_STD_COEF", 2.0),
            bad_std_coef=self._env_float("ECML_SEQUENCE_BAD_STD_COEF", 2.0),
            success_std_coef=self._env_float("ECML_SEQUENCE_SUCCESS_STD_COEF", 0.0),
        )

    def _reset_episode_state_if_needed(self, step: int | None) -> None:
        if step is None:
            return
        if self._last_step is None or step < self._last_step:
            self._accepted_event_details = []
            self._seen_candidate_diff_policy_ids = set()
        elif step == 0 and self._last_step != 0:
            self._accepted_event_details = []
            self._seen_candidate_diff_policy_ids = set()
        self._last_step = step

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        baseline_actions = super().act_many(handles, observations, **kwargs)
        if not self.candidate_policies or self.sequence_scorer is None:
            return baseline_actions
        if diff_row is None or add_delta_features is None or aggregate_event_features is None:
            return baseline_actions

        candidate_batches = [
            (
                candidate_policy,
                candidate_policy.act_many(handles, observations, **kwargs),
                self._raw_policy_actions(candidate_policy, handles, observations),
            )
            for candidate_policy in self.candidate_policies
        ]
        context = runtime_context.get()
        env = context.env
        obs_builder = context.obs_builder
        if (
            env is None
            or obs_builder is None
            or env.get_num_agents() < 6
            or max(env.height, env.width) < 100
        ):
            return baseline_actions

        self._reset_episode_state_if_needed(getattr(env, "_elapsed_steps", None))
        adjusted = dict(baseline_actions)
        observations_by_handle = dict(zip(handles, observations))
        baseline_action_ids = {
            handle: self._action_id(action)
            for handle, action in baseline_actions.items()
        }
        candidate_action_ids = {
            candidate_policy: {
                handle: self._action_id(action)
                for handle, action in candidate_actions.items()
            }
            for candidate_policy, candidate_actions, _ in candidate_batches
        }
        baseline_raw_actions = self._raw_policy_actions(self, handles, observations)

        reserved_targets: set[tuple[int, int]] = set()
        for handle in sorted(adjusted, key=lambda h: self._priority_key(obs_builder, h)):
            baseline_action = baseline_action_ids.get(handle)
            action_id = baseline_action
            if baseline_action is not None:
                for candidate_policy, _, candidate_raw_actions in candidate_batches:
                    candidate_actions = candidate_action_ids[candidate_policy]
                    candidate_action = candidate_actions.get(handle)
                    if (
                        candidate_action is not None
                        and baseline_action != candidate_action
                    ):
                        policy_id = id(candidate_policy)
                        if (
                            self.first_diff_only
                            and policy_id in self._seen_candidate_diff_policy_ids
                        ):
                            continue
                        self._seen_candidate_diff_policy_ids.add(policy_id)
                        if self._accept_candidate_action(
                            env=env,
                            obs_builder=obs_builder,
                            handle=handle,
                            observation=observations_by_handle.get(handle),
                            baseline_action=baseline_action,
                            candidate_action=candidate_action,
                            baseline_raw_actions=baseline_raw_actions,
                            candidate_raw_actions=candidate_raw_actions,
                            baseline_actions=baseline_action_ids,
                            candidate_actions=candidate_actions,
                            candidate_policy=candidate_policy,
                            reserved_targets=reserved_targets,
                        ):
                            adjusted[handle] = RailEnvActions(candidate_action)
                            action_id = candidate_action
                            break

            if action_id is not None:
                self._reserve_action_target(
                    reserved_targets,
                    obs_builder,
                    handle,
                    action_id,
                )

        return adjusted

    def _accept_candidate_action(
        self,
        env: Any,
        obs_builder: Any,
        handle: int,
        observation: Any,
        baseline_action: int,
        candidate_action: int,
        baseline_raw_actions: dict[int, int],
        candidate_raw_actions: dict[int, int],
        baseline_actions: dict[int, int],
        candidate_actions: dict[int, int],
        candidate_policy: RerankPolicy,
        reserved_targets: set[tuple[int, int]],
    ) -> bool:
        if (
            self.max_accepted_events >= 0
            and len(self._accepted_event_details) >= self.max_accepted_events
        ):
            return False
        if candidate_action not in (
            ReservationPolicy.MOVE_LEFT,
            ReservationPolicy.MOVE_FORWARD,
            ReservationPolicy.MOVE_RIGHT,
        ):
            return False
        if (baseline_action, candidate_action) in self.rejected_transitions:
            return False
        if (
            candidate_action == ReservationPolicy.MOVE_LEFT
            and self._current_slack(obs_builder, handle) > self.left_max_slack
        ):
            return False
        if not self._mask_allows(observation, candidate_action):
            return False
        target, _ = obs_builder._action_target(handle, candidate_action)
        if (
            target is None
            or target in reserved_targets
            or obs_builder._occupied_by_other(target, handle)
        ):
            return False

        try:
            detail = diff_row(
                self._feature_args(),
                0,
                env,
                obs_builder,
                handle,
                observation,
                self,
                candidate_policy,
                baseline_raw_actions,
                candidate_raw_actions,
                baseline_actions,
                candidate_actions,
            )
            detail = add_delta_features(detail)
            detail["prefix_index"] = len(self._accepted_event_details) + 1
            aggregate = aggregate_event_features(
                [*self._accepted_event_details, detail]
            )
            if self._too_many_head_on_edge_conflicts(detail):
                if self.trace_all:
                    self._trace_sequence_decision(
                        env=env,
                        handle=handle,
                        baseline_action=baseline_action,
                        candidate_action=candidate_action,
                        candidate_policy=candidate_policy,
                        accepted=False,
                        scores={},
                        detail=detail,
                        aggregate=aggregate,
                    )
                return False
            accepted, scores = self.sequence_scorer.score(aggregate)
            if accepted and self._low_value_same_edge_candidate(detail, scores):
                accepted = False
        except Exception:
            return False

        if accepted or self.trace_all:
            self._trace_sequence_decision(
                env=env,
                handle=handle,
                baseline_action=baseline_action,
                candidate_action=candidate_action,
                candidate_policy=candidate_policy,
                accepted=accepted,
                scores=scores,
                detail=detail,
                aggregate=aggregate,
            )
        if accepted:
            self._accepted_event_details.append(detail)
        return bool(accepted)

    def _too_many_head_on_edge_conflicts(self, detail: dict[str, Any]) -> bool:
        if self.max_head_on_edge_conflicts < 0:
            return False
        try:
            conflicts = float(detail.get("candidate_prefix_head_on_edge_conflicts", 0.0))
        except Exception:
            return False
        return conflicts > self.max_head_on_edge_conflicts

    def _low_value_same_edge_candidate(
        self,
        detail: dict[str, Any],
        scores: dict[str, float],
    ) -> bool:
        if self.same_edge_value_guard_min_conflicts < 0:
            return False
        try:
            conflicts = float(
                detail.get("candidate_prefix_same_edge_conflicts", 0.0)
            )
            value_lcb = float(scores.get("value_lcb", 0.0))
        except Exception:
            return False
        return (
            conflicts >= self.same_edge_value_guard_min_conflicts
            and value_lcb < self.same_edge_value_guard_min_value
        )

    def _trace_sequence_decision(
        self,
        env: Any,
        handle: int,
        baseline_action: int,
        candidate_action: int,
        candidate_policy: RerankPolicy,
        accepted: bool,
        scores: dict[str, float],
        detail: dict[str, Any],
        aggregate: dict[str, Any],
    ) -> None:
        if not self.trace_path:
            return
        row = {
            "accepted": bool(accepted),
            "env_time": int(getattr(env, "_elapsed_steps", -1)),
            "agent_id": int(handle),
            "candidate_checkpoint": self.candidate_policy_paths.get(
                id(candidate_policy),
                "",
            ),
            "baseline_action": int(baseline_action),
            "baseline_action_name": self._action_name(baseline_action),
            "candidate_action": int(candidate_action),
            "candidate_action_name": self._action_name(candidate_action),
            "value_lcb": float(scores.get("value_lcb", 0.0)),
            "bad_ucb": float(scores.get("bad_ucb", 0.0)),
            "success_lcb": float(scores.get("success_lcb", 0.0)),
            "accepted_event_count_before": len(self._accepted_event_details),
        }
        for key in (
            "slack",
            "distance",
            "candidate_distance_delta",
            "candidate_prefix_cell_intersections",
            "candidate_prefix_head_on_edge_conflicts",
            "candidate_prefix_same_edge_conflicts",
            "candidate_prefix_min_pair_deadline_slack",
            "candidate_future_head_on_risk",
            "candidate_deadline_conflict_penalty",
            "candidate_residual_head_on_min_pair_deadline_slack",
            "candidate_raw_candidate_minus_baseline_logit",
            "candidate_raw_top_logit_margin",
            "obs_route_occupancy_count",
            "obs_route_intersection_count",
            "obs_route_intersection_head_on",
        ):
            if key in detail:
                row[key] = self._json_safe(detail[key])
        for key in (
            "event_count",
            "event_unique_agents",
            "event_time_first",
            "event_time_last",
            "event_candidate_action_MOVE_FORWARD",
            "event_candidate_action_MOVE_LEFT",
            "event_transition_MOVE_RIGHT__MOVE_FORWARD",
            "event_transition_MOVE_LEFT__MOVE_FORWARD",
            "event_transition_MOVE_FORWARD__MOVE_LEFT",
            "event_candidate_prefix_cell_intersections_max",
            "event_candidate_prefix_head_on_edge_conflicts_max",
            "event_candidate_prefix_same_edge_conflicts_max",
            "event_candidate_prefix_min_pair_deadline_slack_min",
            "event_candidate_future_head_on_risk_max",
            "event_candidate_deadline_conflict_penalty_max",
            "event_candidate_residual_head_on_min_pair_deadline_slack_min",
        ):
            if key in aggregate:
                row[key] = self._json_safe(aggregate[key])
        try:
            path = Path(self.trace_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as handle_obj:
                json.dump(row, handle_obj, sort_keys=True)
                handle_obj.write("\n")
        except Exception:
            return

    @staticmethod
    def _json_safe(value: Any) -> Any:
        try:
            result = float(value)
        except Exception:
            return str(value)
        if not np.isfinite(result):
            return str(value)
        return result

    @staticmethod
    def _action_name(action: int) -> str:
        try:
            return RailEnvActions(int(action)).name
        except Exception:
            return str(action)

    @staticmethod
    def _feature_args() -> Any:
        class Args:
            pass

        return Args()

    @staticmethod
    def _mask_allows(observation: Any, action: int) -> bool:
        if observation is None:
            return False
        values = np.asarray(observation, dtype=np.float32)
        if values.shape[0] < 5 or action >= 5:
            return False
        return bool(values[-5 + action] >= 0.5)

    @staticmethod
    def _current_slack(obs_builder: Any, handle: int) -> float:
        try:
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            return float(obs_builder._deadline_slack(handle, distance))
        except Exception:
            return float("inf")

    @staticmethod
    def _raw_policy_actions(
        policy: Any,
        handles: list[int],
        observations: list[Any],
    ) -> dict[int, int]:
        raw_policy = getattr(policy, "rl_policy", None)
        if raw_policy is None:
            return {}
        try:
            return {
                handle: RerankPolicy._action_id(action)
                for handle, action in raw_policy.act_many(handles, observations).items()
            }
        except Exception:
            return {}


MyPolicy = SequenceSuccessPolicy
