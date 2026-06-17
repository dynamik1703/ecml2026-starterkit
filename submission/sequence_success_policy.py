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
from submission.my_policy import ActorCritic
from submission.rerank_policy import RerankPolicy
from submission.reservation_policy import ReservationPolicy

try:
    from tools.analyze_policy_action_diffs import diff_row
    from tools.mine_diff_prefix_dataset import (
        add_delta_features,
        aggregate_event_features,
        candidate_source_features,
    )
except Exception:  # pragma: no cover - submission fallback for stripped packages.
    diff_row = None
    add_delta_features = None
    aggregate_event_features = None
    candidate_source_features = None


SUBMISSION_DIR = Path(__file__).resolve().parent
DEFAULT_BASE_CHECKPOINT_PATH = str(
    SUBMISSION_DIR / "models" / "ecml_aux_bc_conflict_neg_currentinit_ppo_v4b.pt"
)
DEFAULT_SEQUENCE_MODEL_PATH = str(
    SUBMISSION_DIR / "models" / "ecml_success_only_sequence_with_rescue.pt"
)
DEFAULT_LISTWISE_MODEL_PATH = str(
    SUBMISSION_DIR
    / "models"
    / "ecml_multicandidate_listwise_ranker_online_prefix_v6_scene3_multiscene.pt"
)
DEFAULT_AUX_LISTWISE_MODEL_PATH = str(
    SUBMISSION_DIR
    / "models"
    / "ecml_multicandidate_listwise_ranker_online_prefix_v9_enriched.pt"
)
DEFAULT_CANDIDATE_CHECKPOINT_PATHS = (
    str(SUBMISSION_DIR / "models" / "ecml_action_conflict_penalty_ppo_seed3700_u12.pt"),
    str(
        SUBMISSION_DIR
        / "models"
        / "ecml_action_conflict_successdiv_penalty_ppo_seed3800_u10.pt"
    ),
    str(SUBMISSION_DIR / "models" / "ecml_aux_bc_counterfactual_v7_probe.pt"),
    str(SUBMISSION_DIR / "models" / "ecml_aux_bc_counterfactual_v7b_3435neg_u1.pt"),
)


class SequenceValueRiskMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_size: int,
        include_success_regression_head: bool = False,
    ):
        super().__init__()
        self.include_success_regression_head = include_success_regression_head
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
        if include_success_regression_head:
            self.success_regression_head = nn.Linear(shared_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.shared(features)
        value_pred = self.value_head(hidden).squeeze(-1)
        if self.include_success_regression_head:
            success_regression_logits = self.success_regression_head(hidden).squeeze(-1)
        else:
            success_regression_logits = torch.full_like(value_pred, 20.0)
        return (
            value_pred,
            self.bad_head(hidden).squeeze(-1),
            self.success_head(hidden).squeeze(-1),
            success_regression_logits,
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
        max_success_regression_probability: float,
        success_regression_std_coef: float,
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.feature_columns = list(checkpoint["feature_columns"])
        self.feature_mean = checkpoint["feature_mean"].detach().cpu().numpy()
        self.feature_std = checkpoint["feature_std"].detach().cpu().numpy()
        self.include_success_regression_head = bool(
            checkpoint.get("include_success_regression_head", False)
        )
        self.min_utility = float(min_utility)
        self.max_bad_probability = float(max_bad_probability)
        self.min_success_probability = float(min_success_probability)
        self.value_std_coef = float(value_std_coef)
        self.bad_std_coef = float(bad_std_coef)
        self.success_std_coef = float(success_std_coef)
        self.max_success_regression_probability = float(
            max_success_regression_probability
        )
        self.success_regression_std_coef = float(success_regression_std_coef)
        self.models = []
        for state_dict in checkpoint["model_state_dicts"]:
            model = SequenceValueRiskMLP(
                input_dim=int(checkpoint["input_dim"]),
                hidden_size=int(checkpoint["hidden_size"]),
                include_success_regression_head=self.include_success_regression_head,
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
        success_regression_probs = []
        with torch.no_grad():
            for model in self.models:
                (
                    value_pred,
                    bad_logits,
                    success_logits,
                    success_regression_logits,
                ) = model(features)
                values.append(float(value_pred.item()))
                bad_probs.append(float(torch.sigmoid(bad_logits).item()))
                success_probs.append(float(torch.sigmoid(success_logits).item()))
                success_regression_probs.append(
                    float(torch.sigmoid(success_regression_logits).item())
                )

        value_mean = float(np.mean(values))
        value_std = float(np.std(values))
        bad_mean = float(np.mean(bad_probs))
        bad_std = float(np.std(bad_probs))
        success_mean = float(np.mean(success_probs))
        success_std = float(np.std(success_probs))
        success_regression_mean = float(np.mean(success_regression_probs))
        success_regression_std = float(np.std(success_regression_probs))
        value_lcb = value_mean - self.value_std_coef * value_std
        bad_ucb = bad_mean + self.bad_std_coef * bad_std
        success_lcb = success_mean - self.success_std_coef * success_std
        success_regression_ucb = (
            success_regression_mean
            + self.success_regression_std_coef * success_regression_std
        )
        success_regression_ok = (
            not np.isfinite(self.max_success_regression_probability)
            or success_regression_ucb <= self.max_success_regression_probability
        )
        accepted = (
            bad_ucb <= self.max_bad_probability
            and success_regression_ok
            and (
                value_lcb >= self.min_utility
                or success_lcb >= self.min_success_probability
            )
        )
        return accepted, {
            "value_lcb": value_lcb,
            "bad_ucb": bad_ucb,
            "success_lcb": success_lcb,
            "success_regression_ucb": success_regression_ucb,
        }


class SequenceListwiseNet(nn.Module):
    def __init__(
        self,
        static_dim: int,
        event_dim: int,
        hidden_size: int,
        event_hidden_size: int,
        dropout: float,
    ):
        super().__init__()
        self.static_encoder = (
            nn.Sequential(nn.Linear(static_dim, hidden_size), nn.ReLU())
            if static_dim > 0
            else None
        )
        self.event_encoder = (
            nn.Sequential(nn.Linear(event_dim, event_hidden_size), nn.ReLU())
            if event_dim > 0
            else None
        )
        self.event_gru = (
            nn.GRU(event_hidden_size, hidden_size, batch_first=True)
            if event_dim > 0
            else None
        )
        combined_dim = hidden_size * (1 + int(static_dim > 0))
        self.head = nn.Sequential(
            nn.Linear(combined_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        static_features: torch.Tensor,
        event_features: torch.Tensor,
        event_mask: torch.Tensor,
    ) -> torch.Tensor:
        parts = []
        if self.event_encoder is not None and self.event_gru is not None:
            encoded_events = self.event_encoder(event_features)
            output, _hidden = self.event_gru(encoded_events)
            lengths = event_mask.sum(dim=1).long().clamp_min(1)
            indices = (lengths - 1).view(-1, 1, 1).expand(-1, 1, output.shape[-1])
            event_summary = output.gather(dim=1, index=indices).squeeze(1)
            event_summary = torch.where(
                event_mask.sum(dim=1, keepdim=True) > 0,
                event_summary,
                torch.zeros_like(event_summary),
            )
            parts.append(event_summary)
        if self.static_encoder is not None:
            parts.append(self.static_encoder(static_features))
        return self.head(torch.cat(parts, dim=1)).squeeze(-1)


class ListwiseSequenceScorer:
    def __init__(
        self,
        checkpoint_path: str,
        margin_threshold: float,
        score_std_coef: float,
        baseline_std_coef: float,
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.static_feature_columns = list(checkpoint["static_feature_columns"])
        self.event_feature_columns = list(checkpoint["event_feature_columns"])
        self.static_feature_mean = checkpoint["static_feature_mean"].detach().cpu().numpy()
        self.static_feature_std = checkpoint["static_feature_std"].detach().cpu().numpy()
        self.event_feature_mean = checkpoint["event_feature_mean"].detach().cpu().numpy()
        self.event_feature_std = checkpoint["event_feature_std"].detach().cpu().numpy()
        self.max_events = int(checkpoint["max_events"])
        self.margin_threshold = float(margin_threshold)
        self.score_std_coef = float(score_std_coef)
        self.baseline_std_coef = float(baseline_std_coef)
        self.models = []
        for state_dict in checkpoint["model_state_dicts"]:
            model = SequenceListwiseNet(
                static_dim=int(checkpoint["static_dim"]),
                event_dim=int(checkpoint["event_dim"]),
                hidden_size=int(checkpoint["hidden_size"]),
                event_hidden_size=int(checkpoint["event_hidden_size"]),
                dropout=float(checkpoint.get("dropout", 0.0)),
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

    def _static_features(self, row: dict[str, Any]) -> np.ndarray:
        raw = np.asarray(
            [
                self._safe_float(row.get(column, 0.0))
                for column in self.static_feature_columns
            ],
            dtype=np.float32,
        )
        return (raw - self.static_feature_mean) / self.static_feature_std

    def _event_feature_value(self, event: dict[str, Any], feature: str) -> float:
        if feature.startswith("num:"):
            return self._safe_float(event.get(feature[4:]))
        if feature.startswith("cat:"):
            _prefix, field, value = feature.split(":", 2)
            return float(str(event.get(field)) == value)
        if feature.startswith("transition:"):
            value = feature[len("transition:") :]
            transition = (
                f"{event.get('baseline_action_name')}"
                f"->{event.get('candidate_action_name')}"
            )
            return float(transition == value)
        return 0.0

    def _event_features(self, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        events = row.get("event_details") or []
        if not isinstance(events, list):
            events = []
        matrix = np.zeros(
            (self.max_events, len(self.event_feature_columns)),
            dtype=np.float32,
        )
        mask = np.zeros((self.max_events,), dtype=np.float32)
        for event_index, event in enumerate(events[: self.max_events]):
            if not isinstance(event, dict):
                continue
            mask[event_index] = 1.0
            for feature_index, feature in enumerate(self.event_feature_columns):
                matrix[event_index, feature_index] = self._event_feature_value(
                    event,
                    feature,
                )
        if len(self.event_feature_columns):
            matrix = (matrix - self.event_feature_mean) / self.event_feature_std
        return matrix, mask

    def _score_raw(self, row: dict[str, Any]) -> tuple[float, float]:
        static_features = torch.as_tensor(
            self._static_features(row)[None, :],
            dtype=torch.float32,
        )
        event_features_np, event_mask_np = self._event_features(row)
        event_features = torch.as_tensor(
            event_features_np[None, :, :],
            dtype=torch.float32,
        )
        event_mask = torch.as_tensor(event_mask_np[None, :], dtype=torch.float32)
        scores = []
        with torch.no_grad():
            for model in self.models:
                scores.append(
                    float(model(static_features, event_features, event_mask).item())
                )
        return float(np.mean(scores)), float(np.std(scores))

    def score(self, row: dict[str, Any]) -> tuple[bool, dict[str, float]]:
        baseline_row = {
            "prefix_len": 0,
            "forced_applied": 0,
            "reward_delta": 0.0,
            "success_delta": 0.0,
            "failed_agents_delta": 0.0,
            "event_details": [],
            "event_count": 0,
            "event_unique_agents": 0,
            "is_baseline_candidate": 1.0,
            "candidate_source_baseline_noop": 1.0,
        }
        candidate_mean, candidate_std = self._score_raw(row)
        baseline_mean, baseline_std = self._score_raw(baseline_row)
        candidate_lcb = candidate_mean - self.score_std_coef * candidate_std
        baseline_ucb = baseline_mean + self.baseline_std_coef * baseline_std
        margin = candidate_lcb - baseline_ucb
        accepted = margin >= self.margin_threshold
        return accepted, {
            "value_lcb": margin,
            "listwise_margin": margin,
            "listwise_candidate_lcb": candidate_lcb,
            "listwise_candidate_mean": candidate_mean,
            "listwise_candidate_std": candidate_std,
            "listwise_baseline_ucb": baseline_ucb,
            "listwise_baseline_mean": baseline_mean,
            "listwise_baseline_std": baseline_std,
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
        base_checkpoint_path = (
            os.environ.get("ECML_SEQUENCE_BASE_CHECKPOINT")
            or checkpoint_path
            or DEFAULT_BASE_CHECKPOINT_PATH
        )
        super().__init__(checkpoint_path=base_checkpoint_path)
        sequence_model_path = (
            os.environ.get("ECML_SEQUENCE_SUCCESS_MODEL")
            or DEFAULT_SEQUENCE_MODEL_PATH
        )
        listwise_model_path = (
            os.environ.get("ECML_SEQUENCE_LISTWISE_MODEL")
            or DEFAULT_LISTWISE_MODEL_PATH
        )
        aux_listwise_model_path = os.environ.get(
            "ECML_SEQUENCE_AUX_LISTWISE_MODEL",
            DEFAULT_AUX_LISTWISE_MODEL_PATH,
        ).strip()
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
        self.listwise_scorer = (
            self._load_listwise_scorer(listwise_model_path)
            if Path(listwise_model_path).exists()
            else None
        )
        self.aux_listwise_scorer = (
            self._load_listwise_scorer(
                aux_listwise_model_path,
                env_prefix="ECML_SEQUENCE_AUX_LISTWISE",
                default_margin_threshold=0.75,
            )
            if aux_listwise_model_path and Path(aux_listwise_model_path).exists()
            else None
        )
        self.selector_mode = os.environ.get(
            "ECML_SEQUENCE_SELECTOR_MODE",
            "listwise",
        ).strip().lower()
        self.max_accepted_events = self._env_int(
            "ECML_SEQUENCE_MAX_ACCEPTED_EVENTS",
            1,
        )
        self.max_accepted_events_per_agent = self._env_int(
            "ECML_SEQUENCE_MAX_ACCEPTED_EVENTS_PER_AGENT",
            -1,
        )
        self.left_max_slack = self._env_float(
            "ECML_SEQUENCE_LEFT_MAX_SLACK",
            float("inf"),
        )
        self.trace_path = os.environ.get("ECML_SEQUENCE_TRACE_PATH", "").strip()
        self.trace_all = bool(self._env_int("ECML_SEQUENCE_TRACE_ALL", 0))
        self.first_diff_only = bool(
            self._env_int("ECML_SEQUENCE_FIRST_DIFF_ONLY", 0)
        )
        self.rejected_transitions = self._env_transition_set(
            "ECML_SEQUENCE_REJECT_TRANSITIONS",
            default="MOVE_RIGHT->MOVE_FORWARD",
        )
        self.aux_listwise_transitions = self._env_transition_set(
            "ECML_SEQUENCE_AUX_LISTWISE_TRANSITIONS",
            default=(
                "STOP_MOVING->MOVE_RIGHT,"
                "MOVE_FORWARD->MOVE_RIGHT"
            ),
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
            -0.25,
        )
        self.right_detour_min_value = self._env_float(
            "ECML_SEQUENCE_RIGHT_DETOUR_MIN_VALUE_LCB",
            -0.4,
        )
        self.left_detour_min_value = self._env_float(
            "ECML_SEQUENCE_LEFT_DETOUR_MIN_VALUE_LCB",
            -0.4,
        )
        self.left_to_forward_min_margin = self._env_float(
            "ECML_SEQUENCE_LEFT_TO_FORWARD_MIN_MARGIN",
            0.5,
        )
        self.stop_to_left_min_raw_margin = self._env_float(
            "ECML_SEQUENCE_STOP_TO_LEFT_MIN_RAW_MARGIN",
            3.0,
        )
        self.forward_to_left_min_raw_margin = self._env_float(
            "ECML_SEQUENCE_FORWARD_TO_LEFT_MIN_RAW_MARGIN",
            1.5,
        )
        self.stop_to_forward_min_raw_margin = self._env_float(
            "ECML_SEQUENCE_STOP_TO_FORWARD_MIN_RAW_MARGIN",
            1.0,
        )
        self.stop_to_forward_nonfinite_no_head_on_min_raw_margin = self._env_float(
            "ECML_SEQUENCE_STOP_TO_FORWARD_NONFINITE_NO_HEAD_ON_MIN_RAW_MARGIN",
            3.0,
        )
        self.stop_to_forward_max_distance_delta = self._env_float(
            "ECML_SEQUENCE_STOP_TO_FORWARD_MAX_DISTANCE_DELTA",
            100.0,
        )
        self.stop_to_forward_max_slack = self._env_float(
            "ECML_SEQUENCE_STOP_TO_FORWARD_MAX_SLACK",
            80.0,
        )
        self.first_detour_min_prefix_conflicts = self._env_float(
            "ECML_SEQUENCE_FIRST_DETOUR_MIN_PREFIX_CONFLICTS",
            1.0,
        )
        self.first_detour_low_conflict_min_margin = self._env_float(
            "ECML_SEQUENCE_FIRST_DETOUR_LOW_CONFLICT_MIN_LISTWISE_MARGIN",
            0.09,
        )
        self.right_detour_max_obs_intersections = self._env_float(
            "ECML_SEQUENCE_RIGHT_DETOUR_MAX_OBS_INTERSECTIONS",
            0.5,
        )
        self.right_detour_low_conflict_max_cells = self._env_float(
            "ECML_SEQUENCE_RIGHT_DETOUR_LOW_CONFLICT_MAX_CELLS",
            1.0,
        )
        self.right_detour_low_conflict_min_slack = self._env_float(
            "ECML_SEQUENCE_RIGHT_DETOUR_LOW_CONFLICT_MIN_SLACK",
            140.0,
        )
        self.right_detour_low_conflict_min_value = self._env_float(
            "ECML_SEQUENCE_RIGHT_DETOUR_LOW_CONFLICT_MIN_VALUE_LCB",
            0.0,
        )
        self.extra_candidate_stems = self._env_string_set(
            "ECML_SEQUENCE_EXTRA_CANDIDATE_STEMS",
            default="ecml_action_conflict_successdiv_penalty_ppo_seed3800_u10",
        )
        self.extra_candidate_max_accepted_events = self._env_int(
            "ECML_SEQUENCE_EXTRA_CANDIDATE_MAX_ACCEPTED_EVENTS",
            2,
        )
        self.extra_candidate_require_existing_event = bool(
            self._env_int("ECML_SEQUENCE_EXTRA_CANDIDATE_REQUIRE_EXISTING_EVENT", 1)
        )
        self.extra_candidate_min_success_probability = self._env_float(
            "ECML_SEQUENCE_EXTRA_CANDIDATE_MIN_SUCCESS_PROBABILITY",
            0.15,
        )
        self.extra_candidate_right_detour_min_value = self._env_float(
            "ECML_SEQUENCE_EXTRA_CANDIDATE_RIGHT_DETOUR_MIN_VALUE_LCB",
            -0.50,
        )
        self.extra_candidate_right_detour_low_conflict_min_value = self._env_float(
            "ECML_SEQUENCE_EXTRA_CANDIDATE_RIGHT_DETOUR_LOW_CONFLICT_MIN_VALUE_LCB",
            -0.30,
        )
        self.risk_head_policy = self._load_risk_head_policy(
            os.environ.get("ECML_SEQUENCE_RISK_HEAD_CHECKPOINT", "").strip()
        )
        self.risk_head_max_candidate = self._env_float(
            "ECML_SEQUENCE_RISK_HEAD_MAX_CANDIDATE",
            float("inf"),
        )
        self.risk_head_max_candidate_minus_baseline = self._env_float(
            "ECML_SEQUENCE_RISK_HEAD_MAX_CANDIDATE_MINUS_BASELINE",
            float("inf"),
        )
        self.risk_head_min_baseline_minus_candidate = self._env_float(
            "ECML_SEQUENCE_RISK_HEAD_MIN_BASELINE_MINUS_CANDIDATE",
            float("-inf"),
        )
        self.risk_relax_enabled = bool(
            self._env_int("ECML_SEQUENCE_RISK_RELAX", 0)
        )
        self.risk_relax_transitions = self._env_transition_set(
            "ECML_SEQUENCE_RISK_RELAX_TRANSITIONS",
            default="MOVE_FORWARD->MOVE_LEFT",
        )
        self.risk_relax_min_baseline_minus_candidate = self._env_float(
            "ECML_SEQUENCE_RISK_RELAX_MIN_BASELINE_MINUS_CANDIDATE",
            0.25,
        )
        self.risk_relax_max_candidate = self._env_float(
            "ECML_SEQUENCE_RISK_RELAX_MAX_CANDIDATE",
            0.50,
        )
        self.risk_relax_min_listwise_margin = self._env_float(
            "ECML_SEQUENCE_RISK_RELAX_MIN_LISTWISE_MARGIN",
            0.70,
        )
        self.risk_relax_min_raw_margin = self._env_float(
            "ECML_SEQUENCE_RISK_RELAX_MIN_RAW_MARGIN",
            0.10,
        )
        self.risk_relax_allow_reject_reason = bool(
            self._env_int("ECML_SEQUENCE_RISK_RELAX_ALLOW_REJECT_REASON", 0)
        )
        detour_model_path = os.environ.get("ECML_SEQUENCE_DETOUR_MODEL", "").strip()
        self.detour_scorer = (
            self._load_detour_scorer(detour_model_path)
            if detour_model_path and Path(detour_model_path).exists()
            else None
        )
        self.detour_transitions = self._env_transition_set(
            "ECML_SEQUENCE_DETOUR_TRANSITIONS",
            default="MOVE_FORWARD->MOVE_LEFT",
        )
        self.detour_min_baseline_minus_candidate = self._env_float(
            "ECML_SEQUENCE_DETOUR_MIN_BASELINE_MINUS_CANDIDATE",
            0.15,
        )
        self.detour_max_candidate_risk = self._env_float(
            "ECML_SEQUENCE_DETOUR_MAX_CANDIDATE_RISK",
            0.65,
        )
        self.detour_min_raw_margin = self._env_float(
            "ECML_SEQUENCE_DETOUR_MIN_RAW_MARGIN",
            0.05,
        )
        self.detour_min_prefix_cell_intersections = self._env_float(
            "ECML_SEQUENCE_DETOUR_MIN_PREFIX_CELL_INTERSECTIONS",
            20.0,
        )
        self.detour_allow_reject_reason = bool(
            self._env_int("ECML_SEQUENCE_DETOUR_ALLOW_REJECT_REASON", 0)
        )
        self._accepted_event_details: list[dict[str, Any]] = []
        self._seen_candidate_diff_policy_ids: set[int] = set()
        self._last_step: int | None = None

    @staticmethod
    def _load_risk_head_policy(checkpoint_path: str) -> ActorCritic | None:
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return None
        try:
            policy = ActorCritic(checkpoint_path=checkpoint_path)
            policy.eval()
            return policy
        except Exception:
            return None

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

    @staticmethod
    def _env_string_set(name: str, default: str = "") -> set[str]:
        value = os.environ.get(name, default)
        return {item.strip() for item in value.split(",") if item.strip()}

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
                0.75,
            ),
            value_std_coef=self._env_float("ECML_SEQUENCE_VALUE_STD_COEF", 2.0),
            bad_std_coef=self._env_float("ECML_SEQUENCE_BAD_STD_COEF", 2.0),
            success_std_coef=self._env_float("ECML_SEQUENCE_SUCCESS_STD_COEF", 0.0),
            max_success_regression_probability=self._env_float(
                "ECML_SEQUENCE_MAX_SUCCESS_REGRESSION_PROBABILITY",
                float("inf"),
            ),
            success_regression_std_coef=self._env_float(
                "ECML_SEQUENCE_SUCCESS_REGRESSION_STD_COEF",
                1.0,
            ),
        )

    def _load_detour_scorer(self, detour_model_path: str) -> SequenceEnsembleScorer:
        return SequenceEnsembleScorer(
            checkpoint_path=detour_model_path,
            min_utility=self._env_float("ECML_SEQUENCE_DETOUR_MIN_UTILITY", 0.0),
            max_bad_probability=self._env_float(
                "ECML_SEQUENCE_DETOUR_MAX_BAD_PROBABILITY",
                0.20,
            ),
            min_success_probability=self._env_float(
                "ECML_SEQUENCE_DETOUR_MIN_SUCCESS_PROBABILITY",
                0.10,
            ),
            value_std_coef=self._env_float("ECML_SEQUENCE_DETOUR_VALUE_STD_COEF", 2.0),
            bad_std_coef=self._env_float("ECML_SEQUENCE_DETOUR_BAD_STD_COEF", 2.0),
            success_std_coef=self._env_float(
                "ECML_SEQUENCE_DETOUR_SUCCESS_STD_COEF",
                0.0,
            ),
            max_success_regression_probability=self._env_float(
                "ECML_SEQUENCE_DETOUR_MAX_SUCCESS_REGRESSION_PROBABILITY",
                0.10,
            ),
            success_regression_std_coef=self._env_float(
                "ECML_SEQUENCE_DETOUR_SUCCESS_REGRESSION_STD_COEF",
                1.0,
            ),
        )

    def _load_listwise_scorer(
        self,
        listwise_model_path: str,
        env_prefix: str = "ECML_SEQUENCE_LISTWISE",
        default_margin_threshold: float = 1.0,
    ) -> ListwiseSequenceScorer:
        return ListwiseSequenceScorer(
            checkpoint_path=listwise_model_path,
            margin_threshold=self._env_float(
                f"{env_prefix}_MARGIN_THRESHOLD",
                default_margin_threshold,
            ),
            score_std_coef=self._env_float(
                f"{env_prefix}_SCORE_STD_COEF",
                0.5,
            ),
            baseline_std_coef=self._env_float(
                f"{env_prefix}_BASELINE_STD_COEF",
                0.5,
            ),
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

    def _candidate_checkpoint_path(self, candidate_policy: RerankPolicy) -> str:
        return self.candidate_policy_paths.get(id(candidate_policy), "")

    def _is_extra_candidate(
        self,
        candidate_policy: RerankPolicy,
        require_existing_event: bool = False,
    ) -> bool:
        if require_existing_event and not self._accepted_event_details:
            return False
        path = Path(self._candidate_checkpoint_path(candidate_policy))
        stem = path.stem
        return bool(stem and stem in self.extra_candidate_stems)

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        baseline_actions = super().act_many(handles, observations, **kwargs)
        if (
            not self.candidate_policies
            or (
                self.sequence_scorer is None
                and self.listwise_scorer is None
                and self.aux_listwise_scorer is None
            )
        ):
            return baseline_actions
        if (
            diff_row is None
            or add_delta_features is None
            or aggregate_event_features is None
            or candidate_source_features is None
        ):
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
        extra_candidate = self._is_extra_candidate(
            candidate_policy,
            require_existing_event=self.extra_candidate_require_existing_event,
        )
        if (
            self.max_accepted_events >= 0
            and len(self._accepted_event_details) >= self.max_accepted_events
        ):
            if (
                not extra_candidate
                or self.extra_candidate_max_accepted_events < 0
                or len(self._accepted_event_details)
                >= self.extra_candidate_max_accepted_events
            ):
                return False
        if (
            self.max_accepted_events_per_agent >= 0
            and sum(
                int(event.get("agent_id", -1) == handle)
                for event in self._accepted_event_details
            )
            >= self.max_accepted_events_per_agent
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
            detail.update(
                candidate_source_features(
                    candidate_policy.__class__.__name__,
                    self._candidate_checkpoint_path(candidate_policy),
                )
            )
            detail = add_delta_features(detail)
            detail["prefix_index"] = len(self._accepted_event_details) + 1
            aggregate = aggregate_event_features(
                [*self._accepted_event_details, detail]
            )
            aggregate.update(
                candidate_source_features(
                    candidate_policy.__class__.__name__,
                    self._candidate_checkpoint_path(candidate_policy),
                )
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
            accepted, scores = self._score_and_guard_candidate(
                aggregate=aggregate,
                detail=detail,
                baseline_action=baseline_action,
                candidate_action=candidate_action,
                extra_candidate=extra_candidate,
            )
            accepted, scores = self._apply_risk_head_guard(
                accepted=accepted,
                scores=scores,
                observation=observation,
                detail=detail,
                baseline_action=baseline_action,
                candidate_action=candidate_action,
            )
            if "selector_source" in scores:
                detail["selector_source"] = str(scores["selector_source"])
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

    def _apply_risk_head_guard(
        self,
        accepted: bool,
        scores: dict[str, float],
        observation: Any,
        detail: dict[str, Any],
        baseline_action: int,
        candidate_action: int,
    ) -> tuple[bool, dict[str, float]]:
        if self.risk_head_policy is None or observation is None:
            return accepted, scores
        try:
            with torch.no_grad():
                risk_logits = self.risk_head_policy.risk_logits(
                    np.asarray(observation, dtype=np.float32)
                )
                risk_probs = torch.sigmoid(risk_logits).squeeze(0).cpu().numpy()
            baseline_risk = float(risk_probs[int(baseline_action)])
            candidate_risk = float(risk_probs[int(candidate_action)])
        except Exception:
            return accepted, scores

        candidate_minus_baseline = candidate_risk - baseline_risk
        baseline_minus_candidate = baseline_risk - candidate_risk
        scores["risk_head_baseline"] = baseline_risk
        scores["risk_head_candidate"] = candidate_risk
        scores["risk_head_candidate_minus_baseline"] = candidate_minus_baseline
        scores["risk_head_baseline_minus_candidate"] = baseline_minus_candidate
        if not accepted:
            if self._risk_head_relaxes_candidate(
                scores=scores,
                detail=detail,
                baseline_action=baseline_action,
                candidate_action=candidate_action,
            ):
                scores["selector_source"] = "risk_relax"
                scores["risk_relax_original_reject_reason"] = str(
                    scores.get("reject_reason", "")
                )
                scores.pop("reject_reason", None)
                return True, scores
            detour_accepted, scores = self._detour_rescue_relaxes_candidate(
                scores=scores,
                detail=detail,
                baseline_action=baseline_action,
                candidate_action=candidate_action,
            )
            if detour_accepted:
                return True, scores
            return accepted, scores
        if candidate_risk > self.risk_head_max_candidate:
            scores["reject_reason"] = "risk_head_candidate_too_high"
            return False, scores
        if candidate_minus_baseline > self.risk_head_max_candidate_minus_baseline:
            scores["reject_reason"] = "risk_head_regression"
            return False, scores
        if baseline_minus_candidate < self.risk_head_min_baseline_minus_candidate:
            scores["reject_reason"] = "risk_head_insufficient_improvement"
            return False, scores
        return accepted, scores

    def _risk_head_relaxes_candidate(
        self,
        scores: dict[str, float],
        detail: dict[str, Any],
        baseline_action: int,
        candidate_action: int,
    ) -> bool:
        if not self.risk_relax_enabled:
            return False
        if not self.risk_relax_allow_reject_reason and scores.get("reject_reason"):
            return False
        if (
            self.risk_relax_transitions
            and (baseline_action, candidate_action) not in self.risk_relax_transitions
        ):
            return False
        try:
            baseline_minus_candidate = float(
                scores.get("risk_head_baseline_minus_candidate", float("-inf"))
            )
            candidate_risk = float(scores.get("risk_head_candidate", float("inf")))
            listwise_margin = float(scores.get("listwise_margin", float("-inf")))
            raw_margin = float(
                detail.get("candidate_raw_candidate_minus_baseline_logit", float("-inf"))
            )
        except Exception:
            return False
        return (
            baseline_minus_candidate
            >= self.risk_relax_min_baseline_minus_candidate
            and candidate_risk <= self.risk_relax_max_candidate
            and listwise_margin >= self.risk_relax_min_listwise_margin
            and raw_margin >= self.risk_relax_min_raw_margin
        )

    def _detour_rescue_relaxes_candidate(
        self,
        scores: dict[str, float],
        detail: dict[str, Any],
        baseline_action: int,
        candidate_action: int,
    ) -> tuple[bool, dict[str, float]]:
        if self.detour_scorer is None:
            return False, scores
        if not self.detour_allow_reject_reason and scores.get("reject_reason"):
            return False, scores
        if (
            self.detour_transitions
            and (baseline_action, candidate_action) not in self.detour_transitions
        ):
            return False, scores
        try:
            baseline_minus_candidate = float(
                scores.get("risk_head_baseline_minus_candidate", float("-inf"))
            )
            candidate_risk = float(scores.get("risk_head_candidate", float("inf")))
            raw_margin = float(
                detail.get("candidate_raw_candidate_minus_baseline_logit", float("-inf"))
            )
            prefix_cells = float(
                detail.get("candidate_prefix_cell_intersections", float("-inf"))
            )
        except Exception:
            return False, scores
        if baseline_minus_candidate < self.detour_min_baseline_minus_candidate:
            return False, scores
        if candidate_risk > self.detour_max_candidate_risk:
            return False, scores
        if raw_margin < self.detour_min_raw_margin:
            return False, scores
        if prefix_cells < self.detour_min_prefix_cell_intersections:
            return False, scores

        score_row = {
            **detail,
            **scores,
            "baseline_action": baseline_action,
            "candidate_action": candidate_action,
        }
        try:
            accepted, detour_scores = self.detour_scorer.score(score_row)
        except Exception:
            return False, scores
        for key, value in detour_scores.items():
            scores[f"detour_{key}"] = value
        if not accepted:
            return False, scores
        scores["selector_source"] = "detour_rescue"
        scores["detour_original_reject_reason"] = str(scores.get("reject_reason", ""))
        scores.pop("reject_reason", None)
        return True, scores

    def _apply_candidate_guards(
        self,
        accepted: bool,
        scores: dict[str, float],
        detail: dict[str, Any],
        baseline_action: int,
        candidate_action: int,
    ) -> tuple[bool, dict[str, float]]:
        if accepted and self._low_value_same_edge_candidate(detail, scores):
            accepted = False
            scores["reject_reason"] = "same_edge_low_value"
        if accepted and self._low_value_right_detour(
            baseline_action,
            candidate_action,
            scores,
        ):
            accepted = False
            scores["reject_reason"] = "right_detour_low_value"
        if accepted and self._low_value_left_detour(
            baseline_action,
            candidate_action,
            scores,
        ):
            accepted = False
            scores["reject_reason"] = "left_detour_low_value"
        if accepted and self._low_confidence_left_to_forward(
            baseline_action,
            candidate_action,
            detail,
        ):
            accepted = False
            scores["reject_reason"] = "left_to_forward_low_confidence"
        if accepted and self._low_confidence_stop_to_left(
            baseline_action,
            candidate_action,
            detail,
        ):
            accepted = False
            scores["reject_reason"] = "stop_to_left_low_confidence"
        if accepted and self._low_confidence_forward_to_left(
            baseline_action,
            candidate_action,
            detail,
        ):
            accepted = False
            scores["reject_reason"] = "forward_to_left_low_confidence"
        if accepted and self._low_confidence_stop_to_forward(
            baseline_action,
            candidate_action,
            detail,
        ):
            accepted = False
            scores["reject_reason"] = "stop_to_forward_low_confidence"
        if accepted and self._low_confidence_stop_to_forward_without_head_on(
            baseline_action,
            candidate_action,
            detail,
        ):
            accepted = False
            scores["reject_reason"] = (
                "stop_to_forward_nonfinite_no_head_on_low_confidence"
            )
        if accepted and self._bad_stop_to_forward_distance_delta(
            baseline_action,
            candidate_action,
            detail,
        ):
            accepted = False
            scores["reject_reason"] = "stop_to_forward_bad_distance_delta"
        if accepted and self._bad_stop_to_forward_slack(
            baseline_action,
            candidate_action,
            detail,
        ):
            accepted = False
            scores["reject_reason"] = "stop_to_forward_bad_slack"
        if accepted and self._low_conflict_first_detour(
            baseline_action,
            candidate_action,
            detail,
            scores,
        ):
            accepted = False
            scores["reject_reason"] = "first_detour_low_prefix_conflict"
        if accepted and self._crowded_right_detour(
            baseline_action,
            candidate_action,
            detail,
        ):
            accepted = False
            scores["reject_reason"] = "right_detour_crowded"
        if accepted and self._low_conflict_right_detour(
            baseline_action,
            candidate_action,
            detail,
            scores,
        ):
            accepted = False
            scores["reject_reason"] = "right_detour_low_conflict"
        return accepted, scores

    def _score_and_guard_candidate(
        self,
        aggregate: dict[str, Any],
        detail: dict[str, Any],
        baseline_action: int,
        candidate_action: int,
        extra_candidate: bool,
    ) -> tuple[bool, dict[str, float]]:
        if (
            (self.listwise_scorer is not None or self.aux_listwise_scorer is not None)
            and self.selector_mode in {"listwise", "listwise_primary"}
        ):
            score_row = dict(aggregate)
            score_row["prefix_len"] = len(self._accepted_event_details) + 1
            score_row["forced_applied"] = len(self._accepted_event_details) + 1
            score_row["event_details"] = [*self._accepted_event_details, detail]
            scores: dict[str, float] = {}
            if self.listwise_scorer is not None:
                accepted, scores = self.listwise_scorer.score(score_row)
                scores["selector_source"] = "listwise"
                accepted, scores = self._apply_candidate_guards(
                    accepted=accepted,
                    scores=scores,
                    detail=detail,
                    baseline_action=baseline_action,
                    candidate_action=candidate_action,
                )
                if accepted:
                    return True, scores

            if (
                self.aux_listwise_scorer is not None
                and (
                    not self.aux_listwise_transitions
                    or (baseline_action, candidate_action)
                    in self.aux_listwise_transitions
                )
            ):
                aux_accepted, aux_scores = self.aux_listwise_scorer.score(score_row)
                aux_scores["selector_source"] = "aux_listwise"
                aux_accepted, aux_scores = self._apply_candidate_guards(
                    accepted=aux_accepted,
                    scores=aux_scores,
                    detail=detail,
                    baseline_action=baseline_action,
                    candidate_action=candidate_action,
                )
                if aux_accepted:
                    return True, aux_scores
                if not scores:
                    return False, aux_scores

            return False, scores

        if self.sequence_scorer is None:
            return False, {}
        old_min_success = self.sequence_scorer.min_success_probability
        old_right_value = self.right_detour_min_value
        old_low_conflict_right_value = self.right_detour_low_conflict_min_value
        try:
            if extra_candidate:
                self.sequence_scorer.min_success_probability = (
                    self.extra_candidate_min_success_probability
                )
                self.right_detour_min_value = (
                    self.extra_candidate_right_detour_min_value
                )
                self.right_detour_low_conflict_min_value = (
                    self.extra_candidate_right_detour_low_conflict_min_value
                )
            accepted, scores = self.sequence_scorer.score(aggregate)
            scores["selector_source"] = "sequence"
            return self._apply_candidate_guards(
                accepted=accepted,
                scores=scores,
                detail=detail,
                baseline_action=baseline_action,
                candidate_action=candidate_action,
            )
        finally:
            self.sequence_scorer.min_success_probability = old_min_success
            self.right_detour_min_value = old_right_value
            self.right_detour_low_conflict_min_value = old_low_conflict_right_value

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

    def _low_value_right_detour(
        self,
        baseline_action: int,
        candidate_action: int,
        scores: dict[str, float],
    ) -> bool:
        if not np.isfinite(self.right_detour_min_value):
            return False
        if (
            baseline_action != ReservationPolicy.MOVE_FORWARD
            or candidate_action != ReservationPolicy.MOVE_RIGHT
        ):
            return False
        try:
            value_lcb = float(scores.get("value_lcb", 0.0))
        except Exception:
            return False
        return value_lcb < self.right_detour_min_value

    def _low_value_left_detour(
        self,
        baseline_action: int,
        candidate_action: int,
        scores: dict[str, float],
    ) -> bool:
        if not np.isfinite(self.left_detour_min_value):
            return False
        if (
            baseline_action != ReservationPolicy.MOVE_FORWARD
            or candidate_action != ReservationPolicy.MOVE_LEFT
        ):
            return False
        try:
            value_lcb = float(scores.get("value_lcb", 0.0))
        except Exception:
            return False
        return value_lcb < self.left_detour_min_value

    def _low_confidence_left_to_forward(
        self,
        baseline_action: int,
        candidate_action: int,
        detail: dict[str, Any],
    ) -> bool:
        if not np.isfinite(self.left_to_forward_min_margin):
            return False
        if (
            baseline_action != ReservationPolicy.MOVE_LEFT
            or candidate_action != ReservationPolicy.MOVE_FORWARD
        ):
            return False
        try:
            margin = float(detail.get("candidate_raw_top_logit_margin", 0.0))
        except Exception:
            return False
        return margin < self.left_to_forward_min_margin

    def _low_confidence_stop_to_left(
        self,
        baseline_action: int,
        candidate_action: int,
        detail: dict[str, Any],
    ) -> bool:
        if not np.isfinite(self.stop_to_left_min_raw_margin):
            return False
        if (
            baseline_action != ReservationPolicy.STOP_MOVING
            or candidate_action != ReservationPolicy.MOVE_LEFT
        ):
            return False
        try:
            margin = float(detail.get("candidate_raw_top_logit_margin", 0.0))
        except Exception:
            return False
        return margin < self.stop_to_left_min_raw_margin

    def _low_confidence_forward_to_left(
        self,
        baseline_action: int,
        candidate_action: int,
        detail: dict[str, Any],
    ) -> bool:
        if not np.isfinite(self.forward_to_left_min_raw_margin):
            return False
        if (
            baseline_action != ReservationPolicy.MOVE_FORWARD
            or candidate_action != ReservationPolicy.MOVE_LEFT
        ):
            return False
        try:
            margin = float(detail.get("candidate_raw_top_logit_margin", 0.0))
        except Exception:
            return False
        return margin < self.forward_to_left_min_raw_margin

    def _low_confidence_stop_to_forward(
        self,
        baseline_action: int,
        candidate_action: int,
        detail: dict[str, Any],
    ) -> bool:
        if not np.isfinite(self.stop_to_forward_min_raw_margin):
            return False
        if (
            baseline_action != ReservationPolicy.STOP_MOVING
            or candidate_action != ReservationPolicy.MOVE_FORWARD
        ):
            return False
        try:
            margin = float(detail.get("candidate_raw_top_logit_margin", 0.0))
        except Exception:
            return False
        return margin < self.stop_to_forward_min_raw_margin

    def _bad_stop_to_forward_distance_delta(
        self,
        baseline_action: int,
        candidate_action: int,
        detail: dict[str, Any],
    ) -> bool:
        if not np.isfinite(self.stop_to_forward_max_distance_delta):
            return False
        if (
            baseline_action != ReservationPolicy.STOP_MOVING
            or candidate_action != ReservationPolicy.MOVE_FORWARD
        ):
            return False
        try:
            distance_delta = float(detail.get("candidate_distance_delta", 0.0))
        except Exception:
            return False
        if not np.isfinite(distance_delta):
            return False
        return distance_delta > self.stop_to_forward_max_distance_delta

    def _low_confidence_stop_to_forward_without_head_on(
        self,
        baseline_action: int,
        candidate_action: int,
        detail: dict[str, Any],
    ) -> bool:
        if not np.isfinite(
            self.stop_to_forward_nonfinite_no_head_on_min_raw_margin
        ):
            return False
        if (
            baseline_action != ReservationPolicy.STOP_MOVING
            or candidate_action != ReservationPolicy.MOVE_FORWARD
        ):
            return False
        try:
            distance_delta = float(detail.get("candidate_distance_delta", 0.0))
            head_on = float(detail.get("obs_route_intersection_head_on", 0.0))
            margin = float(detail.get("candidate_raw_top_logit_margin", 0.0))
        except Exception:
            return False
        return (
            not np.isfinite(distance_delta)
            and head_on <= 0.0
            and margin < self.stop_to_forward_nonfinite_no_head_on_min_raw_margin
        )

    def _bad_stop_to_forward_slack(
        self,
        baseline_action: int,
        candidate_action: int,
        detail: dict[str, Any],
    ) -> bool:
        if not np.isfinite(self.stop_to_forward_max_slack):
            return False
        if (
            baseline_action != ReservationPolicy.STOP_MOVING
            or candidate_action != ReservationPolicy.MOVE_FORWARD
        ):
            return False
        try:
            slack = float(detail.get("slack", 0.0))
        except Exception:
            return False
        return slack > self.stop_to_forward_max_slack

    def _low_conflict_first_detour(
        self,
        baseline_action: int,
        candidate_action: int,
        detail: dict[str, Any],
        scores: dict[str, float],
    ) -> bool:
        if self.first_detour_min_prefix_conflicts <= 0:
            return False
        if self._accepted_event_details:
            return False
        if (
            baseline_action != ReservationPolicy.MOVE_FORWARD
            or candidate_action not in (
                ReservationPolicy.MOVE_LEFT,
                ReservationPolicy.MOVE_RIGHT,
            )
        ):
            return False
        try:
            prefix_conflicts = max(
                float(detail.get("candidate_prefix_cell_intersections", 0.0)),
                float(detail.get("candidate_prefix_head_on_edge_conflicts", 0.0)),
                float(detail.get("candidate_prefix_same_edge_conflicts", 0.0)),
            )
            margin = float(
                scores.get(
                    "listwise_margin",
                    scores.get("value_lcb", float("-inf")),
                )
            )
        except Exception:
            return False
        if (
            prefix_conflicts < self.first_detour_min_prefix_conflicts
            and margin >= self.first_detour_low_conflict_min_margin
        ):
            return False
        return prefix_conflicts < self.first_detour_min_prefix_conflicts

    def _crowded_right_detour(
        self,
        baseline_action: int,
        candidate_action: int,
        detail: dict[str, Any],
    ) -> bool:
        if not np.isfinite(self.right_detour_max_obs_intersections):
            return False
        if (
            baseline_action != ReservationPolicy.MOVE_FORWARD
            or candidate_action != ReservationPolicy.MOVE_RIGHT
        ):
            return False
        try:
            intersections = float(detail.get("obs_route_intersection_count", 0.0))
        except Exception:
            return False
        return intersections > self.right_detour_max_obs_intersections

    def _low_conflict_right_detour(
        self,
        baseline_action: int,
        candidate_action: int,
        detail: dict[str, Any],
        scores: dict[str, float],
    ) -> bool:
        if not np.isfinite(self.right_detour_low_conflict_max_cells):
            return False
        if (
            baseline_action != ReservationPolicy.MOVE_FORWARD
            or candidate_action != ReservationPolicy.MOVE_RIGHT
        ):
            return False
        try:
            cells = float(detail.get("candidate_prefix_cell_intersections", 0.0))
            slack = float(detail.get("slack", 0.0))
            value_lcb = float(scores.get("value_lcb", 0.0))
        except Exception:
            return False
        return (
            cells <= self.right_detour_low_conflict_max_cells
            and slack < self.right_detour_low_conflict_min_slack
            and value_lcb < self.right_detour_low_conflict_min_value
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
        context = runtime_context.get()
        row = {
            "accepted": bool(accepted),
            "seed": context.seed,
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
            "success_regression_ucb": float(
                scores.get("success_regression_ucb", 0.0)
            ),
            "listwise_margin": float(scores.get("listwise_margin", 0.0)),
            "listwise_candidate_lcb": float(
                scores.get("listwise_candidate_lcb", 0.0)
            ),
            "listwise_baseline_ucb": float(
                scores.get("listwise_baseline_ucb", 0.0)
            ),
            "accepted_event_count_before": len(self._accepted_event_details),
        }
        if "reject_reason" in scores:
            row["reject_reason"] = str(scores["reject_reason"])
        if "risk_relax_original_reject_reason" in scores:
            row["risk_relax_original_reject_reason"] = str(
                scores["risk_relax_original_reject_reason"]
            )
        if "detour_original_reject_reason" in scores:
            row["detour_original_reject_reason"] = str(
                scores["detour_original_reject_reason"]
            )
        if "selector_source" in scores:
            row["selector_source"] = str(scores["selector_source"])
        for key in (
            "risk_head_baseline",
            "risk_head_candidate",
            "risk_head_candidate_minus_baseline",
            "risk_head_baseline_minus_candidate",
            "detour_value_lcb",
            "detour_bad_ucb",
            "detour_success_lcb",
            "detour_success_regression_ucb",
        ):
            if key in scores:
                row[key] = float(scores[key])
        for key in (
            "slack",
            "distance",
            "candidate_distance_delta",
            "candidate_distance_delta_per_distance",
            "candidate_distance_delta_per_slack_abs",
            "candidate_distance_delta_minus_slack",
            "candidate_distance_delta_exceeds_slack",
            "is_stop_to_move",
            "is_stop_to_forward",
            "is_stop_to_left",
            "is_stop_to_right",
            "stop_distance_delta_per_distance",
            "stop_distance_delta_minus_slack",
            "stop_near_target_large_detour",
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
            "event_candidate_distance_delta_per_distance_last",
            "event_candidate_distance_delta_minus_slack_last",
            "event_stop_distance_delta_per_distance_last",
            "event_stop_distance_delta_minus_slack_last",
            "event_stop_near_target_large_detour_max",
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
