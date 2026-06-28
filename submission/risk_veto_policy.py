from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from flatland.core.grid.grid4_utils import get_new_position
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
        self.extra_candidate_policies: list[tuple[str, RerankPolicy]] = []
        for index, extra_checkpoint in enumerate(
            item.strip()
            for item in os.environ.get(
                "ECML_RISK_VETO_EXTRA_CANDIDATE_CHECKPOINTS",
                "",
            ).split(",")
        ):
            if not extra_checkpoint:
                continue
            self.extra_candidate_policies.append(
                (
                    f"extra_rerank_{index}",
                    RerankPolicy(checkpoint_path=extra_checkpoint),
                )
            )
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
        self.max_raw_topn_stop_right_distance_delta = self._env_float(
            "ECML_RISK_VETO_MAX_RAW_TOPN_STOP_RIGHT_DISTANCE_DELTA",
            float("inf"),
        )
        self.max_stop_right_distance_delta = self._env_float(
            "ECML_RISK_VETO_MAX_STOP_RIGHT_DISTANCE_DELTA",
            float("inf"),
        )
        self.require_stop_left_conflict = bool(
            int(os.environ.get("ECML_RISK_VETO_REQUIRE_STOP_LEFT_CONFLICT", "0") or "0")
        )
        self.stop_left_min_slack_for_unconflicted = self._env_float(
            "ECML_RISK_VETO_STOP_LEFT_MIN_SLACK_FOR_UNCONFLICTED",
            float("-inf"),
        )
        self.max_unconflicted_stop_left_distance_delta = self._env_float(
            "ECML_RISK_VETO_MAX_UNCONFLICTED_STOP_LEFT_DISTANCE_DELTA",
            float("inf"),
        )
        self.stop_left_same_edge_min_conflicts = self._env_float(
            "ECML_RISK_VETO_STOP_LEFT_SAME_EDGE_MIN_CONFLICTS",
            float("inf"),
        )
        self.stop_left_same_edge_max_eta_gap = self._env_float(
            "ECML_RISK_VETO_STOP_LEFT_SAME_EDGE_MAX_ETA_GAP",
            float("inf"),
        )
        self.stop_left_same_edge_relax_enabled = bool(
            int(
                os.environ.get(
                    "ECML_RISK_VETO_STOP_LEFT_SAME_EDGE_RELAX_ENABLED",
                    "0",
                )
                or "0"
            )
        )
        self.stop_left_same_edge_relax_max_candidate_risk = self._env_float(
            "ECML_RISK_VETO_STOP_LEFT_SAME_EDGE_RELAX_MAX_CANDIDATE_RISK",
            float("inf"),
        )
        self.stop_left_same_edge_relax_max_reward_risk = self._env_float(
            "ECML_RISK_VETO_STOP_LEFT_SAME_EDGE_RELAX_MAX_REWARD_RISK",
            float("inf"),
        )
        self.stop_left_same_edge_relax_min_risk_improvement = self._env_float(
            "ECML_RISK_VETO_STOP_LEFT_SAME_EDGE_RELAX_MIN_RISK_IMPROVEMENT",
            float("inf"),
        )
        self.stop_left_same_edge_relax_min_reward_improvement = self._env_float(
            "ECML_RISK_VETO_STOP_LEFT_SAME_EDGE_RELAX_MIN_REWARD_IMPROVEMENT",
            float("inf"),
        )
        self.deadline_right_relax_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_ENABLED", "0") or "0")
        )
        self.deadline_right_relax_min_step = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MIN_STEP",
            250.0,
        )
        self.deadline_right_relax_max_step = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MAX_STEP",
            float("inf"),
        )
        self.deadline_right_relax_max_candidate_risk = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MAX_CANDIDATE_RISK",
            0.17,
        )
        self.deadline_right_relax_min_risk_improvement = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MIN_RISK_IMPROVEMENT",
            0.32,
        )
        self.deadline_right_relax_max_reward_risk = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MAX_REWARD_RISK",
            0.56,
        )
        self.deadline_right_relax_min_reward_improvement = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MIN_REWARD_IMPROVEMENT",
            0.32,
        )
        self.deadline_right_relax_min_active_fraction = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MIN_ACTIVE_FRACTION",
            0.75,
        )
        self.deadline_right_relax_max_active_fraction = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MAX_ACTIVE_FRACTION",
            0.90,
        )
        self.deadline_right_relax_min_time_slack = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MIN_TIME_SLACK",
            0.25,
        )
        self.deadline_right_relax_max_time_slack = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MAX_TIME_SLACK",
            0.33,
        )
        self.deadline_right_relax_max_route_occupancy_distance = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MAX_ROUTE_OCCUPANCY_DISTANCE",
            0.03,
        )
        self.deadline_right_relax_min_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MIN_INTERSECTION_ETA_RISK",
            0.80,
        )
        self.deadline_right_relax_min_distance_delta = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MIN_DISTANCE_DELTA",
            250.0,
        )
        self.deadline_right_relax_max_distance_delta = self._env_float(
            "ECML_RISK_VETO_DEADLINE_RIGHT_RELAX_MAX_DISTANCE_DELTA",
            350.0,
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
                "insufficient_reward_risk_improvement"
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
        self.extra_relax_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_EXTRA_RELAX_ENABLED", "0") or "0")
        )
        self.extra_relax_allowed_sources = {
            source.strip()
            for source in os.environ.get(
                "ECML_RISK_VETO_EXTRA_RELAX_ALLOWED_SOURCES",
                "extra_rerank_0",
            ).split(",")
            if source.strip()
        }
        self.extra_relax_allowed_reject_reasons = {
            reason.strip()
            for reason in os.environ.get(
                "ECML_RISK_VETO_EXTRA_RELAX_ALLOWED_REJECT_REASONS",
                "candidate_risk_regression,reward_risk_too_high,reward_risk_regression",
            ).split(",")
            if reason.strip()
        }
        self.extra_relax_transitions = self._env_transition_set(
            "ECML_RISK_VETO_EXTRA_RELAX_TRANSITIONS",
        )
        self.extra_relax_min_step = self._env_float(
            "ECML_RISK_VETO_EXTRA_RELAX_MIN_STEP",
            float("-inf"),
        )
        self.extra_relax_max_step = self._env_float(
            "ECML_RISK_VETO_EXTRA_RELAX_MAX_STEP",
            float("inf"),
        )
        self.extra_relax_min_candidate_minus_baseline = self._env_float(
            "ECML_RISK_VETO_EXTRA_RELAX_MIN_CANDIDATE_MINUS_BASELINE",
            float("-inf"),
        )
        self.extra_relax_max_candidate_minus_baseline = self._env_float(
            "ECML_RISK_VETO_EXTRA_RELAX_MAX_CANDIDATE_MINUS_BASELINE",
            float("inf"),
        )
        self.extra_relax_max_candidate_risk = self._env_float(
            "ECML_RISK_VETO_EXTRA_RELAX_MAX_CANDIDATE_RISK",
            float("inf"),
        )
        self.extra_relax_max_reward_risk = self._env_float(
            "ECML_RISK_VETO_EXTRA_RELAX_MAX_REWARD_RISK",
            float("inf"),
        )
        self.extra_relax_max_reward_candidate_minus_baseline = self._env_float(
            "ECML_RISK_VETO_EXTRA_RELAX_MAX_REWARD_CANDIDATE_MINUS_BASELINE",
            float("inf"),
        )
        self.extra_relax_min_reward_baseline_minus_candidate = self._env_float(
            "ECML_RISK_VETO_EXTRA_RELAX_MIN_REWARD_BASELINE_MINUS_CANDIDATE",
            float("-inf"),
        )
        self.start_relax_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_START_RELAX_ENABLED", "0") or "0")
        )
        self.start_relax_allowed_sources = {
            source.strip()
            for source in os.environ.get(
                "ECML_RISK_VETO_START_RELAX_ALLOWED_SOURCES",
                "rerank",
            ).split(",")
            if source.strip()
        }
        self.start_relax_allowed_reject_reasons = {
            reason.strip()
            for reason in os.environ.get(
                "ECML_RISK_VETO_START_RELAX_ALLOWED_REJECT_REASONS",
                "candidate_risk_regression,reward_risk_too_high,reward_risk_regression",
            ).split(",")
            if reason.strip()
        }
        self.start_relax_transitions = self._env_transition_set(
            "ECML_RISK_VETO_START_RELAX_TRANSITIONS",
        )
        self.start_relax_min_step = self._env_float(
            "ECML_RISK_VETO_START_RELAX_MIN_STEP",
            float("-inf"),
        )
        self.start_relax_max_step = self._env_float(
            "ECML_RISK_VETO_START_RELAX_MAX_STEP",
            float("inf"),
        )
        self.start_relax_min_candidate_minus_baseline = self._env_float(
            "ECML_RISK_VETO_START_RELAX_MIN_CANDIDATE_MINUS_BASELINE",
            float("-inf"),
        )
        self.start_relax_max_candidate_minus_baseline = self._env_float(
            "ECML_RISK_VETO_START_RELAX_MAX_CANDIDATE_MINUS_BASELINE",
            float("inf"),
        )
        self.start_relax_max_candidate_risk = self._env_float(
            "ECML_RISK_VETO_START_RELAX_MAX_CANDIDATE_RISK",
            float("inf"),
        )
        self.start_relax_max_reward_risk = self._env_float(
            "ECML_RISK_VETO_START_RELAX_MAX_REWARD_RISK",
            float("inf"),
        )
        self.start_relax_max_reward_candidate_minus_baseline = self._env_float(
            "ECML_RISK_VETO_START_RELAX_MAX_REWARD_CANDIDATE_MINUS_BASELINE",
            float("inf"),
        )
        self.start_relax_min_reward_baseline_minus_candidate = self._env_float(
            "ECML_RISK_VETO_START_RELAX_MIN_REWARD_BASELINE_MINUS_CANDIDATE",
            float("-inf"),
        )
        self.start_relax_active_fraction_guard_min = self._env_float(
            "ECML_RISK_VETO_START_RELAX_ACTIVE_FRACTION_GUARD_MIN",
            float("inf"),
        )
        self.start_relax_active_fraction_min_reward_improvement = self._env_float(
            "ECML_RISK_VETO_START_RELAX_ACTIVE_FRACTION_MIN_REWARD_IMPROVEMENT",
            float("-inf"),
        )
        self.start_relax_active_distance_guard_min = self._env_float(
            "ECML_RISK_VETO_START_RELAX_ACTIVE_DISTANCE_GUARD_MIN",
            float("inf"),
        )
        self.start_relax_active_distance_max_delta = self._env_float(
            "ECML_RISK_VETO_START_RELAX_ACTIVE_DISTANCE_MAX_DELTA",
            float("-inf"),
        )
        self.start_relax_active_distance_min_reward_improvement = self._env_float(
            "ECML_RISK_VETO_START_RELAX_ACTIVE_DISTANCE_MIN_REWARD_IMPROVEMENT",
            float("-inf"),
        )
        self.start_relax_opposing_route_max_distance = self._env_float(
            "ECML_RISK_VETO_START_RELAX_OPPOSING_ROUTE_MAX_DISTANCE",
            float("-inf"),
        )
        self.start_relax_opposing_route_min_active_fraction = self._env_float(
            "ECML_RISK_VETO_START_RELAX_OPPOSING_ROUTE_MIN_ACTIVE_FRACTION",
            float("-inf"),
        )
        self.start_relax_opposing_route_max_active_fraction = self._env_float(
            "ECML_RISK_VETO_START_RELAX_OPPOSING_ROUTE_MAX_ACTIVE_FRACTION",
            float("inf"),
        )
        self.start_relax_opposing_route_min_candidate_minus_baseline = self._env_float(
            "ECML_RISK_VETO_START_RELAX_OPPOSING_ROUTE_MIN_CANDIDATE_MINUS_BASELINE",
            float("inf"),
        )
        self.start_relax_opposing_route_max_reward_improvement = self._env_float(
            "ECML_RISK_VETO_START_RELAX_OPPOSING_ROUTE_MAX_REWARD_IMPROVEMENT",
            float("-inf"),
        )
        self.trace_path = os.environ.get("ECML_RISK_VETO_TRACE_PATH", "").strip()
        self.start_delay_guard_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_START_DELAY_GUARD_ENABLED", "0") or "0")
        )
        self.start_delay_guard_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_START_DELAY_GUARD_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.start_delay_guard_min_step = self._env_float(
            "ECML_RISK_VETO_START_DELAY_GUARD_MIN_STEP",
            0.0,
        )
        self.start_delay_guard_max_step = self._env_float(
            "ECML_RISK_VETO_START_DELAY_GUARD_MAX_STEP",
            140.0,
        )
        self.start_delay_guard_max_holds = self._env_float(
            "ECML_RISK_VETO_START_DELAY_GUARD_MAX_HOLDS",
            24.0,
        )
        self.start_delay_guard_min_slack = self._env_float(
            "ECML_RISK_VETO_START_DELAY_GUARD_MIN_SLACK",
            55.0,
        )
        self.start_delay_guard_max_slack = self._env_float(
            "ECML_RISK_VETO_START_DELAY_GUARD_MAX_SLACK",
            float("inf"),
        )
        self.start_delay_guard_min_distance = self._env_float(
            "ECML_RISK_VETO_START_DELAY_GUARD_MIN_DISTANCE",
            120.0,
        )
        self.start_delay_guard_lookahead = self._env_float(
            "ECML_RISK_VETO_START_DELAY_GUARD_LOOKAHEAD",
            320.0,
        )
        self.start_delay_guard_eta_window = self._env_float(
            "ECML_RISK_VETO_START_DELAY_GUARD_ETA_WINDOW",
            12.0,
        )
        self.start_delay_guard_min_other_slack_advantage = self._env_float(
            "ECML_RISK_VETO_START_DELAY_GUARD_MIN_OTHER_SLACK_ADVANTAGE",
            5.0,
        )
        self.start_delay_guard_min_priority_conflicts = self._env_float(
            "ECML_RISK_VETO_START_DELAY_GUARD_MIN_PRIORITY_CONFLICTS",
            1.0,
        )
        self._start_delay_counts: dict[int, int] = {}
        self._start_delay_last_step: int | None = None
        self._start_delay_last_seed: int | None = None
        self.yield_stop_guard_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_YIELD_STOP_GUARD_ENABLED", "0") or "0")
        )
        self.yield_stop_guard_transitions = self._env_transition_set(
            "ECML_RISK_VETO_YIELD_STOP_GUARD_TRANSITIONS",
        ) or {(2, 4), (3, 4)}
        self.yield_stop_guard_min_step = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD_MIN_STEP",
            0.0,
        )
        self.yield_stop_guard_max_step = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD_MAX_STEP",
            90.0,
        )
        self.yield_stop_guard_max_holds = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD_MAX_HOLDS",
            1.0,
        )
        self.yield_stop_guard_min_slack = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD_MIN_SLACK",
            85.0,
        )
        self.yield_stop_guard_max_slack = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD_MAX_SLACK",
            float("inf"),
        )
        self.yield_stop_guard_min_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD_MIN_DISTANCE",
            100.0,
        )
        self.yield_stop_guard_max_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD_MAX_DISTANCE",
            float("inf"),
        )
        self.yield_stop_guard_min_tighter_agents = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD_MIN_TIGHTER_AGENTS",
            2.0,
        )
        self.yield_stop_guard_max_route_occupancy_count = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD_MAX_ROUTE_OCCUPANCY_COUNT",
            0.01,
        )
        self.yield_stop_guard_max_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD_MAX_INTERSECTION_ETA_RISK",
            0.20,
        )
        self.yield_stop_guard2_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_YIELD_STOP_GUARD2_ENABLED", "0") or "0")
        )
        self.yield_stop_guard2_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_YIELD_STOP_GUARD2_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.yield_stop_guard2_transitions = self._env_transition_set(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_TRANSITIONS",
        ) or {(1, 4), (3, 4)}
        self.yield_stop_guard2_min_step = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MIN_STEP",
            0.0,
        )
        self.yield_stop_guard2_max_step = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_STEP",
            float("inf"),
        )
        self.yield_stop_guard2_max_holds = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_HOLDS",
            1.0,
        )
        self.yield_stop_guard2_min_slack = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MIN_SLACK",
            float("-inf"),
        )
        self.yield_stop_guard2_max_slack = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_SLACK",
            float("inf"),
        )
        self.yield_stop_guard2_min_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MIN_DISTANCE",
            0.0,
        )
        self.yield_stop_guard2_max_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_DISTANCE",
            float("inf"),
        )
        self.yield_stop_guard2_min_tighter_agents = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MIN_TIGHTER_AGENTS",
            0.0,
        )
        self.yield_stop_guard2_min_active_fraction = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MIN_ACTIVE_FRACTION",
            float("-inf"),
        )
        self.yield_stop_guard2_max_active_fraction = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_ACTIVE_FRACTION",
            float("inf"),
        )
        self.yield_stop_guard2_min_stop_proximity = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MIN_STOP_PROXIMITY",
            float("-inf"),
        )
        self.yield_stop_guard2_max_stop_proximity = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_STOP_PROXIMITY",
            float("inf"),
        )
        self.yield_stop_guard2_min_route_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MIN_ROUTE_DISTANCE",
            float("-inf"),
        )
        self.yield_stop_guard2_max_route_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_ROUTE_DISTANCE",
            float("inf"),
        )
        self.yield_stop_guard2_min_route_opposing = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MIN_ROUTE_OPPOSING",
            0.5,
        )
        self.yield_stop_guard2_max_route_same_direction = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_ROUTE_SAME_DIRECTION",
            0.5,
        )
        self.yield_stop_guard2_max_route_other_tighter = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_ROUTE_OTHER_TIGHTER",
            0.5,
        )
        self.yield_stop_guard2_min_route_count = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MIN_ROUTE_COUNT",
            float("-inf"),
        )
        self.yield_stop_guard2_max_route_count = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_ROUTE_COUNT",
            float("inf"),
        )
        self.yield_stop_guard2_min_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MIN_INTERSECTION_OWN_DISTANCE",
            float("-inf"),
        )
        self.yield_stop_guard2_max_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_INTERSECTION_OWN_DISTANCE",
            float("inf"),
        )
        self.yield_stop_guard2_min_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MIN_INTERSECTION_ETA_RISK",
            float("-inf"),
        )
        self.yield_stop_guard2_max_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_INTERSECTION_ETA_RISK",
            float("inf"),
        )
        self.yield_stop_guard2_max_intersection_other_first = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_INTERSECTION_OTHER_FIRST",
            0.5,
        )
        self.yield_stop_guard2_max_intersection_other_tighter = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_INTERSECTION_OTHER_TIGHTER",
            0.5,
        )
        self.yield_stop_guard2_min_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MIN_PREFIX_CONFLICT_COUNT",
            float("-inf"),
        )
        self.yield_stop_guard2_max_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD2_MAX_PREFIX_CONFLICT_COUNT",
            float("inf"),
        )
        self.yield_stop_guard3_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_YIELD_STOP_GUARD3_ENABLED", "0") or "0")
        )
        self.yield_stop_guard3_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_YIELD_STOP_GUARD3_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.yield_stop_guard3_transitions = self._env_transition_set(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_TRANSITIONS",
        ) or {(2, 4)}
        self.yield_stop_guard3_min_step = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MIN_STEP",
            0.0,
        )
        self.yield_stop_guard3_max_step = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_STEP",
            float("inf"),
        )
        self.yield_stop_guard3_max_holds = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_HOLDS",
            1.0,
        )
        self.yield_stop_guard3_min_slack = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MIN_SLACK",
            float("-inf"),
        )
        self.yield_stop_guard3_max_slack = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_SLACK",
            float("inf"),
        )
        self.yield_stop_guard3_min_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MIN_DISTANCE",
            0.0,
        )
        self.yield_stop_guard3_max_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_DISTANCE",
            float("inf"),
        )
        self.yield_stop_guard3_min_tighter_agents = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MIN_TIGHTER_AGENTS",
            0.0,
        )
        self.yield_stop_guard3_max_on_switch = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_ON_SWITCH",
            0.5,
        )
        self.yield_stop_guard3_min_active_fraction = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MIN_ACTIVE_FRACTION",
            float("-inf"),
        )
        self.yield_stop_guard3_max_active_fraction = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_ACTIVE_FRACTION",
            float("inf"),
        )
        self.yield_stop_guard3_min_stop_proximity = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MIN_STOP_PROXIMITY",
            float("-inf"),
        )
        self.yield_stop_guard3_max_stop_proximity = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_STOP_PROXIMITY",
            float("inf"),
        )
        self.yield_stop_guard3_min_route_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MIN_ROUTE_DISTANCE",
            float("-inf"),
        )
        self.yield_stop_guard3_max_route_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_ROUTE_DISTANCE",
            float("inf"),
        )
        self.yield_stop_guard3_max_route_opposing = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_ROUTE_OPPOSING",
            0.5,
        )
        self.yield_stop_guard3_max_route_same_direction = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_ROUTE_SAME_DIRECTION",
            0.5,
        )
        self.yield_stop_guard3_max_route_other_tighter = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_ROUTE_OTHER_TIGHTER",
            0.5,
        )
        self.yield_stop_guard3_min_route_count = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MIN_ROUTE_COUNT",
            float("-inf"),
        )
        self.yield_stop_guard3_max_route_count = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_ROUTE_COUNT",
            float("inf"),
        )
        self.yield_stop_guard3_min_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MIN_INTERSECTION_OWN_DISTANCE",
            float("-inf"),
        )
        self.yield_stop_guard3_max_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_INTERSECTION_OWN_DISTANCE",
            float("inf"),
        )
        self.yield_stop_guard3_min_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MIN_INTERSECTION_ETA_RISK",
            float("-inf"),
        )
        self.yield_stop_guard3_max_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_INTERSECTION_ETA_RISK",
            float("inf"),
        )
        self.yield_stop_guard3_max_intersection_other_first = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_INTERSECTION_OTHER_FIRST",
            0.5,
        )
        self.yield_stop_guard3_max_intersection_other_tighter = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_INTERSECTION_OTHER_TIGHTER",
            0.5,
        )
        self.yield_stop_guard3_min_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MIN_PREFIX_CONFLICT_COUNT",
            float("-inf"),
        )
        self.yield_stop_guard3_max_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_YIELD_STOP_GUARD3_MAX_PREFIX_CONFLICT_COUNT",
            float("inf"),
        )
        self._yield_stop_counts: dict[int, int] = {}
        self._yield_stop_last_step: int | None = None
        self._yield_stop_last_seed: int | None = None
        self.detour_left_guard_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_DETOUR_LEFT_GUARD_ENABLED", "0") or "0")
        )
        self.detour_left_guard_min_step = self._env_float(
            "ECML_RISK_VETO_DETOUR_LEFT_GUARD_MIN_STEP",
            60.0,
        )
        self.detour_left_guard_max_step = self._env_float(
            "ECML_RISK_VETO_DETOUR_LEFT_GUARD_MAX_STEP",
            80.0,
        )
        self.detour_left_guard_max_holds = self._env_float(
            "ECML_RISK_VETO_DETOUR_LEFT_GUARD_MAX_HOLDS",
            1.0,
        )
        self.detour_left_guard_min_slack = self._env_float(
            "ECML_RISK_VETO_DETOUR_LEFT_GUARD_MIN_SLACK",
            80.0,
        )
        self.detour_left_guard_max_slack = self._env_float(
            "ECML_RISK_VETO_DETOUR_LEFT_GUARD_MAX_SLACK",
            85.0,
        )
        self.detour_left_guard_min_distance = self._env_float(
            "ECML_RISK_VETO_DETOUR_LEFT_GUARD_MIN_DISTANCE",
            235.0,
        )
        self.detour_left_guard_max_distance = self._env_float(
            "ECML_RISK_VETO_DETOUR_LEFT_GUARD_MAX_DISTANCE",
            250.0,
        )
        self.detour_left_guard_max_tighter_agents = self._env_float(
            "ECML_RISK_VETO_DETOUR_LEFT_GUARD_MAX_TIGHTER_AGENTS",
            1.0,
        )
        self.detour_left_guard_min_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_DETOUR_LEFT_GUARD_MIN_INTERSECTION_ETA_RISK",
            0.30,
        )
        self.detour_left_guard_max_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_DETOUR_LEFT_GUARD_MAX_INTERSECTION_ETA_RISK",
            0.45,
        )
        self.detour_left_guard_max_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_DETOUR_LEFT_GUARD_MAX_INTERSECTION_OWN_DISTANCE",
            0.03,
        )
        self._detour_left_counts: dict[int, int] = {}
        self._detour_left_last_step: int | None = None
        self._detour_left_last_seed: int | None = None
        self.switch_escape_guard_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_ENABLED", "0") or "0")
        )
        self.switch_escape_guard_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.switch_escape_guard_allowed_actions = self._env_action_set(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_ALLOWED_ACTIONS",
        ) or {2}
        self.switch_escape_guard_min_step = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MIN_STEP",
            145.0,
        )
        self.switch_escape_guard_max_step = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MAX_STEP",
            175.0,
        )
        self.switch_escape_guard_max_holds = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MAX_HOLDS",
            1.0,
        )
        self.switch_escape_guard_min_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MIN_DISTANCE",
            50.0,
        )
        self.switch_escape_guard_max_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MAX_DISTANCE",
            75.0,
        )
        self.switch_escape_guard_min_slack = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MIN_SLACK",
            35.0,
        )
        self.switch_escape_guard_max_slack = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MAX_SLACK",
            50.0,
        )
        self.switch_escape_guard_min_active_fraction = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MIN_ACTIVE_FRACTION",
            0.75,
        )
        self.switch_escape_guard_min_stop_proximity = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MIN_STOP_PROXIMITY",
            0.70,
        )
        self.switch_escape_guard_max_route_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MAX_ROUTE_DISTANCE",
            0.30,
        )
        self.switch_escape_guard_min_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MIN_INTERSECTION_ETA_RISK",
            0.75,
        )
        self.switch_escape_guard_max_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MAX_INTERSECTION_OWN_DISTANCE",
            0.04,
        )
        self.switch_escape_guard_min_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MIN_PREFIX_CONFLICT_COUNT",
            0.25,
        )
        self.switch_escape_guard_min_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MIN_DISTANCE_DELTA",
            -5.0,
        )
        self.switch_escape_guard_max_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_ESCAPE_GUARD_MAX_DISTANCE_DELTA",
            0.0,
        )
        self._switch_escape_counts: dict[int, int] = {}
        self._switch_escape_last_step: int | None = None
        self._switch_escape_last_seed: int | None = None
        self.switch_forward_guard_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_SWITCH_FORWARD_GUARD_ENABLED", "0") or "0")
        )
        self.switch_forward_guard_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.switch_forward_guard_min_step = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MIN_STEP",
            279.0,
        )
        self.switch_forward_guard_max_step = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MAX_STEP",
            279.0,
        )
        self.switch_forward_guard_max_holds = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MAX_HOLDS",
            1.0,
        )
        self.switch_forward_guard_min_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MIN_DISTANCE",
            110.0,
        )
        self.switch_forward_guard_max_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MAX_DISTANCE",
            118.0,
        )
        self.switch_forward_guard_min_slack = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MIN_SLACK",
            72.0,
        )
        self.switch_forward_guard_max_slack = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MAX_SLACK",
            78.0,
        )
        self.switch_forward_guard_min_active_fraction = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MIN_ACTIVE_FRACTION",
            0.45,
        )
        self.switch_forward_guard_max_active_fraction = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MAX_ACTIVE_FRACTION",
            0.55,
        )
        self.switch_forward_guard_min_stop_proximity = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MIN_STOP_PROXIMITY",
            0.55,
        )
        self.switch_forward_guard_min_route_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MIN_ROUTE_DISTANCE",
            0.48,
        )
        self.switch_forward_guard_max_route_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MAX_ROUTE_DISTANCE",
            0.54,
        )
        self.switch_forward_guard_max_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MAX_INTERSECTION_OWN_DISTANCE",
            0.03,
        )
        self.switch_forward_guard_min_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MIN_INTERSECTION_ETA_RISK",
            0.64,
        )
        self.switch_forward_guard_max_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MAX_INTERSECTION_ETA_RISK",
            0.70,
        )
        self.switch_forward_guard_min_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MIN_PREFIX_CONFLICT_COUNT",
            0.60,
        )
        self.switch_forward_guard_max_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MAX_PREFIX_CONFLICT_COUNT",
            0.70,
        )
        self.switch_forward_guard_min_priority_tighter_fraction = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MIN_PRIORITY_TIGHTER_FRACTION",
            0.55,
        )
        self.switch_forward_guard_max_priority_tighter_fraction = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MAX_PRIORITY_TIGHTER_FRACTION",
            0.65,
        )
        self.switch_forward_guard_min_forward_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MIN_FORWARD_DISTANCE_DELTA",
            -1.5,
        )
        self.switch_forward_guard_max_forward_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MAX_FORWARD_DISTANCE_DELTA",
            -0.5,
        )
        self.switch_forward_guard_min_left_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MIN_LEFT_DISTANCE_DELTA",
            -1.5,
        )
        self.switch_forward_guard_max_left_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD_MAX_LEFT_DISTANCE_DELTA",
            -0.5,
        )
        self.switch_forward_guard2_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_ENABLED", "0") or "0")
        )
        self.switch_forward_guard2_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.switch_forward_guard2_min_step = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MIN_STEP",
            0.0,
        )
        self.switch_forward_guard2_max_step = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MAX_STEP",
            float("inf"),
        )
        self.switch_forward_guard2_max_holds = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MAX_HOLDS",
            1.0,
        )
        self.switch_forward_guard2_min_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MIN_DISTANCE",
            0.0,
        )
        self.switch_forward_guard2_max_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MAX_DISTANCE",
            float("inf"),
        )
        self.switch_forward_guard2_min_slack = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MIN_SLACK",
            float("-inf"),
        )
        self.switch_forward_guard2_max_slack = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MAX_SLACK",
            float("inf"),
        )
        self.switch_forward_guard2_min_active_fraction = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MIN_ACTIVE_FRACTION",
            float("-inf"),
        )
        self.switch_forward_guard2_max_active_fraction = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MAX_ACTIVE_FRACTION",
            float("inf"),
        )
        self.switch_forward_guard2_min_stop_proximity = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MIN_STOP_PROXIMITY",
            float("-inf"),
        )
        self.switch_forward_guard2_min_route_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MIN_ROUTE_DISTANCE",
            float("-inf"),
        )
        self.switch_forward_guard2_max_route_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MAX_ROUTE_DISTANCE",
            float("inf"),
        )
        self.switch_forward_guard2_max_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MAX_INTERSECTION_OWN_DISTANCE",
            float("inf"),
        )
        self.switch_forward_guard2_min_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MIN_INTERSECTION_ETA_RISK",
            float("-inf"),
        )
        self.switch_forward_guard2_max_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MAX_INTERSECTION_ETA_RISK",
            float("inf"),
        )
        self.switch_forward_guard2_min_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MIN_PREFIX_CONFLICT_COUNT",
            float("-inf"),
        )
        self.switch_forward_guard2_max_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MAX_PREFIX_CONFLICT_COUNT",
            float("inf"),
        )
        self.switch_forward_guard2_min_priority_tighter_fraction = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MIN_PRIORITY_TIGHTER_FRACTION",
            float("-inf"),
        )
        self.switch_forward_guard2_max_priority_tighter_fraction = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MAX_PRIORITY_TIGHTER_FRACTION",
            float("inf"),
        )
        self.switch_forward_guard2_min_forward_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MIN_FORWARD_DISTANCE_DELTA",
            float("-inf"),
        )
        self.switch_forward_guard2_max_forward_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MAX_FORWARD_DISTANCE_DELTA",
            float("inf"),
        )
        self.switch_forward_guard2_min_left_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MIN_LEFT_DISTANCE_DELTA",
            float("-inf"),
        )
        self.switch_forward_guard2_max_left_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD2_MAX_LEFT_DISTANCE_DELTA",
            float("inf"),
        )
        self.switch_forward_guard3_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_ENABLED", "0") or "0")
        )
        self.switch_forward_guard3_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.switch_forward_guard3_min_step = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MIN_STEP",
            0.0,
        )
        self.switch_forward_guard3_max_step = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_STEP",
            float("inf"),
        )
        self.switch_forward_guard3_max_holds = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_HOLDS",
            1.0,
        )
        self.switch_forward_guard3_min_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MIN_DISTANCE",
            0.0,
        )
        self.switch_forward_guard3_max_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_DISTANCE",
            float("inf"),
        )
        self.switch_forward_guard3_min_slack = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MIN_SLACK",
            float("-inf"),
        )
        self.switch_forward_guard3_max_slack = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_SLACK",
            float("inf"),
        )
        self.switch_forward_guard3_min_active_fraction = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MIN_ACTIVE_FRACTION",
            float("-inf"),
        )
        self.switch_forward_guard3_max_active_fraction = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_ACTIVE_FRACTION",
            float("inf"),
        )
        self.switch_forward_guard3_min_stop_proximity = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MIN_STOP_PROXIMITY",
            float("-inf"),
        )
        self.switch_forward_guard3_min_route_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MIN_ROUTE_DISTANCE",
            float("-inf"),
        )
        self.switch_forward_guard3_max_route_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_ROUTE_DISTANCE",
            float("inf"),
        )
        self.switch_forward_guard3_max_route_other_tighter = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_ROUTE_OTHER_TIGHTER",
            0.5,
        )
        self.switch_forward_guard3_max_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_INTERSECTION_OWN_DISTANCE",
            float("inf"),
        )
        self.switch_forward_guard3_min_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MIN_INTERSECTION_ETA_RISK",
            float("-inf"),
        )
        self.switch_forward_guard3_max_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_INTERSECTION_ETA_RISK",
            float("inf"),
        )
        self.switch_forward_guard3_max_intersection_other_tighter = (
            self._env_float(
                "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_INTERSECTION_OTHER_TIGHTER",
                0.5,
            )
        )
        self.switch_forward_guard3_min_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MIN_PREFIX_CONFLICT_COUNT",
            float("-inf"),
        )
        self.switch_forward_guard3_max_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_PREFIX_CONFLICT_COUNT",
            float("inf"),
        )
        self.switch_forward_guard3_min_priority_tighter_fraction = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MIN_PRIORITY_TIGHTER_FRACTION",
            float("-inf"),
        )
        self.switch_forward_guard3_max_priority_tighter_fraction = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_PRIORITY_TIGHTER_FRACTION",
            float("inf"),
        )
        self.switch_forward_guard3_min_forward_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MIN_FORWARD_DISTANCE_DELTA",
            float("-inf"),
        )
        self.switch_forward_guard3_max_forward_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_FORWARD_DISTANCE_DELTA",
            float("inf"),
        )
        self.switch_forward_guard3_min_left_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MIN_LEFT_DISTANCE_DELTA",
            float("-inf"),
        )
        self.switch_forward_guard3_max_left_distance_delta = self._env_float(
            "ECML_RISK_VETO_SWITCH_FORWARD_GUARD3_MAX_LEFT_DISTANCE_DELTA",
            float("inf"),
        )
        self._switch_forward_counts: dict[int, int] = {}
        self._switch_forward_last_step: int | None = None
        self._switch_forward_last_seed: int | None = None
        self.right_forward_guard_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_RIGHT_FORWARD_GUARD_ENABLED", "0") or "0")
        )
        self.right_forward_guard_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.right_forward_guard_min_step = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_STEP",
            0.0,
        )
        self.right_forward_guard_max_step = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_STEP",
            float("inf"),
        )
        self.right_forward_guard_max_holds = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_HOLDS",
            1.0,
        )
        self.right_forward_guard_min_distance = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_DISTANCE",
            0.0,
        )
        self.right_forward_guard_max_distance = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_DISTANCE",
            float("inf"),
        )
        self.right_forward_guard_min_slack = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_SLACK",
            float("-inf"),
        )
        self.right_forward_guard_max_slack = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_SLACK",
            float("inf"),
        )
        self.right_forward_guard_min_active_fraction = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_ACTIVE_FRACTION",
            float("-inf"),
        )
        self.right_forward_guard_max_active_fraction = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_ACTIVE_FRACTION",
            float("inf"),
        )
        self.right_forward_guard_min_stop_proximity = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_STOP_PROXIMITY",
            float("-inf"),
        )
        self.right_forward_guard_max_stop_proximity = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_STOP_PROXIMITY",
            float("inf"),
        )
        self.right_forward_guard_min_route_distance = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_ROUTE_DISTANCE",
            float("-inf"),
        )
        self.right_forward_guard_max_route_distance = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_ROUTE_DISTANCE",
            float("inf"),
        )
        self.right_forward_guard_min_route_count = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_ROUTE_COUNT",
            float("-inf"),
        )
        self.right_forward_guard_max_route_count = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_ROUTE_COUNT",
            float("inf"),
        )
        self.right_forward_guard_min_route_other_tighter = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_ROUTE_OTHER_TIGHTER",
            0.5,
        )
        self.right_forward_guard_min_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_INTERSECTION_ETA_RISK",
            float("-inf"),
        )
        self.right_forward_guard_max_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_INTERSECTION_ETA_RISK",
            float("inf"),
        )
        self.right_forward_guard_max_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_INTERSECTION_OWN_DISTANCE",
            float("inf"),
        )
        self.right_forward_guard_min_intersection_other_tighter = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_INTERSECTION_OTHER_TIGHTER",
            0.5,
        )
        self.right_forward_guard_min_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_PREFIX_CONFLICT_COUNT",
            float("-inf"),
        )
        self.right_forward_guard_max_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_PREFIX_CONFLICT_COUNT",
            float("inf"),
        )
        self.right_forward_guard_min_priority_tighter_fraction = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_PRIORITY_TIGHTER_FRACTION",
            float("-inf"),
        )
        self.right_forward_guard_max_priority_tighter_fraction = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_PRIORITY_TIGHTER_FRACTION",
            float("inf"),
        )
        self.right_forward_guard_min_forward_distance_delta = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_FORWARD_DISTANCE_DELTA",
            float("-inf"),
        )
        self.right_forward_guard_max_forward_distance_delta = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_FORWARD_DISTANCE_DELTA",
            float("inf"),
        )
        self.right_forward_guard_min_right_distance_delta = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MIN_RIGHT_DISTANCE_DELTA",
            float("-inf"),
        )
        self.right_forward_guard_max_right_distance_delta = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_MAX_RIGHT_DISTANCE_DELTA",
            float("inf"),
        )
        self.right_forward_guard_scene5_max_step = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_SCENE5_MAX_STEP",
            float("inf"),
        )
        self.right_forward_guard_scene5_max_distance = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_SCENE5_MAX_DISTANCE",
            float("inf"),
        )
        self.right_forward_guard_scene5_max_slack = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_SCENE5_MAX_SLACK",
            float("inf"),
        )
        self.right_forward_guard_scene5_max_route_other_tighter = self._env_float(
            "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_SCENE5_MAX_ROUTE_OTHER_TIGHTER",
            float("inf"),
        )
        self.right_forward_guard_scene5_max_intersection_other_tighter = (
            self._env_float(
                "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_SCENE5_MAX_INTERSECTION_OTHER_TIGHTER",
                float("inf"),
            )
        )
        self.right_forward_guard_scene5_max_priority_tighter_fraction = (
            self._env_float(
                "ECML_RISK_VETO_RIGHT_FORWARD_GUARD_SCENE5_MAX_PRIORITY_TIGHTER_FRACTION",
                float("inf"),
            )
        )
        self._right_forward_counts: dict[int, int] = {}
        self._right_forward_last_step: int | None = None
        self._right_forward_last_seed: int | None = None
        self.deadline_forward_guard_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_ENABLED", "0") or "0")
        )
        self.deadline_forward_guard_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.deadline_forward_guard_min_step = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MIN_STEP",
            0.0,
        )
        self.deadline_forward_guard_max_step = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MAX_STEP",
            float("inf"),
        )
        self.deadline_forward_guard_min_distance = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MIN_DISTANCE",
            0.0,
        )
        self.deadline_forward_guard_max_distance = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MAX_DISTANCE",
            float("inf"),
        )
        self.deadline_forward_guard_min_slack = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MIN_SLACK",
            float("-inf"),
        )
        self.deadline_forward_guard_max_slack = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MAX_SLACK",
            float("inf"),
        )
        self.deadline_forward_guard_min_active_fraction = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MIN_ACTIVE_FRACTION",
            float("-inf"),
        )
        self.deadline_forward_guard_max_active_fraction = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MAX_ACTIVE_FRACTION",
            float("inf"),
        )
        self.deadline_forward_guard_min_stop_proximity = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MIN_STOP_PROXIMITY",
            float("-inf"),
        )
        self.deadline_forward_guard_min_time_slack = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MIN_TIME_SLACK",
            float("-inf"),
        )
        self.deadline_forward_guard_max_time_slack = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MAX_TIME_SLACK",
            float("inf"),
        )
        self.deadline_forward_guard_min_route_distance = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MIN_ROUTE_DISTANCE",
            float("-inf"),
        )
        self.deadline_forward_guard_max_route_distance = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MAX_ROUTE_DISTANCE",
            float("inf"),
        )
        self.deadline_forward_guard_min_route_count = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MIN_ROUTE_COUNT",
            float("-inf"),
        )
        self.deadline_forward_guard_max_route_count = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MAX_ROUTE_COUNT",
            float("inf"),
        )
        self.deadline_forward_guard_max_route_other_tighter = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MAX_ROUTE_OTHER_TIGHTER",
            float("inf"),
        )
        self.deadline_forward_guard_max_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MAX_INTERSECTION_OWN_DISTANCE",
            float("inf"),
        )
        self.deadline_forward_guard_min_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MIN_INTERSECTION_ETA_RISK",
            float("-inf"),
        )
        self.deadline_forward_guard_max_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MAX_INTERSECTION_ETA_RISK",
            float("inf"),
        )
        self.deadline_forward_guard_max_intersection_other_tighter = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MAX_INTERSECTION_OTHER_TIGHTER",
            float("inf"),
        )
        self.deadline_forward_guard_min_forward_corridor_len = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MIN_FORWARD_CORRIDOR_LEN",
            0.0,
        )
        self.deadline_forward_guard_max_forward_corridor_len = self._env_float(
            "ECML_RISK_VETO_DEADLINE_FORWARD_GUARD_MAX_FORWARD_CORRIDOR_LEN",
            float("inf"),
        )
        self.inf_right_guard_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_INF_RIGHT_GUARD_ENABLED", "0") or "0")
        )
        self.inf_right_guard_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_INF_RIGHT_GUARD_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.inf_right_guard_min_step = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MIN_STEP",
            72.0,
        )
        self.inf_right_guard_max_step = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MAX_STEP",
            72.0,
        )
        self.inf_right_guard_max_holds = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MAX_HOLDS",
            1.0,
        )
        self.inf_right_guard_min_distance = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MIN_DISTANCE",
            230.0,
        )
        self.inf_right_guard_max_distance = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MAX_DISTANCE",
            240.0,
        )
        self.inf_right_guard_min_slack = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MIN_SLACK",
            135.0,
        )
        self.inf_right_guard_max_slack = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MAX_SLACK",
            145.0,
        )
        self.inf_right_guard_min_active_fraction = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MIN_ACTIVE_FRACTION",
            0.60,
        )
        self.inf_right_guard_max_active_fraction = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MAX_ACTIVE_FRACTION",
            0.70,
        )
        self.inf_right_guard_max_stop_proximity = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MAX_STOP_PROXIMITY",
            0.20,
        )
        self.inf_right_guard_min_route_distance = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MIN_ROUTE_DISTANCE",
            0.05,
        )
        self.inf_right_guard_max_route_distance = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MAX_ROUTE_DISTANCE",
            0.08,
        )
        self.inf_right_guard_min_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MIN_INTERSECTION_OWN_DISTANCE",
            0.10,
        )
        self.inf_right_guard_max_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MAX_INTERSECTION_OWN_DISTANCE",
            0.12,
        )
        self.inf_right_guard_min_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MIN_INTERSECTION_ETA_RISK",
            0.90,
        )
        self.inf_right_guard_max_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MAX_INTERSECTION_ETA_RISK",
            0.93,
        )
        self.inf_right_guard_min_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MIN_PREFIX_CONFLICT_COUNT",
            0.30,
        )
        self.inf_right_guard_max_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MAX_PREFIX_CONFLICT_COUNT",
            0.35,
        )
        self.inf_right_guard_min_priority_tighter_fraction = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MIN_PRIORITY_TIGHTER_FRACTION",
            0.55,
        )
        self.inf_right_guard_max_priority_tighter_fraction = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MAX_PRIORITY_TIGHTER_FRACTION",
            0.65,
        )
        self.inf_right_guard_min_right_distance_delta = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MIN_RIGHT_DISTANCE_DELTA",
            -1.5,
        )
        self.inf_right_guard_max_right_distance_delta = self._env_float(
            "ECML_RISK_VETO_INF_RIGHT_GUARD_MAX_RIGHT_DISTANCE_DELTA",
            -0.5,
        )
        self._inf_right_counts: dict[int, int] = {}
        self._inf_right_last_step: int | None = None
        self._inf_right_last_seed: int | None = None
        self.short_block_stop_guard_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_ENABLED", "0") or "0")
        )
        self.short_block_stop_guard_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.short_block_stop_guard_actions = self._env_action_set(
            "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_ACTIONS",
        ) or {1, 3}
        self.short_block_stop_guard_min_step = self._env_float(
            "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_MIN_STEP",
            0.0,
        )
        self.short_block_stop_guard_max_step = self._env_float(
            "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_MAX_STEP",
            float("inf"),
        )
        self.short_block_stop_guard_max_holds = self._env_float(
            "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_MAX_HOLDS",
            1.0,
        )
        self.short_block_stop_guard_min_distance = self._env_float(
            "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_MIN_DISTANCE",
            0.0,
        )
        self.short_block_stop_guard_max_distance = self._env_float(
            "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_MAX_DISTANCE",
            float("inf"),
        )
        self.short_block_stop_guard_min_slack = self._env_float(
            "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_MIN_SLACK",
            float("-inf"),
        )
        self.short_block_stop_guard_max_slack = self._env_float(
            "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_MAX_SLACK",
            float("inf"),
        )
        self.short_block_stop_guard_min_detour_distance_delta = self._env_float(
            "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_MIN_DETOUR_DISTANCE_DELTA",
            100.0,
        )
        self.short_block_stop_guard_max_forward_distance_delta = self._env_float(
            "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_MAX_FORWARD_DISTANCE_DELTA",
            0.0,
        )
        self.short_block_stop_guard_max_route_distance = self._env_float(
            "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_MAX_ROUTE_DISTANCE",
            float("inf"),
        )
        self.short_block_stop_guard_min_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_MIN_INTERSECTION_ETA_RISK",
            float("-inf"),
        )
        self.short_block_stop_guard_min_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_SHORT_BLOCK_STOP_GUARD_MIN_PREFIX_CONFLICT_COUNT",
            float("-inf"),
        )
        self._short_block_stop_counts: dict[int, int] = {}
        self._short_block_stop_last_step: int | None = None
        self._short_block_stop_last_seed: int | None = None
        self.early_yield_guard_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_EARLY_YIELD_GUARD_ENABLED", "0") or "0")
        )
        self.early_yield_guard_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_EARLY_YIELD_GUARD_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.early_yield_guard_actions = self._env_action_set(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_ACTIONS",
        ) or {2}
        self.early_yield_guard_min_step = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MIN_STEP",
            0.0,
        )
        self.early_yield_guard_max_step = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MAX_STEP",
            float("inf"),
        )
        self.early_yield_guard_max_holds = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MAX_HOLDS",
            1.0,
        )
        self.early_yield_guard_min_departure_lag = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MIN_DEPARTURE_LAG",
            float("-inf"),
        )
        self.early_yield_guard_max_departure_lag = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MAX_DEPARTURE_LAG",
            float("inf"),
        )
        self.early_yield_guard_min_distance = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MIN_DISTANCE",
            0.0,
        )
        self.early_yield_guard_max_distance = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MAX_DISTANCE",
            float("inf"),
        )
        self.early_yield_guard_min_slack = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MIN_SLACK",
            float("-inf"),
        )
        self.early_yield_guard_max_slack = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MAX_SLACK",
            float("inf"),
        )
        self.early_yield_guard_min_active_fraction = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MIN_ACTIVE_FRACTION",
            float("-inf"),
        )
        self.early_yield_guard_max_active_fraction = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MAX_ACTIVE_FRACTION",
            float("inf"),
        )
        self.early_yield_guard_min_stop_proximity = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MIN_STOP_PROXIMITY",
            float("-inf"),
        )
        self.early_yield_guard_max_stop_proximity = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MAX_STOP_PROXIMITY",
            float("inf"),
        )
        self.early_yield_guard_min_route_distance = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MIN_ROUTE_DISTANCE",
            float("-inf"),
        )
        self.early_yield_guard_max_route_count = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MAX_ROUTE_COUNT",
            float("inf"),
        )
        self.early_yield_guard_max_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MAX_PREFIX_CONFLICT_COUNT",
            float("inf"),
        )
        self.early_yield_guard_max_action_distance_delta = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD_MAX_ACTION_DISTANCE_DELTA",
            0.0,
        )
        self.early_yield_guard2_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_EARLY_YIELD_GUARD2_ENABLED", "0") or "0")
        )
        self.early_yield_guard2_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_EARLY_YIELD_GUARD2_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.early_yield_guard2_actions = self._env_action_set(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_ACTIONS",
        ) or {2}
        self.early_yield_guard2_min_step = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MIN_STEP",
            0.0,
        )
        self.early_yield_guard2_max_step = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MAX_STEP",
            float("inf"),
        )
        self.early_yield_guard2_max_holds = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MAX_HOLDS",
            1.0,
        )
        self.early_yield_guard2_min_departure_lag = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MIN_DEPARTURE_LAG",
            float("-inf"),
        )
        self.early_yield_guard2_max_departure_lag = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MAX_DEPARTURE_LAG",
            float("inf"),
        )
        self.early_yield_guard2_min_distance = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MIN_DISTANCE",
            0.0,
        )
        self.early_yield_guard2_max_distance = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MAX_DISTANCE",
            float("inf"),
        )
        self.early_yield_guard2_min_slack = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MIN_SLACK",
            float("-inf"),
        )
        self.early_yield_guard2_max_slack = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MAX_SLACK",
            float("inf"),
        )
        self.early_yield_guard2_min_active_fraction = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MIN_ACTIVE_FRACTION",
            float("-inf"),
        )
        self.early_yield_guard2_max_active_fraction = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MAX_ACTIVE_FRACTION",
            float("inf"),
        )
        self.early_yield_guard2_min_stop_proximity = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MIN_STOP_PROXIMITY",
            float("-inf"),
        )
        self.early_yield_guard2_max_stop_proximity = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MAX_STOP_PROXIMITY",
            float("inf"),
        )
        self.early_yield_guard2_min_route_distance = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MIN_ROUTE_DISTANCE",
            float("-inf"),
        )
        self.early_yield_guard2_max_route_count = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MAX_ROUTE_COUNT",
            float("inf"),
        )
        self.early_yield_guard2_max_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MAX_PREFIX_CONFLICT_COUNT",
            float("inf"),
        )
        self.early_yield_guard2_max_action_distance_delta = self._env_float(
            "ECML_RISK_VETO_EARLY_YIELD_GUARD2_MAX_ACTION_DISTANCE_DELTA",
            0.0,
        )
        self._early_yield_counts: dict[int, int] = {}
        self._early_yield_last_step: int | None = None
        self._early_yield_last_seed: int | None = None
        self.head_on_yield_guard_enabled = bool(
            int(os.environ.get("ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_ENABLED", "0") or "0")
        )
        self.head_on_yield_guard_allowed_scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_ALLOWED_SCENES",
                "",
            ).split(",")
            if scene.strip()
        }
        self.head_on_yield_guard_actions = self._env_action_set(
            "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_ACTIONS",
        ) or {2}
        self.head_on_yield_guard_min_step = self._env_float(
            "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_MIN_STEP",
            138.0,
        )
        self.head_on_yield_guard_max_step = self._env_float(
            "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_MAX_STEP",
            139.0,
        )
        self.head_on_yield_guard_max_holds = self._env_float(
            "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_MAX_HOLDS",
            1.0,
        )
        self.head_on_yield_guard_min_distance = self._env_float(
            "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_MIN_DISTANCE",
            165.0,
        )
        self.head_on_yield_guard_max_distance = self._env_float(
            "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_MAX_DISTANCE",
            175.0,
        )
        self.head_on_yield_guard_min_slack = self._env_float(
            "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_MIN_SLACK",
            125.0,
        )
        self.head_on_yield_guard_max_slack = self._env_float(
            "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_MAX_SLACK",
            130.0,
        )
        self.head_on_yield_guard_max_route_distance = self._env_float(
            "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_MAX_ROUTE_DISTANCE",
            0.15,
        )
        self.head_on_yield_guard_min_intersection_eta_risk = self._env_float(
            "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_MIN_INTERSECTION_ETA_RISK",
            0.90,
        )
        self.head_on_yield_guard_max_intersection_own_distance = self._env_float(
            "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_MAX_INTERSECTION_OWN_DISTANCE",
            0.05,
        )
        self.head_on_yield_guard_min_priority_tighter_fraction = self._env_float(
            "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_MIN_PRIORITY_TIGHTER_FRACTION",
            0.55,
        )
        self.head_on_yield_guard_min_prefix_conflict_count = self._env_float(
            "ECML_RISK_VETO_HEAD_ON_YIELD_GUARD_MIN_PREFIX_CONFLICT_COUNT",
            0.25,
        )
        self._head_on_yield_counts: dict[int, int] = {}
        self._head_on_yield_last_step: int | None = None
        self._head_on_yield_last_seed: int | None = None

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except Exception:
            return default

    @staticmethod
    def _observation_scalar(observation: Any, index: int) -> float | None:
        try:
            values = np.asarray(observation, dtype=np.float32).reshape(-1)
            if index < 0 or index >= values.shape[0]:
                return None
            value = float(values[index])
            if not np.isfinite(value):
                return None
            return value
        except Exception:
            return None

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
    def _env_action_set(name: str) -> set[int]:
        raw_value = os.environ.get(name, "").strip()
        if not raw_value:
            return set()
        actions: set[int] = set()
        for item in raw_value.split(","):
            if not item.strip():
                continue
            try:
                actions.add(int(item))
            except Exception:
                continue
        return actions

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

    def _extra_candidate_rescue_relax(
        self,
        observation: Any,
        baseline_action: int,
        candidate_action: int,
        step: int | None,
        scores: dict[str, Any],
    ) -> bool:
        if not self.extra_relax_enabled:
            return False
        source = str(scores.get("candidate_source", ""))
        if source not in self.extra_relax_allowed_sources:
            return False
        reject_reason = str(scores.get("reject_reason", ""))
        if (
            self.extra_relax_allowed_reject_reasons
            and reject_reason not in self.extra_relax_allowed_reject_reasons
        ):
            return False
        if self.extra_relax_transitions and (
            int(baseline_action),
            int(candidate_action),
        ) not in self.extra_relax_transitions:
            return False
        if step is not None and (
            step < self.extra_relax_min_step or step > self.extra_relax_max_step
        ):
            return False

        if self.reward_risk_policy is not None and (
            "reward_risk_head_candidate" not in scores
        ):
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
                scores["extra_relax_reject_reason"] = "reward_risk_score_error"
                return False

        candidate_risk = float(scores.get("risk_head_candidate", float("inf")))
        risk_delta = float(
            scores.get("risk_head_candidate_minus_baseline", float("inf"))
        )
        if candidate_risk > self.extra_relax_max_candidate_risk:
            scores["extra_relax_reject_reason"] = "candidate_risk_too_high"
            return False
        if risk_delta < self.extra_relax_min_candidate_minus_baseline:
            scores["extra_relax_reject_reason"] = "candidate_risk_delta_too_low"
            return False
        if risk_delta > self.extra_relax_max_candidate_minus_baseline:
            scores["extra_relax_reject_reason"] = "candidate_risk_delta_too_high"
            return False

        reward_risk = float(scores.get("reward_risk_head_candidate", float("inf")))
        reward_delta = float(
            scores.get("reward_risk_head_candidate_minus_baseline", float("inf"))
        )
        reward_baseline_minus_candidate = float(
            scores.get(
                "reward_risk_head_baseline_minus_candidate",
                -reward_delta,
            )
        )
        if reward_risk > self.extra_relax_max_reward_risk:
            scores["extra_relax_reject_reason"] = "reward_risk_too_high"
            return False
        if reward_delta > self.extra_relax_max_reward_candidate_minus_baseline:
            scores["extra_relax_reject_reason"] = "reward_risk_regression"
            return False
        if (
            reward_baseline_minus_candidate
            < self.extra_relax_min_reward_baseline_minus_candidate
        ):
            scores["extra_relax_reject_reason"] = "insufficient_reward_improvement"
            return False

        scores["selector_source"] = "extra_relax"
        scores["extra_relax_accepted"] = 1.0
        scores.pop("reject_reason", None)
        return True

    def _deadline_right_rescue_relax(
        self,
        obs_builder: Any | None,
        handle: int,
        observation: Any,
        baseline_action: int,
        candidate_action: int,
        step: int | None,
        scores: dict[str, Any],
    ) -> bool:
        if not self.deadline_right_relax_enabled:
            return False
        if str(scores.get("candidate_source", "")) != "rerank":
            return False
        if int(baseline_action) != 4 or int(candidate_action) != 3:
            return False
        if str(scores.get("reject_reason", "")) != "reward_risk_too_high":
            return False
        if step is not None and (
            step < self.deadline_right_relax_min_step
            or step > self.deadline_right_relax_max_step
        ):
            return False

        active_fraction = self._observation_scalar(observation, 32)
        on_switch = self._observation_scalar(observation, 7)
        time_slack = self._observation_scalar(observation, 35)
        route_distance = self._observation_scalar(observation, 36)
        route_same_direction = self._observation_scalar(observation, 38)
        route_intersection_eta_risk = self._observation_scalar(observation, 46)
        route_intersection_other_tighter = self._observation_scalar(observation, 50)
        obs_checks = {
            "deadline_right_relax_obs_active_fraction": active_fraction,
            "deadline_right_relax_obs_on_switch": on_switch,
            "deadline_right_relax_obs_time_slack": time_slack,
            "deadline_right_relax_obs_route_distance": route_distance,
            "deadline_right_relax_obs_route_same_direction": route_same_direction,
            "deadline_right_relax_obs_intersection_eta_risk": route_intersection_eta_risk,
            "deadline_right_relax_obs_intersection_other_tighter": (
                route_intersection_other_tighter
            ),
        }
        for key, value in obs_checks.items():
            if value is None:
                scores[f"{key}_missing"] = 1.0
                return False
            scores[key] = float(value)
        if (
            active_fraction < self.deadline_right_relax_min_active_fraction
            or active_fraction > self.deadline_right_relax_max_active_fraction
            or on_switch < 0.5
            or time_slack < self.deadline_right_relax_min_time_slack
            or time_slack > self.deadline_right_relax_max_time_slack
            or route_distance > self.deadline_right_relax_max_route_occupancy_distance
            or route_same_direction < 0.5
            or route_intersection_eta_risk
            < self.deadline_right_relax_min_intersection_eta_risk
            or route_intersection_other_tighter < 0.5
        ):
            return False

        candidate_risk = float(scores.get("risk_head_candidate", float("inf")))
        risk_improvement = float(
            scores.get("risk_head_baseline_minus_candidate", float("-inf"))
        )
        reward_risk = float(scores.get("reward_risk_head_candidate", float("inf")))
        reward_improvement = float(
            scores.get("reward_risk_head_baseline_minus_candidate", float("-inf"))
        )
        if (
            candidate_risk > self.deadline_right_relax_max_candidate_risk
            or risk_improvement < self.deadline_right_relax_min_risk_improvement
            or reward_risk > self.deadline_right_relax_max_reward_risk
            or reward_improvement < self.deadline_right_relax_min_reward_improvement
        ):
            return False

        distance_delta = self._candidate_distance_delta(
            obs_builder,
            handle,
            candidate_action,
        )
        if distance_delta is None:
            return False
        scores["deadline_right_relax_distance_delta"] = float(distance_delta)
        if (
            distance_delta < self.deadline_right_relax_min_distance_delta
            or distance_delta > self.deadline_right_relax_max_distance_delta
        ):
            return False

        scores["selector_source"] = "deadline_right_relax"
        scores["deadline_right_relax_accepted"] = 1.0
        scores.pop("reject_reason", None)
        return True

    def _start_candidate_rescue_relax(
        self,
        observation: Any,
        baseline_action: int,
        candidate_action: int,
        step: int | None,
        scores: dict[str, Any],
    ) -> bool:
        if not self.start_relax_enabled:
            return False
        source = str(scores.get("candidate_source", ""))
        if source not in self.start_relax_allowed_sources:
            return False
        reject_reason = str(scores.get("reject_reason", ""))
        if (
            self.start_relax_allowed_reject_reasons
            and reject_reason not in self.start_relax_allowed_reject_reasons
        ):
            return False
        if self.start_relax_transitions and (
            int(baseline_action),
            int(candidate_action),
        ) not in self.start_relax_transitions:
            return False
        if step is not None and (
            step < self.start_relax_min_step or step > self.start_relax_max_step
        ):
            return False

        if self.reward_risk_policy is not None and (
            "reward_risk_head_candidate" not in scores
        ):
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
                scores["start_relax_reject_reason"] = "reward_risk_score_error"
                return False

        candidate_risk = float(scores.get("risk_head_candidate", float("inf")))
        risk_delta = float(
            scores.get("risk_head_candidate_minus_baseline", float("inf"))
        )
        if candidate_risk > self.start_relax_max_candidate_risk:
            scores["start_relax_reject_reason"] = "candidate_risk_too_high"
            return False
        if risk_delta < self.start_relax_min_candidate_minus_baseline:
            scores["start_relax_reject_reason"] = "candidate_risk_delta_too_low"
            return False
        if risk_delta > self.start_relax_max_candidate_minus_baseline:
            scores["start_relax_reject_reason"] = "candidate_risk_delta_too_high"
            return False

        reward_risk = float(scores.get("reward_risk_head_candidate", float("inf")))
        reward_delta = float(
            scores.get("reward_risk_head_candidate_minus_baseline", float("inf"))
        )
        reward_baseline_minus_candidate = float(
            scores.get(
                "reward_risk_head_baseline_minus_candidate",
                -reward_delta,
            )
        )
        if reward_risk > self.start_relax_max_reward_risk:
            scores["start_relax_reject_reason"] = "reward_risk_too_high"
            return False
        if reward_delta > self.start_relax_max_reward_candidate_minus_baseline:
            scores["start_relax_reject_reason"] = "reward_risk_regression"
            return False
        if (
            reward_baseline_minus_candidate
            < self.start_relax_min_reward_baseline_minus_candidate
        ):
            scores["start_relax_reject_reason"] = "insufficient_reward_improvement"
            return False
        active_fraction = self._observation_scalar(observation, 32)
        if active_fraction is not None:
            scores["start_relax_obs_active_fraction"] = float(active_fraction)
        if (
            active_fraction is not None
            and active_fraction >= self.start_relax_active_fraction_guard_min
            and reward_baseline_minus_candidate
            < self.start_relax_active_fraction_min_reward_improvement
        ):
            scores["start_relax_reject_reason"] = (
                "active_fraction_insufficient_reward_improvement"
            )
            return False

        scores["selector_source"] = "start_relax"
        scores["start_relax_accepted"] = 1.0
        scores.pop("reject_reason", None)
        return True

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
        baseline_action: int,
        candidate_action: int,
        scores: dict[str, Any],
    ) -> bool:
        if scores.get("selector_source") == "deadline_right_relax":
            return False
        action_limit = self.max_action_distance_delta.get(
            int(candidate_action),
            float("inf"),
        )
        source_transition_limit = float("inf")
        if int(baseline_action) == 4 and int(candidate_action) == 3:
            source_transition_limit = self.max_stop_right_distance_delta
            if scores.get("candidate_source") == "raw_topn":
                source_transition_limit = min(
                    source_transition_limit,
                    self.max_raw_topn_stop_right_distance_delta,
                )
        max_distance_delta = min(self.max_candidate_distance_delta, action_limit)
        max_distance_delta = min(max_distance_delta, source_transition_limit)
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

    def _start_relax_active_distance_veto(
        self,
        obs_builder: Any | None,
        handle: int,
        candidate_action: int,
        scores: dict[str, Any],
    ) -> bool:
        if scores.get("selector_source") != "start_relax":
            return False
        active_fraction = scores.get("start_relax_obs_active_fraction")
        try:
            active_fraction = float(active_fraction)
        except Exception:
            return False
        if active_fraction < self.start_relax_active_distance_guard_min:
            return False

        distance_delta = self._candidate_distance_delta(
            obs_builder,
            handle,
            candidate_action,
        )
        if distance_delta is None:
            return False
        scores["start_relax_active_distance_delta"] = float(distance_delta)
        if distance_delta > self.start_relax_active_distance_max_delta:
            return False

        reward_improvement = float(
            scores.get(
                "reward_risk_head_baseline_minus_candidate",
                float("-inf"),
            )
        )
        if (
            reward_improvement
            >= self.start_relax_active_distance_min_reward_improvement
        ):
            return False
        scores["reject_reason"] = "start_relax_active_distance_reward_guard"
        return True

    def _start_relax_opposing_route_veto(
        self,
        observation: Any,
        scores: dict[str, Any],
    ) -> bool:
        if scores.get("selector_source") != "start_relax":
            return False
        route_distance = self._observation_scalar(observation, 36)
        route_opposing = self._observation_scalar(observation, 37)
        active_fraction = self._observation_scalar(observation, 32)
        if (
            route_distance is None
            or route_opposing is None
            or active_fraction is None
        ):
            return False
        scores["start_relax_route_occupancy_distance"] = float(route_distance)
        scores["start_relax_route_occupancy_opposing"] = float(route_opposing)
        risk_delta = float(
            scores.get("risk_head_candidate_minus_baseline", float("-inf"))
        )
        reward_improvement = float(
            scores.get(
                "reward_risk_head_baseline_minus_candidate",
                float("inf"),
            )
        )
        if route_opposing < 0.5:
            return False
        if route_distance > self.start_relax_opposing_route_max_distance:
            return False
        if active_fraction < self.start_relax_opposing_route_min_active_fraction:
            return False
        if active_fraction > self.start_relax_opposing_route_max_active_fraction:
            return False
        if risk_delta < self.start_relax_opposing_route_min_candidate_minus_baseline:
            return False
        if reward_improvement > self.start_relax_opposing_route_max_reward_improvement:
            return False
        scores["reject_reason"] = "start_relax_opposing_route_reward_guard"
        return True

    @staticmethod
    def _prefix_edges(
        prefix: list[dict[str, Any]],
    ) -> dict[tuple[tuple[int, int], tuple[int, int]], dict[str, Any]]:
        edges = {}
        for node in prefix:
            previous = node.get("prev_position")
            position = node.get("position")
            if previous is None or position is None or previous == position:
                continue
            edges[(previous, position)] = node
        return edges

    @staticmethod
    def _agent_eta(obs_builder: Any, handle: int, step: int) -> float:
        try:
            speed = float(obs_builder.env.agents[handle].speed_counter.speed)
        except Exception:
            speed = 1.0
        if speed <= 0.0:
            speed = 1.0
        return float(step) / speed

    def _candidate_prefix_summary(
        self,
        obs_builder: Any | None,
        planned_prefixes: dict[int, list[dict[str, Any]]] | None,
        handle: int,
        candidate_action: int,
    ) -> dict[str, float] | None:
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

        candidate_positions: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for node in candidate_prefix:
            position = node.get("position")
            if position is not None:
                candidate_positions.setdefault(position, []).append(node)
        if not candidate_positions:
            return {
                "cell_intersections": 0.0,
                "same_edge_conflicts": 0.0,
                "min_intersection_eta_gap": 999.0,
            }

        cell_intersections = 0
        same_edge_conflicts = 0
        min_intersection_eta_gap = float("inf")
        candidate_edges = self._prefix_edges(candidate_prefix)
        for other, other_prefix in planned_prefixes.items():
            if other == handle:
                continue
            other_positions: dict[tuple[int, int], list[dict[str, Any]]] = {}
            for node in other_prefix:
                position = node.get("position")
                if position is not None:
                    other_positions.setdefault(position, []).append(node)
            for position, own_nodes in candidate_positions.items():
                for other_node in other_positions.get(position, []):
                    for own_node in own_nodes:
                        cell_intersections += 1
                        eta_gap = abs(
                            self._agent_eta(
                                obs_builder,
                                handle,
                                int(own_node.get("step", 0)),
                            )
                            - self._agent_eta(
                                obs_builder,
                                other,
                                int(other_node.get("step", 0)),
                            )
                        )
                        min_intersection_eta_gap = min(
                            min_intersection_eta_gap,
                            eta_gap,
                        )
            other_edges = self._prefix_edges(other_prefix)
            for source, target in candidate_edges:
                if (source, target) in other_edges:
                    same_edge_conflicts += 1
        if not np.isfinite(min_intersection_eta_gap):
            min_intersection_eta_gap = 999.0
        return {
            "cell_intersections": float(cell_intersections),
            "same_edge_conflicts": float(same_edge_conflicts),
            "min_intersection_eta_gap": float(min_intersection_eta_gap),
        }

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
        prefix_summary = self._candidate_prefix_summary(
            obs_builder,
            planned_prefixes,
            handle,
            candidate_action,
        )
        if prefix_summary is None:
            return False
        prefix_intersections = prefix_summary["cell_intersections"]
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
        scores["candidate_prefix_same_edge_conflicts"] = float(
            prefix_summary["same_edge_conflicts"]
        )
        scores["candidate_prefix_min_intersection_eta_gap"] = float(
            prefix_summary["min_intersection_eta_gap"]
        )
        scores["candidate_deadline_conflict_penalty"] = float(deadline_penalty)
        if (
            np.isfinite(self.stop_left_same_edge_min_conflicts)
            and np.isfinite(self.stop_left_same_edge_max_eta_gap)
            and prefix_summary["same_edge_conflicts"]
            >= self.stop_left_same_edge_min_conflicts
            and prefix_summary["min_intersection_eta_gap"]
            <= self.stop_left_same_edge_max_eta_gap
        ):
            scores["stop_left_same_edge_min_conflicts"] = float(
                self.stop_left_same_edge_min_conflicts
            )
            scores["stop_left_same_edge_max_eta_gap"] = float(
                self.stop_left_same_edge_max_eta_gap
            )
            if self.stop_left_same_edge_relax_enabled:
                candidate_risk = float(scores.get("risk_head_candidate", float("inf")))
                risk_improvement = float(
                    scores.get("risk_head_baseline_minus_candidate", float("-inf"))
                )
                reward_risk = float(
                    scores.get("reward_risk_head_candidate", float("inf"))
                )
                reward_improvement = float(
                    scores.get(
                        "reward_risk_head_baseline_minus_candidate",
                        float("-inf"),
                    )
                )
                if (
                    candidate_risk
                    <= self.stop_left_same_edge_relax_max_candidate_risk
                    and reward_risk
                    <= self.stop_left_same_edge_relax_max_reward_risk
                    and risk_improvement
                    >= self.stop_left_same_edge_relax_min_risk_improvement
                    and reward_improvement
                    >= self.stop_left_same_edge_relax_min_reward_improvement
                ):
                    scores["stop_left_same_edge_relax_accepted"] = 1.0
                    return False
            scores["reject_reason"] = "stop_left_same_edge_eta_gap_too_tight"
            return True
        if prefix_intersections <= 0.0 and deadline_penalty <= 0.0:
            if np.isfinite(self.max_unconflicted_stop_left_distance_delta):
                distance_delta = self._candidate_distance_delta(
                    obs_builder,
                    handle,
                    candidate_action,
                )
                if distance_delta is not None:
                    scores["unconflicted_stop_left_distance_delta"] = float(
                        distance_delta
                    )
                    scores["max_unconflicted_stop_left_distance_delta"] = float(
                        self.max_unconflicted_stop_left_distance_delta
                    )
                    if (
                        distance_delta
                        > self.max_unconflicted_stop_left_distance_delta
                    ):
                        scores["reject_reason"] = (
                            "unconflicted_stop_left_distance_delta_too_high"
                        )
                        return True
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

    def _trace_start_delay_guard(
        self,
        *,
        handle: int,
        seed: int | None,
        step: int | None,
        previous_action: int,
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
            "baseline_action": int(previous_action),
            "baseline_action_name": self._action_name(int(previous_action)),
            "candidate_action": 0,
            "candidate_action_name": self._action_name(0),
            "accepted": True,
            "candidate_source": "start_delay_guard",
            **scores,
        }
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle_out:
            handle_out.write(json.dumps(row, sort_keys=True) + "\n")

    def _trace_yield_stop_guard(
        self,
        *,
        handle: int,
        seed: int | None,
        step: int | None,
        previous_action: int,
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
            "baseline_action": int(previous_action),
            "baseline_action_name": self._action_name(int(previous_action)),
            "candidate_action": 4,
            "candidate_action_name": self._action_name(4),
            "accepted": True,
            "candidate_source": "yield_stop_guard",
            **scores,
        }
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle_out:
            handle_out.write(json.dumps(row, sort_keys=True) + "\n")

    def _trace_head_on_yield_guard(
        self,
        *,
        handle: int,
        seed: int | None,
        step: int | None,
        previous_action: int,
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
            "baseline_action": int(previous_action),
            "baseline_action_name": self._action_name(int(previous_action)),
            "candidate_action": 4,
            "candidate_action_name": self._action_name(4),
            "accepted": True,
            "candidate_source": "head_on_yield_guard",
            **scores,
        }
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle_out:
            handle_out.write(json.dumps(row, sort_keys=True) + "\n")

    def _trace_detour_left_guard(
        self,
        *,
        handle: int,
        seed: int | None,
        step: int | None,
        previous_action: int,
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
            "baseline_action": int(previous_action),
            "baseline_action_name": self._action_name(int(previous_action)),
            "candidate_action": 1,
            "candidate_action_name": self._action_name(1),
            "accepted": True,
            "candidate_source": "detour_left_guard",
            **scores,
        }
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle_out:
            handle_out.write(json.dumps(row, sort_keys=True) + "\n")

    def _trace_switch_escape_guard(
        self,
        *,
        handle: int,
        seed: int | None,
        step: int | None,
        previous_action: int,
        candidate_action: int,
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
            "baseline_action": int(previous_action),
            "baseline_action_name": self._action_name(int(previous_action)),
            "candidate_action": int(candidate_action),
            "candidate_action_name": self._action_name(int(candidate_action)),
            "accepted": True,
            "candidate_source": "switch_escape_guard",
            **scores,
        }
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle_out:
            handle_out.write(json.dumps(row, sort_keys=True) + "\n")

    def _trace_switch_forward_guard(
        self,
        *,
        handle: int,
        seed: int | None,
        step: int | None,
        previous_action: int,
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
            "baseline_action": int(previous_action),
            "baseline_action_name": self._action_name(int(previous_action)),
            "candidate_action": 2,
            "candidate_action_name": self._action_name(2),
            "accepted": True,
            "candidate_source": "switch_forward_guard",
            **scores,
        }
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle_out:
            handle_out.write(json.dumps(row, sort_keys=True) + "\n")

    def _trace_inf_right_guard(
        self,
        *,
        handle: int,
        seed: int | None,
        step: int | None,
        previous_action: int,
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
            "baseline_action": int(previous_action),
            "baseline_action_name": self._action_name(int(previous_action)),
            "candidate_action": 3,
            "candidate_action_name": self._action_name(3),
            "accepted": True,
            "candidate_source": "inf_right_guard",
            **scores,
        }
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle_out:
            handle_out.write(json.dumps(row, sort_keys=True) + "\n")

    def _trace_short_block_stop_guard(
        self,
        *,
        handle: int,
        seed: int | None,
        step: int | None,
        previous_action: int,
        candidate_action: int,
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
            "baseline_action": int(previous_action),
            "baseline_action_name": self._action_name(int(previous_action)),
            "candidate_action": int(candidate_action),
            "candidate_action_name": self._action_name(int(candidate_action)),
            "accepted": True,
            "candidate_source": "short_block_stop_guard",
            **scores,
        }
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle_out:
            handle_out.write(json.dumps(row, sort_keys=True) + "\n")

    def _trace_early_yield_guard(
        self,
        *,
        handle: int,
        seed: int | None,
        step: int | None,
        previous_action: int,
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
            "baseline_action": int(previous_action),
            "baseline_action_name": self._action_name(int(previous_action)),
            "candidate_action": 4,
            "candidate_action_name": self._action_name(4),
            "accepted": True,
            "candidate_source": "early_yield_guard",
            **scores,
        }
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle_out:
            handle_out.write(json.dumps(row, sort_keys=True) + "\n")

    def _reset_start_delay_guard_state(
        self,
        seed: int | None,
        step: int | None,
    ) -> None:
        if step is None:
            return
        if (
            self._start_delay_last_seed != seed
            or self._start_delay_last_step is None
            or step < self._start_delay_last_step
        ):
            self._start_delay_counts = {}
        self._start_delay_last_seed = seed
        self._start_delay_last_step = step

    def _reset_yield_stop_guard_state(
        self,
        seed: int | None,
        step: int | None,
    ) -> None:
        if step is None:
            return
        if (
            self._yield_stop_last_seed != seed
            or self._yield_stop_last_step is None
            or step < self._yield_stop_last_step
        ):
            self._yield_stop_counts = {}
        self._yield_stop_last_seed = seed
        self._yield_stop_last_step = step

    def _reset_head_on_yield_guard_state(
        self,
        seed: int | None,
        step: int | None,
    ) -> None:
        if step is None:
            return
        if (
            self._head_on_yield_last_seed != seed
            or self._head_on_yield_last_step is None
            or step < self._head_on_yield_last_step
        ):
            self._head_on_yield_counts = {}
        self._head_on_yield_last_seed = seed
        self._head_on_yield_last_step = step

    def _reset_detour_left_guard_state(
        self,
        seed: int | None,
        step: int | None,
    ) -> None:
        if step is None:
            return
        if (
            self._detour_left_last_seed != seed
            or self._detour_left_last_step is None
            or step < self._detour_left_last_step
        ):
            self._detour_left_counts = {}
        self._detour_left_last_seed = seed
        self._detour_left_last_step = step

    def _reset_switch_escape_guard_state(
        self,
        seed: int | None,
        step: int | None,
    ) -> None:
        if step is None:
            return
        if (
            self._switch_escape_last_seed != seed
            or self._switch_escape_last_step is None
            or step < self._switch_escape_last_step
        ):
            self._switch_escape_counts = {}
        self._switch_escape_last_seed = seed
        self._switch_escape_last_step = step

    def _reset_switch_forward_guard_state(
        self,
        seed: int | None,
        step: int | None,
    ) -> None:
        if step is None:
            return
        if (
            self._switch_forward_last_seed != seed
            or self._switch_forward_last_step is None
            or step < self._switch_forward_last_step
        ):
            self._switch_forward_counts = {}
        self._switch_forward_last_seed = seed
        self._switch_forward_last_step = step

    def _reset_right_forward_guard_state(
        self,
        seed: int | None,
        step: int | None,
    ) -> None:
        if step is None:
            return
        if (
            self._right_forward_last_seed != seed
            or self._right_forward_last_step is None
            or step < self._right_forward_last_step
        ):
            self._right_forward_counts = {}
        self._right_forward_last_seed = seed
        self._right_forward_last_step = step

    def _reset_inf_right_guard_state(
        self,
        seed: int | None,
        step: int | None,
    ) -> None:
        if step is None:
            return
        if (
            self._inf_right_last_seed != seed
            or self._inf_right_last_step is None
            or step < self._inf_right_last_step
        ):
            self._inf_right_counts = {}
        self._inf_right_last_seed = seed
        self._inf_right_last_step = step

    def _reset_short_block_stop_guard_state(
        self,
        seed: int | None,
        step: int | None,
    ) -> None:
        if step is None:
            return
        if (
            self._short_block_stop_last_seed != seed
            or self._short_block_stop_last_step is None
            or step < self._short_block_stop_last_step
        ):
            self._short_block_stop_counts = {}
        self._short_block_stop_last_seed = seed
        self._short_block_stop_last_step = step

    def _reset_early_yield_guard_state(
        self,
        seed: int | None,
        step: int | None,
    ) -> None:
        if step is None:
            return
        if (
            self._early_yield_last_seed != seed
            or self._early_yield_last_step is None
            or step < self._early_yield_last_step
        ):
            self._early_yield_counts = {}
        self._early_yield_last_seed = seed
        self._early_yield_last_step = step

    @staticmethod
    def _guard_speed(agent: Any) -> float:
        try:
            speed = float(agent.speed_counter.speed)
            return speed if speed > 0.0 else 1.0
        except Exception:
            return 1.0

    def _guard_route_prefix(
        self,
        obs_builder: Any,
        handle: int,
        first_action: int | None,
        max_cells: int,
    ) -> list[dict[str, Any]]:
        try:
            agent = obs_builder.env.agents[handle]
            if obs_builder._state_matches(agent.state, "DONE", "DONE_REMOVED"):
                return []
            if first_action in (1, 2, 3):
                target_position, target_direction = obs_builder._action_target(
                    handle,
                    first_action,
                )
                if target_position is None or target_direction is None:
                    return []
                prefix = [
                    {
                        "step": 1,
                        "prev_position": agent.position,
                        "position": target_position,
                        "direction": target_direction,
                    }
                ]
                current_position = target_position
                current_direction = target_direction
                next_step = 2
            else:
                current_position, current_direction = obs_builder._agent_route_start(handle)
                if current_position is None and getattr(agent, "initial_position", None) is not None:
                    current_position = agent.initial_position
                    current_direction = agent.initial_direction
                if current_position is None or current_direction is None:
                    return []
                prefix = [
                    {
                        "step": 0,
                        "prev_position": None,
                        "position": current_position,
                        "direction": current_direction,
                    }
                ]
                next_step = 1

            if not obs_builder._is_in_bounds(current_position):
                return []
            distance_map = obs_builder._get_distance_map(handle)
            seen = {(current_position, current_direction)}
            for step_index in range(next_step, max_cells + 1):
                transitions = obs_builder.env.rail.get_transitions(
                    (current_position, current_direction)
                )
                next_direction = obs_builder._best_progress_direction(
                    transitions,
                    current_position,
                    distance_map,
                )
                if next_direction is None:
                    break
                next_position = get_new_position(current_position, next_direction)
                if not obs_builder._is_in_bounds(next_position):
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
        except Exception:
            return []

    @staticmethod
    def _guard_departure_offset(obs_builder: Any, handle: int) -> float:
        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "WAITING"):
                return 0.0
            earliest_departure = getattr(agent, "earliest_departure", None)
            if earliest_departure is None:
                return 0.0
            return max(0.0, float(earliest_departure) - float(obs_builder.env._elapsed_steps))
        except Exception:
            return 0.0

    def _guard_effective_slack(
        self,
        obs_builder: Any,
        handle: int,
        distance: float,
    ) -> float:
        try:
            slack = float(obs_builder._deadline_slack(handle, distance))
            return slack - self._guard_departure_offset(obs_builder, handle)
        except Exception:
            return float("inf")

    def _guard_route_priority_conflicts(
        self,
        obs_builder: Any,
        handle: int,
        own_prefix: list[dict[str, Any]],
        own_slack: float,
        max_cells: int,
    ) -> tuple[int, dict[str, Any]]:
        try:
            own_agent = obs_builder.env.agents[handle]
        except Exception:
            return 0, {}
        own_speed = self._guard_speed(own_agent)
        own_positions: dict[tuple[int, int], float] = {}
        own_edges: dict[tuple[tuple[int, int] | None, tuple[int, int]], float] = {}
        for node in own_prefix:
            try:
                eta = float(node["step"]) / own_speed
                position = node["position"]
                prev_position = node.get("prev_position")
            except Exception:
                continue
            own_positions.setdefault(position, eta)
            own_edges.setdefault((prev_position, position), eta)

        conflicts = 0
        min_other_slack = float("inf")
        min_eta_gap = float("inf")
        first_other = None
        env = obs_builder.env
        for other_handle, other_agent in enumerate(env.agents):
            if other_handle == handle:
                continue
            try:
                if (
                    self._start_delay_counts.get(other_handle, 0) > 0
                    and obs_builder._state_matches(other_agent.state, "READY_TO_DEPART")
                ):
                    continue
                if obs_builder._state_matches(
                    other_agent.state,
                    "DONE",
                    "DONE_REMOVED",
                ):
                    continue
                other_distance = float(obs_builder._current_distance_to_waypoint(other_handle))
                other_slack = self._guard_effective_slack(
                    obs_builder,
                    other_handle,
                    other_distance,
                )
            except Exception:
                continue
            if not np.isfinite(other_slack):
                continue
            if (
                own_slack - other_slack
                < self.start_delay_guard_min_other_slack_advantage
            ):
                continue
            other_prefix = self._guard_route_prefix(
                obs_builder,
                other_handle,
                None,
                max_cells,
            )
            if not other_prefix:
                continue
            other_speed = self._guard_speed(other_agent)
            other_offset = self._guard_departure_offset(obs_builder, other_handle)
            has_conflict = False
            best_gap = float("inf")
            for node in other_prefix:
                try:
                    other_eta = other_offset + float(node["step"]) / other_speed
                    position = node["position"]
                    prev_position = node.get("prev_position")
                except Exception:
                    continue
                own_eta = own_positions.get(position)
                if own_eta is not None:
                    gap = abs(own_eta - other_eta)
                    if gap <= self.start_delay_guard_eta_window:
                        has_conflict = True
                        best_gap = min(best_gap, gap)
                reverse_eta = own_edges.get((position, prev_position))
                if reverse_eta is not None:
                    gap = abs(reverse_eta - other_eta)
                    if gap <= self.start_delay_guard_eta_window:
                        has_conflict = True
                        best_gap = min(best_gap, gap)
                if has_conflict and best_gap <= 0.0:
                    break
            if not has_conflict:
                continue
            conflicts += 1
            if other_slack < min_other_slack:
                min_other_slack = other_slack
                first_other = other_handle
            min_eta_gap = min(min_eta_gap, best_gap)

        return conflicts, {
            "start_delay_guard_priority_conflicts": int(conflicts),
            "start_delay_guard_min_other_slack": (
                float(min_other_slack) if np.isfinite(min_other_slack) else None
            ),
            "start_delay_guard_min_eta_gap": (
                float(min_eta_gap) if np.isfinite(min_eta_gap) else None
            ),
            "start_delay_guard_first_other": first_other,
        }

    def _start_delay_guard_scores(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        scores: dict[str, Any] = {}
        if not self.start_delay_guard_enabled:
            return False, scores
        if obs_builder is None or step is None:
            return False, scores
        if self.start_delay_guard_allowed_scenes:
            try:
                scene = runtime_context.get().scene
            except Exception:
                scene = None
            if scene not in self.start_delay_guard_allowed_scenes:
                return False, scores
        if step < self.start_delay_guard_min_step or step > self.start_delay_guard_max_step:
            return False, scores
        if action not in (1, 2, 3):
            return False, scores
        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "READY_TO_DEPART"):
                return False, scores
        except Exception:
            return False, scores
        holds = self._start_delay_counts.get(handle, 0)
        if holds >= self.start_delay_guard_max_holds:
            return False, scores
        try:
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = self._guard_effective_slack(obs_builder, handle, distance)
        except Exception:
            return False, scores
        scores.update(
            {
                "start_delay_guard_distance": float(distance),
                "start_delay_guard_slack": float(slack),
                "start_delay_guard_holds": int(holds),
            }
        )
        if observation is not None and self.risk_policy is not None:
            try:
                scores.update(
                    self._action_risk_scores(
                        self.risk_policy,
                        observation,
                        action,
                        0,
                        "start_delay_risk_head",
                    )
                )
            except Exception:
                scores["start_delay_risk_score_error"] = 1.0
        if observation is not None and self.reward_risk_policy is not None:
            try:
                scores.update(
                    self._action_risk_scores(
                        self.reward_risk_policy,
                        observation,
                        action,
                        0,
                        "start_delay_reward_risk_head",
                    )
                )
            except Exception:
                scores["start_delay_reward_risk_score_error"] = 1.0
        if observation is not None and self.value_policy is not None:
            try:
                with torch.no_grad():
                    values = self.value_policy.action_value_scores(
                        np.asarray(observation, dtype=np.float32)
                    ).squeeze(0).cpu().numpy()
                move_value = float(values[action])
                delay_value = float(values[0])
                scores.update(
                    {
                        "start_delay_value_head_move": move_value,
                        "start_delay_value_head_delay": delay_value,
                        "start_delay_value_head_delay_minus_move": delay_value
                        - move_value,
                    }
                )
            except Exception:
                scores["start_delay_value_score_error"] = 1.0
        if not np.isfinite(distance) or distance < self.start_delay_guard_min_distance:
            return False, scores
        if (
            not np.isfinite(slack)
            or slack < self.start_delay_guard_min_slack
            or slack > self.start_delay_guard_max_slack
        ):
            return False, scores

        own_prefix = self._guard_route_prefix(
            obs_builder,
            handle,
            action,
            int(self.start_delay_guard_lookahead),
        )
        if not own_prefix:
            return False, scores
        conflicts, conflict_scores = self._guard_route_priority_conflicts(
            obs_builder,
            handle,
            own_prefix,
            slack,
            int(self.start_delay_guard_lookahead),
        )
        scores.update(conflict_scores)
        return conflicts >= self.start_delay_guard_min_priority_conflicts, scores

    def _apply_start_delay_guard(
        self,
        output: dict[int, RailEnvActions],
        handles: List[int],
        obs_builder: Any,
        observations_by_handle: dict[int, Any],
        seed: int | None,
        step: int | None,
    ) -> dict[int, RailEnvActions]:
        if not self.start_delay_guard_enabled or obs_builder is None:
            return output
        self._reset_start_delay_guard_state(seed, step)
        adjusted = dict(output)
        for handle in handles:
            if handle not in adjusted:
                continue
            previous_action = self._action_id(adjusted[handle])
            accepted, scores = self._start_delay_guard_scores(
                obs_builder,
                int(handle),
                previous_action,
                step,
                observations_by_handle.get(int(handle)),
            )
            if not accepted:
                continue
            adjusted[handle] = RailEnvActions.DO_NOTHING
            self._start_delay_counts[int(handle)] = (
                self._start_delay_counts.get(int(handle), 0) + 1
            )
            self._trace_start_delay_guard(
                handle=int(handle),
                seed=seed,
                step=step,
                previous_action=previous_action,
                scores=scores,
            )
        return adjusted

    def _yield_stop_guard_scores(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        scores: dict[str, Any] = {}
        if not self.yield_stop_guard_enabled:
            return False, scores
        if obs_builder is None or step is None:
            return False, scores
        if step < self.yield_stop_guard_min_step or step > self.yield_stop_guard_max_step:
            return False, scores
        if (int(action), 4) not in self.yield_stop_guard_transitions:
            return False, scores
        holds = self._yield_stop_counts.get(handle, 0)
        if holds >= self.yield_stop_guard_max_holds:
            return False, scores
        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "MOVING"):
                return False, scores
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = self._guard_effective_slack(obs_builder, handle, distance)
        except Exception:
            return False, scores
        tighter_agents = 0
        try:
            for other_handle in obs_builder.env.get_agent_handles():
                if other_handle == handle:
                    continue
                other_distance = float(
                    obs_builder._current_distance_to_waypoint(other_handle)
                )
                other_slack = self._guard_effective_slack(
                    obs_builder,
                    other_handle,
                    other_distance,
                )
                if np.isfinite(other_slack) and other_slack < slack:
                    tighter_agents += 1
        except Exception:
            tighter_agents = 0

        route_occupancy_count = (
            self._observation_scalar(observation, 43)
            if observation is not None
            else None
        )
        intersection_eta_risk = (
            self._observation_scalar(observation, 46)
            if observation is not None
            else None
        )
        scores.update(
            {
                "yield_stop_guard_distance": float(distance),
                "yield_stop_guard_slack": float(slack),
                "yield_stop_guard_tighter_agents": int(tighter_agents),
                "yield_stop_guard_holds": int(holds),
            }
        )
        if route_occupancy_count is not None:
            scores["yield_stop_guard_route_occupancy_count"] = float(
                route_occupancy_count
            )
        if intersection_eta_risk is not None:
            scores["yield_stop_guard_intersection_eta_risk"] = float(
                intersection_eta_risk
            )

        if (
            not np.isfinite(distance)
            or distance < self.yield_stop_guard_min_distance
            or distance > self.yield_stop_guard_max_distance
        ):
            return False, scores
        if (
            not np.isfinite(slack)
            or slack < self.yield_stop_guard_min_slack
            or slack > self.yield_stop_guard_max_slack
        ):
            return False, scores
        if tighter_agents < self.yield_stop_guard_min_tighter_agents:
            return False, scores
        if (
            route_occupancy_count is not None
            and route_occupancy_count
            > self.yield_stop_guard_max_route_occupancy_count
        ):
            return False, scores
        if (
            intersection_eta_risk is not None
            and intersection_eta_risk
            > self.yield_stop_guard_max_intersection_eta_risk
        ):
            return False, scores
        return True, scores

    def _yield_stop_guard2_scores(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        scores: dict[str, Any] = {}
        if not self.yield_stop_guard2_enabled:
            return False, scores
        if obs_builder is None or step is None or observation is None:
            return False, scores
        if step < self.yield_stop_guard2_min_step or step > self.yield_stop_guard2_max_step:
            return False, scores
        try:
            scene = runtime_context.get().scene
        except Exception:
            scene = None
        if self.yield_stop_guard2_allowed_scenes:
            if scene not in self.yield_stop_guard2_allowed_scenes:
                return False, scores
        if (int(action), 4) not in self.yield_stop_guard2_transitions:
            return False, scores

        holds = self._yield_stop_counts.get(handle, 0)
        if holds >= self.yield_stop_guard2_max_holds:
            return False, scores

        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "MOVING"):
                return False, scores
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = self._guard_effective_slack(obs_builder, handle, distance)
        except Exception:
            return False, scores

        tighter_agents = 0
        try:
            for other_handle in obs_builder.env.get_agent_handles():
                if other_handle == handle:
                    continue
                other_distance = float(
                    obs_builder._current_distance_to_waypoint(other_handle)
                )
                other_slack = self._guard_effective_slack(
                    obs_builder,
                    other_handle,
                    other_distance,
                )
                if np.isfinite(other_slack) and other_slack < slack:
                    tighter_agents += 1
        except Exception:
            tighter_agents = 0

        active_fraction = self._observation_scalar(observation, 32)
        stop_proximity = self._observation_scalar(observation, 33)
        route_distance = self._observation_scalar(observation, 36)
        route_opposing = self._observation_scalar(observation, 37)
        route_same_direction = self._observation_scalar(observation, 38)
        route_other_tighter = self._observation_scalar(observation, 40)
        route_count = self._observation_scalar(observation, 43)
        intersection_own_distance = self._observation_scalar(observation, 44)
        intersection_eta_risk = self._observation_scalar(observation, 46)
        intersection_other_first = self._observation_scalar(observation, 47)
        intersection_other_tighter = self._observation_scalar(observation, 50)
        prefix_conflict_count = self._observation_scalar(observation, 51)
        obs_checks = {
            "yield_stop_guard2_active_fraction": active_fraction,
            "yield_stop_guard2_stop_proximity": stop_proximity,
            "yield_stop_guard2_route_distance": route_distance,
            "yield_stop_guard2_route_opposing": route_opposing,
            "yield_stop_guard2_route_same_direction": route_same_direction,
            "yield_stop_guard2_route_other_tighter": route_other_tighter,
            "yield_stop_guard2_route_count": route_count,
            "yield_stop_guard2_intersection_own_distance": (
                intersection_own_distance
            ),
            "yield_stop_guard2_intersection_eta_risk": intersection_eta_risk,
            "yield_stop_guard2_intersection_other_first": (
                intersection_other_first
            ),
            "yield_stop_guard2_intersection_other_tighter": (
                intersection_other_tighter
            ),
            "yield_stop_guard2_prefix_conflict_count": prefix_conflict_count,
        }
        for key, value in obs_checks.items():
            if value is None:
                return False, scores
            scores[key] = float(value)
        scores.update(
            {
                "yield_stop_guard_profile": "scene3_conflict_stop",
                "yield_stop_guard2_distance": float(distance),
                "yield_stop_guard2_slack": float(slack),
                "yield_stop_guard2_tighter_agents": int(tighter_agents),
                "yield_stop_guard2_holds": int(holds),
            }
        )

        if (
            not np.isfinite(distance)
            or distance < self.yield_stop_guard2_min_distance
            or distance > self.yield_stop_guard2_max_distance
            or not np.isfinite(slack)
            or slack < self.yield_stop_guard2_min_slack
            or slack > self.yield_stop_guard2_max_slack
            or tighter_agents < self.yield_stop_guard2_min_tighter_agents
            or active_fraction < self.yield_stop_guard2_min_active_fraction
            or active_fraction > self.yield_stop_guard2_max_active_fraction
            or stop_proximity < self.yield_stop_guard2_min_stop_proximity
            or stop_proximity > self.yield_stop_guard2_max_stop_proximity
            or route_distance < self.yield_stop_guard2_min_route_distance
            or route_distance > self.yield_stop_guard2_max_route_distance
            or route_opposing < self.yield_stop_guard2_min_route_opposing
            or route_same_direction
            > self.yield_stop_guard2_max_route_same_direction
            or route_other_tighter > self.yield_stop_guard2_max_route_other_tighter
            or route_count < self.yield_stop_guard2_min_route_count
            or route_count > self.yield_stop_guard2_max_route_count
            or intersection_own_distance
            < self.yield_stop_guard2_min_intersection_own_distance
            or intersection_own_distance
            > self.yield_stop_guard2_max_intersection_own_distance
            or intersection_eta_risk
            < self.yield_stop_guard2_min_intersection_eta_risk
            or intersection_eta_risk
            > self.yield_stop_guard2_max_intersection_eta_risk
            or intersection_other_first
            > self.yield_stop_guard2_max_intersection_other_first
            or intersection_other_tighter
            > self.yield_stop_guard2_max_intersection_other_tighter
            or prefix_conflict_count
            < self.yield_stop_guard2_min_prefix_conflict_count
            or prefix_conflict_count
            > self.yield_stop_guard2_max_prefix_conflict_count
        ):
            return False, scores
        return True, scores

    def _yield_stop_guard3_scores(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        scores: dict[str, Any] = {}
        if not self.yield_stop_guard3_enabled:
            return False, scores
        if obs_builder is None or step is None or observation is None:
            return False, scores
        if step < self.yield_stop_guard3_min_step or step > self.yield_stop_guard3_max_step:
            return False, scores
        try:
            scene = runtime_context.get().scene
        except Exception:
            scene = None
        if self.yield_stop_guard3_allowed_scenes and (
            scene not in self.yield_stop_guard3_allowed_scenes
        ):
            return False, scores
        if (int(action), 4) not in self.yield_stop_guard3_transitions:
            return False, scores

        holds = self._yield_stop_counts.get(handle, 0)
        if holds >= self.yield_stop_guard3_max_holds:
            return False, scores

        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "MOVING"):
                return False, scores
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = self._guard_effective_slack(obs_builder, handle, distance)
        except Exception:
            return False, scores

        tighter_agents = 0
        try:
            for other_handle in obs_builder.env.get_agent_handles():
                if other_handle == handle:
                    continue
                other_distance = float(
                    obs_builder._current_distance_to_waypoint(other_handle)
                )
                other_slack = self._guard_effective_slack(
                    obs_builder,
                    other_handle,
                    other_distance,
                )
                if np.isfinite(other_slack) and other_slack < slack:
                    tighter_agents += 1
        except Exception:
            tighter_agents = 0

        on_switch = self._observation_scalar(observation, 7)
        active_fraction = self._observation_scalar(observation, 32)
        stop_proximity = self._observation_scalar(observation, 33)
        route_distance = self._observation_scalar(observation, 36)
        route_opposing = self._observation_scalar(observation, 37)
        route_same_direction = self._observation_scalar(observation, 38)
        route_other_tighter = self._observation_scalar(observation, 40)
        route_count = self._observation_scalar(observation, 43)
        intersection_own_distance = self._observation_scalar(observation, 44)
        intersection_eta_risk = self._observation_scalar(observation, 46)
        intersection_other_first = self._observation_scalar(observation, 47)
        intersection_other_tighter = self._observation_scalar(observation, 50)
        prefix_conflict_count = self._observation_scalar(observation, 51)
        obs_checks = {
            "yield_stop_guard3_on_switch": on_switch,
            "yield_stop_guard3_active_fraction": active_fraction,
            "yield_stop_guard3_stop_proximity": stop_proximity,
            "yield_stop_guard3_route_distance": route_distance,
            "yield_stop_guard3_route_opposing": route_opposing,
            "yield_stop_guard3_route_same_direction": route_same_direction,
            "yield_stop_guard3_route_other_tighter": route_other_tighter,
            "yield_stop_guard3_route_count": route_count,
            "yield_stop_guard3_intersection_own_distance": (
                intersection_own_distance
            ),
            "yield_stop_guard3_intersection_eta_risk": intersection_eta_risk,
            "yield_stop_guard3_intersection_other_first": (
                intersection_other_first
            ),
            "yield_stop_guard3_intersection_other_tighter": (
                intersection_other_tighter
            ),
            "yield_stop_guard3_prefix_conflict_count": prefix_conflict_count,
        }
        for key, value in obs_checks.items():
            if value is None:
                return False, scores
            scores[key] = float(value)
        scores.update(
            {
                "yield_stop_guard_profile": "scene4_timing_stop",
                "yield_stop_guard3_distance": float(distance),
                "yield_stop_guard3_slack": float(slack),
                "yield_stop_guard3_tighter_agents": int(tighter_agents),
                "yield_stop_guard3_holds": int(holds),
            }
        )

        if (
            not np.isfinite(distance)
            or distance < self.yield_stop_guard3_min_distance
            or distance > self.yield_stop_guard3_max_distance
            or not np.isfinite(slack)
            or slack < self.yield_stop_guard3_min_slack
            or slack > self.yield_stop_guard3_max_slack
            or tighter_agents < self.yield_stop_guard3_min_tighter_agents
            or on_switch > self.yield_stop_guard3_max_on_switch
            or active_fraction < self.yield_stop_guard3_min_active_fraction
            or active_fraction > self.yield_stop_guard3_max_active_fraction
            or stop_proximity < self.yield_stop_guard3_min_stop_proximity
            or stop_proximity > self.yield_stop_guard3_max_stop_proximity
            or route_distance < self.yield_stop_guard3_min_route_distance
            or route_distance > self.yield_stop_guard3_max_route_distance
            or route_opposing > self.yield_stop_guard3_max_route_opposing
            or route_same_direction
            > self.yield_stop_guard3_max_route_same_direction
            or route_other_tighter > self.yield_stop_guard3_max_route_other_tighter
            or route_count < self.yield_stop_guard3_min_route_count
            or route_count > self.yield_stop_guard3_max_route_count
            or intersection_own_distance
            < self.yield_stop_guard3_min_intersection_own_distance
            or intersection_own_distance
            > self.yield_stop_guard3_max_intersection_own_distance
            or intersection_eta_risk
            < self.yield_stop_guard3_min_intersection_eta_risk
            or intersection_eta_risk
            > self.yield_stop_guard3_max_intersection_eta_risk
            or intersection_other_first
            > self.yield_stop_guard3_max_intersection_other_first
            or intersection_other_tighter
            > self.yield_stop_guard3_max_intersection_other_tighter
            or prefix_conflict_count
            < self.yield_stop_guard3_min_prefix_conflict_count
            or prefix_conflict_count
            > self.yield_stop_guard3_max_prefix_conflict_count
        ):
            return False, scores
        return True, scores

    def _apply_yield_stop_guard(
        self,
        output: dict[int, RailEnvActions],
        handles: List[int],
        obs_builder: Any,
        observations_by_handle: dict[int, Any],
        seed: int | None,
        step: int | None,
    ) -> dict[int, RailEnvActions]:
        if (
            not (
                self.yield_stop_guard_enabled
                or self.yield_stop_guard2_enabled
                or self.yield_stop_guard3_enabled
            )
            or obs_builder is None
        ):
            return output
        self._reset_yield_stop_guard_state(seed, step)
        adjusted = dict(output)
        for handle in handles:
            if handle not in adjusted:
                continue
            previous_action = self._action_id(adjusted[handle])
            accepted, scores = self._yield_stop_guard_scores(
                obs_builder,
                int(handle),
                previous_action,
                step,
                observations_by_handle.get(int(handle)),
            )
            if not accepted:
                accepted, scores = self._yield_stop_guard2_scores(
                    obs_builder,
                    int(handle),
                    previous_action,
                    step,
                    observations_by_handle.get(int(handle)),
                )
            if not accepted:
                accepted, scores = self._yield_stop_guard3_scores(
                    obs_builder,
                    int(handle),
                    previous_action,
                    step,
                    observations_by_handle.get(int(handle)),
                )
            if not accepted:
                continue
            adjusted[handle] = RailEnvActions.STOP_MOVING
            self._yield_stop_counts[int(handle)] = (
                self._yield_stop_counts.get(int(handle), 0) + 1
            )
            self._trace_yield_stop_guard(
                handle=int(handle),
                seed=seed,
                step=step,
                previous_action=previous_action,
                scores=scores,
            )
        return adjusted

    def _head_on_yield_guard_scores(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        scores: dict[str, Any] = {}
        if not self.head_on_yield_guard_enabled:
            return False, scores
        if obs_builder is None or step is None or observation is None:
            return False, scores
        if int(action) not in self.head_on_yield_guard_actions:
            return False, scores
        if (
            step < self.head_on_yield_guard_min_step
            or step > self.head_on_yield_guard_max_step
        ):
            return False, scores
        if self.head_on_yield_guard_allowed_scenes:
            try:
                scene = runtime_context.get().scene
            except Exception:
                scene = None
            if scene not in self.head_on_yield_guard_allowed_scenes:
                return False, scores
        holds = self._head_on_yield_counts.get(handle, 0)
        if holds >= self.head_on_yield_guard_max_holds:
            return False, scores
        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "MOVING"):
                return False, scores
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = self._guard_effective_slack(obs_builder, handle, distance)
        except Exception:
            return False, scores

        route_distance = self._observation_scalar(observation, 36)
        route_opposing = self._observation_scalar(observation, 37)
        route_other_tighter = self._observation_scalar(observation, 40)
        route_count = self._observation_scalar(observation, 43)
        intersection_own_distance = self._observation_scalar(observation, 44)
        intersection_eta_risk = self._observation_scalar(observation, 46)
        intersection_other_tighter = self._observation_scalar(observation, 50)
        prefix_conflict_count = self._observation_scalar(observation, 51)
        priority_tighter_fraction = self._observation_scalar(observation, 53)
        obs_checks = {
            "head_on_yield_guard_route_distance": route_distance,
            "head_on_yield_guard_route_opposing": route_opposing,
            "head_on_yield_guard_route_other_tighter": route_other_tighter,
            "head_on_yield_guard_route_count": route_count,
            "head_on_yield_guard_intersection_own_distance": (
                intersection_own_distance
            ),
            "head_on_yield_guard_intersection_eta_risk": intersection_eta_risk,
            "head_on_yield_guard_intersection_other_tighter": (
                intersection_other_tighter
            ),
            "head_on_yield_guard_prefix_conflict_count": prefix_conflict_count,
            "head_on_yield_guard_priority_tighter_fraction": (
                priority_tighter_fraction
            ),
        }
        for key, value in obs_checks.items():
            if value is None:
                return False, scores
            scores[key] = float(value)
        scores.update(
            {
                "head_on_yield_guard_distance": float(distance),
                "head_on_yield_guard_slack": float(slack),
                "head_on_yield_guard_holds": int(holds),
            }
        )
        if (
            not np.isfinite(distance)
            or distance < self.head_on_yield_guard_min_distance
            or distance > self.head_on_yield_guard_max_distance
        ):
            return False, scores
        if (
            not np.isfinite(slack)
            or slack < self.head_on_yield_guard_min_slack
            or slack > self.head_on_yield_guard_max_slack
        ):
            return False, scores
        if (
            route_distance > self.head_on_yield_guard_max_route_distance
            or route_opposing < 0.5
            or route_other_tighter < 0.5
            or route_count <= 0.0
            or intersection_own_distance
            > self.head_on_yield_guard_max_intersection_own_distance
            or intersection_eta_risk
            < self.head_on_yield_guard_min_intersection_eta_risk
            or intersection_other_tighter < 0.5
            or prefix_conflict_count
            < self.head_on_yield_guard_min_prefix_conflict_count
            or priority_tighter_fraction
            < self.head_on_yield_guard_min_priority_tighter_fraction
        ):
            return False, scores
        return True, scores

    def _apply_head_on_yield_guard(
        self,
        output: dict[int, RailEnvActions],
        handles: List[int],
        obs_builder: Any,
        observations_by_handle: dict[int, Any],
        seed: int | None,
        step: int | None,
    ) -> dict[int, RailEnvActions]:
        if not self.head_on_yield_guard_enabled or obs_builder is None:
            return output
        self._reset_head_on_yield_guard_state(seed, step)
        adjusted = dict(output)
        for handle in handles:
            if handle not in adjusted:
                continue
            previous_action = self._action_id(adjusted[handle])
            accepted, scores = self._head_on_yield_guard_scores(
                obs_builder,
                int(handle),
                previous_action,
                step,
                observations_by_handle.get(int(handle)),
            )
            if not accepted:
                continue
            adjusted[handle] = RailEnvActions.STOP_MOVING
            self._head_on_yield_counts[int(handle)] = (
                self._head_on_yield_counts.get(int(handle), 0) + 1
            )
            self._trace_head_on_yield_guard(
                handle=int(handle),
                seed=seed,
                step=step,
                previous_action=previous_action,
                scores=scores,
            )
        return adjusted

    def _detour_left_guard_scores(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        scores: dict[str, Any] = {}
        if not self.detour_left_guard_enabled:
            return False, scores
        if obs_builder is None or step is None or observation is None:
            return False, scores
        if step < self.detour_left_guard_min_step or step > self.detour_left_guard_max_step:
            return False, scores
        if int(action) != 2:
            return False, scores
        holds = self._detour_left_counts.get(handle, 0)
        if holds >= self.detour_left_guard_max_holds:
            return False, scores
        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "MOVING"):
                return False, scores
            target_position, target_direction = obs_builder._action_target(handle, 1)
            if target_position is None or target_direction is None:
                return False, scores
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = self._guard_effective_slack(obs_builder, handle, distance)
        except Exception:
            return False, scores

        tighter_agents = 0
        try:
            for other_handle in obs_builder.env.get_agent_handles():
                if other_handle == handle:
                    continue
                other_distance = float(
                    obs_builder._current_distance_to_waypoint(other_handle)
                )
                other_slack = self._guard_effective_slack(
                    obs_builder,
                    other_handle,
                    other_distance,
                )
                if np.isfinite(other_slack) and other_slack < slack:
                    tighter_agents += 1
        except Exception:
            tighter_agents = 0

        intersection_own_distance = self._observation_scalar(observation, 44)
        intersection_eta_risk = self._observation_scalar(observation, 46)
        intersection_other_first = self._observation_scalar(observation, 47)
        intersection_other_tighter = self._observation_scalar(observation, 50)
        priority_tighter_fraction = self._observation_scalar(observation, 53)
        route_occupancy_opposing = self._observation_scalar(observation, 37)
        route_occupancy_count = self._observation_scalar(observation, 43)
        scores.update(
            {
                "detour_left_guard_distance": float(distance),
                "detour_left_guard_slack": float(slack),
                "detour_left_guard_tighter_agents": int(tighter_agents),
                "detour_left_guard_holds": int(holds),
            }
        )
        obs_checks = {
            "detour_left_guard_intersection_own_distance": (
                intersection_own_distance
            ),
            "detour_left_guard_intersection_eta_risk": intersection_eta_risk,
            "detour_left_guard_intersection_other_first": intersection_other_first,
            "detour_left_guard_intersection_other_tighter": (
                intersection_other_tighter
            ),
            "detour_left_guard_priority_tighter_fraction": (
                priority_tighter_fraction
            ),
            "detour_left_guard_route_occupancy_opposing": route_occupancy_opposing,
            "detour_left_guard_route_occupancy_count": route_occupancy_count,
        }
        for key, value in obs_checks.items():
            if value is None:
                return False, scores
            scores[key] = float(value)

        if (
            not np.isfinite(distance)
            or distance < self.detour_left_guard_min_distance
            or distance > self.detour_left_guard_max_distance
        ):
            return False, scores
        if (
            not np.isfinite(slack)
            or slack < self.detour_left_guard_min_slack
            or slack > self.detour_left_guard_max_slack
        ):
            return False, scores
        try:
            other_count = max(1, obs_builder.env.get_num_agents() - 1)
        except Exception:
            other_count = 1
        if priority_tighter_fraction is not None:
            max_tighter_fraction = (
                self.detour_left_guard_max_tighter_agents / other_count
            )
            scores["detour_left_guard_max_tighter_fraction"] = float(
                max_tighter_fraction
            )
            if priority_tighter_fraction > max_tighter_fraction + 1e-6:
                return False, scores
        elif tighter_agents > self.detour_left_guard_max_tighter_agents:
            return False, scores
        if (
            intersection_eta_risk < self.detour_left_guard_min_intersection_eta_risk
            or intersection_eta_risk > self.detour_left_guard_max_intersection_eta_risk
            or intersection_other_first > 0.5
            or intersection_other_tighter > 0.5
            or intersection_own_distance
            > self.detour_left_guard_max_intersection_own_distance
            or route_occupancy_opposing > 0.5
            or route_occupancy_count > 0.01
        ):
            return False, scores
        return True, scores

    def _apply_detour_left_guard(
        self,
        output: dict[int, RailEnvActions],
        handles: List[int],
        obs_builder: Any,
        observations_by_handle: dict[int, Any],
        seed: int | None,
        step: int | None,
    ) -> dict[int, RailEnvActions]:
        if not self.detour_left_guard_enabled or obs_builder is None:
            return output
        self._reset_detour_left_guard_state(seed, step)
        adjusted = dict(output)
        for handle in handles:
            if handle not in adjusted:
                continue
            previous_action = self._action_id(adjusted[handle])
            accepted, scores = self._detour_left_guard_scores(
                obs_builder,
                int(handle),
                previous_action,
                step,
                observations_by_handle.get(int(handle)),
            )
            if not accepted:
                continue
            adjusted[handle] = RailEnvActions.MOVE_LEFT
            self._detour_left_counts[int(handle)] = (
                self._detour_left_counts.get(int(handle), 0) + 1
            )
            self._trace_detour_left_guard(
                handle=int(handle),
                seed=seed,
                step=step,
                previous_action=previous_action,
                scores=scores,
            )
        return adjusted

    def _switch_forward_guard_scores_for_profile(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
        *,
        profile_name: str,
        enabled: bool,
        allowed_scenes: set[str],
        min_step: float,
        max_step: float,
        max_holds: float,
        min_distance: float,
        max_distance: float,
        min_slack: float,
        max_slack: float,
        min_active_fraction: float,
        max_active_fraction: float,
        min_stop_proximity: float,
        min_route_distance: float,
        max_route_distance: float,
        max_intersection_own_distance: float,
        min_intersection_eta_risk: float,
        max_intersection_eta_risk: float,
        min_prefix_conflict_count: float,
        max_prefix_conflict_count: float,
        min_priority_tighter_fraction: float,
        max_priority_tighter_fraction: float,
        min_forward_distance_delta: float,
        max_forward_distance_delta: float,
        min_left_distance_delta: float,
        max_left_distance_delta: float,
    ) -> tuple[bool, dict[str, Any]]:
        scores: dict[str, Any] = {}
        if not enabled:
            return False, scores
        if obs_builder is None or step is None or observation is None:
            return False, scores
        if int(action) != 1:
            return False, scores
        if step < min_step or step > max_step:
            return False, scores
        if allowed_scenes:
            try:
                scene = runtime_context.get().scene
            except Exception:
                scene = None
            if scene not in allowed_scenes:
                return False, scores
        holds = self._switch_forward_counts.get(handle, 0)
        if holds >= max_holds:
            return False, scores

        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "MOVING"):
                return False, scores
            if agent.position is None:
                return False, scores
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = self._guard_effective_slack(obs_builder, handle, distance)
            local_mask = obs_builder._build_local_action_mask(handle)
            coordinated_mask = obs_builder._coordination_masks.get(handle, local_mask)
            target_position, target_direction = obs_builder._action_target(handle, 2)
        except Exception:
            return False, scores

        if (
            target_position is None
            or target_direction is None
            or local_mask[2] < 0.5
            or coordinated_mask[2] < 0.5
        ):
            return False, scores
        try:
            if obs_builder._occupied_by_other(target_position, handle):
                return False, scores
        except Exception:
            return False, scores

        forward_distance_delta = self._candidate_distance_delta(
            obs_builder,
            handle,
            2,
        )
        left_distance_delta = self._candidate_distance_delta(
            obs_builder,
            handle,
            1,
        )
        if (
            forward_distance_delta is None
            or left_distance_delta is None
            or not np.isfinite(forward_distance_delta)
            or not np.isfinite(left_distance_delta)
        ):
            return False, scores

        on_switch = self._observation_scalar(observation, 7)
        active_fraction = self._observation_scalar(observation, 32)
        stop_proximity = self._observation_scalar(observation, 33)
        route_distance = self._observation_scalar(observation, 36)
        route_opposing = self._observation_scalar(observation, 37)
        route_same_direction = self._observation_scalar(observation, 38)
        route_other_tighter = self._observation_scalar(observation, 40)
        route_count = self._observation_scalar(observation, 43)
        intersection_own_distance = self._observation_scalar(observation, 44)
        intersection_eta_risk = self._observation_scalar(observation, 46)
        intersection_other_first = self._observation_scalar(observation, 47)
        intersection_other_tighter = self._observation_scalar(observation, 50)
        prefix_conflict_count = self._observation_scalar(observation, 51)
        priority_tighter_fraction = self._observation_scalar(observation, 53)
        obs_checks = {
            "switch_forward_guard_on_switch": on_switch,
            "switch_forward_guard_active_fraction": active_fraction,
            "switch_forward_guard_stop_proximity": stop_proximity,
            "switch_forward_guard_route_distance": route_distance,
            "switch_forward_guard_route_opposing": route_opposing,
            "switch_forward_guard_route_same_direction": route_same_direction,
            "switch_forward_guard_route_other_tighter": route_other_tighter,
            "switch_forward_guard_route_count": route_count,
            "switch_forward_guard_intersection_own_distance": (
                intersection_own_distance
            ),
            "switch_forward_guard_intersection_eta_risk": intersection_eta_risk,
            "switch_forward_guard_intersection_other_first": (
                intersection_other_first
            ),
            "switch_forward_guard_intersection_other_tighter": (
                intersection_other_tighter
            ),
            "switch_forward_guard_prefix_conflict_count": prefix_conflict_count,
            "switch_forward_guard_priority_tighter_fraction": (
                priority_tighter_fraction
            ),
        }
        for key, value in obs_checks.items():
            if value is None:
                return False, scores
            scores[key] = float(value)
        scores.update(
            {
                "switch_forward_guard_profile": profile_name,
                "switch_forward_guard_distance": float(distance),
                "switch_forward_guard_slack": float(slack),
                "switch_forward_guard_holds": int(holds),
                "switch_forward_guard_forward_distance_delta": float(
                    forward_distance_delta
                ),
                "switch_forward_guard_left_distance_delta": float(left_distance_delta),
                "switch_forward_guard_forward_target": str(target_position),
            }
        )

        if (
            not np.isfinite(distance)
            or distance < min_distance
            or distance > max_distance
        ):
            return False, scores
        if (
            not np.isfinite(slack)
            or slack < min_slack
            or slack > max_slack
        ):
            return False, scores
        max_route_other_tighter = 0.5
        max_intersection_other_tighter = 0.5
        if profile_name == "tertiary":
            max_route_other_tighter = self.switch_forward_guard3_max_route_other_tighter
            max_intersection_other_tighter = (
                self.switch_forward_guard3_max_intersection_other_tighter
            )
        if (
            on_switch < 0.5
            or active_fraction < min_active_fraction
            or active_fraction > max_active_fraction
            or stop_proximity < min_stop_proximity
            or route_distance < min_route_distance
            or route_distance > max_route_distance
            or route_opposing < 0.5
            or route_same_direction > 0.5
            or route_other_tighter > max_route_other_tighter
            or route_count < 0.5
            or intersection_own_distance
            > max_intersection_own_distance
            or intersection_eta_risk
            < min_intersection_eta_risk
            or intersection_eta_risk
            > max_intersection_eta_risk
            or intersection_other_first > 0.5
            or intersection_other_tighter > max_intersection_other_tighter
            or prefix_conflict_count
            < min_prefix_conflict_count
            or prefix_conflict_count
            > max_prefix_conflict_count
            or priority_tighter_fraction
            < min_priority_tighter_fraction
            or priority_tighter_fraction
            > max_priority_tighter_fraction
            or forward_distance_delta < min_forward_distance_delta
            or forward_distance_delta > max_forward_distance_delta
            or left_distance_delta < min_left_distance_delta
            or left_distance_delta > max_left_distance_delta
        ):
            return False, scores
        return True, scores

    def _switch_forward_guard_scores(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        profiles = [
            (
                "primary",
                self.switch_forward_guard_enabled,
                self.switch_forward_guard_allowed_scenes,
                self.switch_forward_guard_min_step,
                self.switch_forward_guard_max_step,
                self.switch_forward_guard_max_holds,
                self.switch_forward_guard_min_distance,
                self.switch_forward_guard_max_distance,
                self.switch_forward_guard_min_slack,
                self.switch_forward_guard_max_slack,
                self.switch_forward_guard_min_active_fraction,
                self.switch_forward_guard_max_active_fraction,
                self.switch_forward_guard_min_stop_proximity,
                self.switch_forward_guard_min_route_distance,
                self.switch_forward_guard_max_route_distance,
                self.switch_forward_guard_max_intersection_own_distance,
                self.switch_forward_guard_min_intersection_eta_risk,
                self.switch_forward_guard_max_intersection_eta_risk,
                self.switch_forward_guard_min_prefix_conflict_count,
                self.switch_forward_guard_max_prefix_conflict_count,
                self.switch_forward_guard_min_priority_tighter_fraction,
                self.switch_forward_guard_max_priority_tighter_fraction,
                self.switch_forward_guard_min_forward_distance_delta,
                self.switch_forward_guard_max_forward_distance_delta,
                self.switch_forward_guard_min_left_distance_delta,
                self.switch_forward_guard_max_left_distance_delta,
            ),
            (
                "secondary",
                self.switch_forward_guard2_enabled,
                self.switch_forward_guard2_allowed_scenes,
                self.switch_forward_guard2_min_step,
                self.switch_forward_guard2_max_step,
                self.switch_forward_guard2_max_holds,
                self.switch_forward_guard2_min_distance,
                self.switch_forward_guard2_max_distance,
                self.switch_forward_guard2_min_slack,
                self.switch_forward_guard2_max_slack,
                self.switch_forward_guard2_min_active_fraction,
                self.switch_forward_guard2_max_active_fraction,
                self.switch_forward_guard2_min_stop_proximity,
                self.switch_forward_guard2_min_route_distance,
                self.switch_forward_guard2_max_route_distance,
                self.switch_forward_guard2_max_intersection_own_distance,
                self.switch_forward_guard2_min_intersection_eta_risk,
                self.switch_forward_guard2_max_intersection_eta_risk,
                self.switch_forward_guard2_min_prefix_conflict_count,
                self.switch_forward_guard2_max_prefix_conflict_count,
                self.switch_forward_guard2_min_priority_tighter_fraction,
                self.switch_forward_guard2_max_priority_tighter_fraction,
                self.switch_forward_guard2_min_forward_distance_delta,
                self.switch_forward_guard2_max_forward_distance_delta,
                self.switch_forward_guard2_min_left_distance_delta,
                self.switch_forward_guard2_max_left_distance_delta,
            ),
            (
                "tertiary",
                self.switch_forward_guard3_enabled,
                self.switch_forward_guard3_allowed_scenes,
                self.switch_forward_guard3_min_step,
                self.switch_forward_guard3_max_step,
                self.switch_forward_guard3_max_holds,
                self.switch_forward_guard3_min_distance,
                self.switch_forward_guard3_max_distance,
                self.switch_forward_guard3_min_slack,
                self.switch_forward_guard3_max_slack,
                self.switch_forward_guard3_min_active_fraction,
                self.switch_forward_guard3_max_active_fraction,
                self.switch_forward_guard3_min_stop_proximity,
                self.switch_forward_guard3_min_route_distance,
                self.switch_forward_guard3_max_route_distance,
                self.switch_forward_guard3_max_intersection_own_distance,
                self.switch_forward_guard3_min_intersection_eta_risk,
                self.switch_forward_guard3_max_intersection_eta_risk,
                self.switch_forward_guard3_min_prefix_conflict_count,
                self.switch_forward_guard3_max_prefix_conflict_count,
                self.switch_forward_guard3_min_priority_tighter_fraction,
                self.switch_forward_guard3_max_priority_tighter_fraction,
                self.switch_forward_guard3_min_forward_distance_delta,
                self.switch_forward_guard3_max_forward_distance_delta,
                self.switch_forward_guard3_min_left_distance_delta,
                self.switch_forward_guard3_max_left_distance_delta,
            ),
        ]
        last_scores: dict[str, Any] = {}
        for profile in profiles:
            accepted, scores = self._switch_forward_guard_scores_for_profile(
                obs_builder,
                handle,
                action,
                step,
                observation,
                profile_name=profile[0],
                enabled=profile[1],
                allowed_scenes=profile[2],
                min_step=profile[3],
                max_step=profile[4],
                max_holds=profile[5],
                min_distance=profile[6],
                max_distance=profile[7],
                min_slack=profile[8],
                max_slack=profile[9],
                min_active_fraction=profile[10],
                max_active_fraction=profile[11],
                min_stop_proximity=profile[12],
                min_route_distance=profile[13],
                max_route_distance=profile[14],
                max_intersection_own_distance=profile[15],
                min_intersection_eta_risk=profile[16],
                max_intersection_eta_risk=profile[17],
                min_prefix_conflict_count=profile[18],
                max_prefix_conflict_count=profile[19],
                min_priority_tighter_fraction=profile[20],
                max_priority_tighter_fraction=profile[21],
                min_forward_distance_delta=profile[22],
                max_forward_distance_delta=profile[23],
                min_left_distance_delta=profile[24],
                max_left_distance_delta=profile[25],
            )
            if accepted:
                return True, scores
            if scores:
                last_scores = scores
        return False, last_scores

    def _apply_switch_forward_guard(
        self,
        output: dict[int, RailEnvActions],
        handles: List[int],
        obs_builder: Any,
        observations_by_handle: dict[int, Any],
        seed: int | None,
        step: int | None,
    ) -> dict[int, RailEnvActions]:
        if (
            not (
                self.switch_forward_guard_enabled
                or self.switch_forward_guard2_enabled
                or self.switch_forward_guard3_enabled
            )
            or obs_builder is None
        ):
            return output
        self._reset_switch_forward_guard_state(seed, step)
        adjusted = dict(output)
        for handle in handles:
            if handle not in adjusted:
                continue
            previous_action = self._action_id(adjusted[handle])
            accepted, scores = self._switch_forward_guard_scores(
                obs_builder,
                int(handle),
                previous_action,
                step,
                observations_by_handle.get(int(handle)),
            )
            if not accepted:
                continue
            adjusted[handle] = RailEnvActions.MOVE_FORWARD
            self._switch_forward_counts[int(handle)] = (
                self._switch_forward_counts.get(int(handle), 0) + 1
            )
            self._trace_switch_forward_guard(
                handle=int(handle),
                seed=seed,
                step=step,
                previous_action=previous_action,
                scores=scores,
            )
        return adjusted

    def _right_forward_guard_scores(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        scores: dict[str, Any] = {}
        if not self.right_forward_guard_enabled:
            return False, scores
        if obs_builder is None or step is None or observation is None:
            return False, scores
        if int(action) != 3:
            return False, scores
        if (
            step < self.right_forward_guard_min_step
            or step > self.right_forward_guard_max_step
        ):
            return False, scores
        try:
            scene = runtime_context.get().scene
        except Exception:
            scene = None
        if self.right_forward_guard_allowed_scenes:
            if scene not in self.right_forward_guard_allowed_scenes:
                return False, scores
        holds = self._right_forward_counts.get(handle, 0)
        if holds >= self.right_forward_guard_max_holds:
            return False, scores

        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "MOVING"):
                return False, scores
            if agent.position is None:
                return False, scores
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = self._guard_effective_slack(obs_builder, handle, distance)
            local_mask = obs_builder._build_local_action_mask(handle)
            coordinated_mask = obs_builder._coordination_masks.get(handle, local_mask)
            target_position, target_direction = obs_builder._action_target(handle, 2)
        except Exception:
            return False, scores

        if (
            target_position is None
            or target_direction is None
            or local_mask[2] < 0.5
            or coordinated_mask[2] < 0.5
        ):
            return False, scores
        try:
            if obs_builder._occupied_by_other(target_position, handle):
                return False, scores
        except Exception:
            return False, scores

        forward_distance_delta = self._candidate_distance_delta(
            obs_builder,
            handle,
            2,
        )
        right_distance_delta = self._candidate_distance_delta(
            obs_builder,
            handle,
            3,
        )
        if (
            forward_distance_delta is None
            or right_distance_delta is None
            or not np.isfinite(forward_distance_delta)
            or not np.isfinite(right_distance_delta)
        ):
            return False, scores

        on_switch = self._observation_scalar(observation, 7)
        active_fraction = self._observation_scalar(observation, 32)
        stop_proximity = self._observation_scalar(observation, 33)
        route_distance = self._observation_scalar(observation, 36)
        route_opposing = self._observation_scalar(observation, 37)
        route_same_direction = self._observation_scalar(observation, 38)
        route_other_tighter = self._observation_scalar(observation, 40)
        route_count = self._observation_scalar(observation, 43)
        intersection_own_distance = self._observation_scalar(observation, 44)
        intersection_eta_risk = self._observation_scalar(observation, 46)
        intersection_other_first = self._observation_scalar(observation, 47)
        intersection_other_tighter = self._observation_scalar(observation, 50)
        prefix_conflict_count = self._observation_scalar(observation, 51)
        priority_tighter_fraction = self._observation_scalar(observation, 53)
        obs_checks = {
            "right_forward_guard_on_switch": on_switch,
            "right_forward_guard_active_fraction": active_fraction,
            "right_forward_guard_stop_proximity": stop_proximity,
            "right_forward_guard_route_distance": route_distance,
            "right_forward_guard_route_opposing": route_opposing,
            "right_forward_guard_route_same_direction": route_same_direction,
            "right_forward_guard_route_other_tighter": route_other_tighter,
            "right_forward_guard_route_count": route_count,
            "right_forward_guard_intersection_own_distance": (
                intersection_own_distance
            ),
            "right_forward_guard_intersection_eta_risk": intersection_eta_risk,
            "right_forward_guard_intersection_other_first": (
                intersection_other_first
            ),
            "right_forward_guard_intersection_other_tighter": (
                intersection_other_tighter
            ),
            "right_forward_guard_prefix_conflict_count": prefix_conflict_count,
            "right_forward_guard_priority_tighter_fraction": (
                priority_tighter_fraction
            ),
        }
        for key, value in obs_checks.items():
            if value is None:
                return False, scores
            scores[key] = float(value)
        scores.update(
            {
                "switch_forward_guard_profile": "right_forward",
                "right_forward_guard_distance": float(distance),
                "right_forward_guard_slack": float(slack),
                "right_forward_guard_holds": int(holds),
                "right_forward_guard_forward_distance_delta": float(
                    forward_distance_delta
                ),
                "right_forward_guard_right_distance_delta": float(
                    right_distance_delta
                ),
                "right_forward_guard_forward_target": str(target_position),
            }
        )

        if (
            not np.isfinite(distance)
            or distance < self.right_forward_guard_min_distance
            or distance > self.right_forward_guard_max_distance
            or not np.isfinite(slack)
            or slack < self.right_forward_guard_min_slack
            or slack > self.right_forward_guard_max_slack
            or on_switch < 0.5
            or active_fraction < self.right_forward_guard_min_active_fraction
            or active_fraction > self.right_forward_guard_max_active_fraction
            or stop_proximity < self.right_forward_guard_min_stop_proximity
            or stop_proximity > self.right_forward_guard_max_stop_proximity
            or route_distance < self.right_forward_guard_min_route_distance
            or route_distance > self.right_forward_guard_max_route_distance
            or route_opposing < 0.5
            or route_same_direction > 0.5
            or route_other_tighter < self.right_forward_guard_min_route_other_tighter
            or route_count < self.right_forward_guard_min_route_count
            or route_count > self.right_forward_guard_max_route_count
            or intersection_own_distance
            > self.right_forward_guard_max_intersection_own_distance
            or intersection_eta_risk
            < self.right_forward_guard_min_intersection_eta_risk
            or intersection_eta_risk
            > self.right_forward_guard_max_intersection_eta_risk
            or intersection_other_first > 0.5
            or intersection_other_tighter
            < self.right_forward_guard_min_intersection_other_tighter
            or prefix_conflict_count
            < self.right_forward_guard_min_prefix_conflict_count
            or prefix_conflict_count
            > self.right_forward_guard_max_prefix_conflict_count
            or priority_tighter_fraction
            < self.right_forward_guard_min_priority_tighter_fraction
            or priority_tighter_fraction
            > self.right_forward_guard_max_priority_tighter_fraction
            or forward_distance_delta
            < self.right_forward_guard_min_forward_distance_delta
            or forward_distance_delta
            > self.right_forward_guard_max_forward_distance_delta
            or right_distance_delta < self.right_forward_guard_min_right_distance_delta
            or right_distance_delta > self.right_forward_guard_max_right_distance_delta
        ):
            return False, scores
        if scene == "scene_5" and (
            step > self.right_forward_guard_scene5_max_step
            or distance > self.right_forward_guard_scene5_max_distance
            or slack > self.right_forward_guard_scene5_max_slack
            or route_other_tighter
            > self.right_forward_guard_scene5_max_route_other_tighter
            or intersection_other_tighter
            > self.right_forward_guard_scene5_max_intersection_other_tighter
            or priority_tighter_fraction
            > self.right_forward_guard_scene5_max_priority_tighter_fraction
        ):
            return False, scores
        return True, scores

    def _apply_right_forward_guard(
        self,
        output: dict[int, RailEnvActions],
        handles: List[int],
        obs_builder: Any,
        observations_by_handle: dict[int, Any],
        seed: int | None,
        step: int | None,
    ) -> dict[int, RailEnvActions]:
        if not self.right_forward_guard_enabled or obs_builder is None:
            return output
        self._reset_right_forward_guard_state(seed, step)
        adjusted = dict(output)
        for handle in handles:
            if handle not in adjusted:
                continue
            previous_action = self._action_id(adjusted[handle])
            accepted, scores = self._right_forward_guard_scores(
                obs_builder,
                int(handle),
                previous_action,
                step,
                observations_by_handle.get(int(handle)),
            )
            if not accepted:
                continue
            adjusted[handle] = RailEnvActions.MOVE_FORWARD
            self._right_forward_counts[int(handle)] = (
                self._right_forward_counts.get(int(handle), 0) + 1
            )
            self._trace_switch_forward_guard(
                handle=int(handle),
                seed=seed,
                step=step,
                previous_action=previous_action,
                scores=scores,
            )
        return adjusted

    def _deadline_forward_guard_scores(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        scores: dict[str, Any] = {}
        if not self.deadline_forward_guard_enabled:
            return False, scores
        if obs_builder is None or step is None or observation is None:
            return False, scores
        if int(action) != 4:
            return False, scores
        if (
            step < self.deadline_forward_guard_min_step
            or step > self.deadline_forward_guard_max_step
        ):
            return False, scores
        if self.deadline_forward_guard_allowed_scenes:
            try:
                scene = runtime_context.get().scene
            except Exception:
                scene = None
            if scene not in self.deadline_forward_guard_allowed_scenes:
                return False, scores

        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "MOVING", "STOPPED"):
                return False, scores
            if agent.position is None:
                return False, scores
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = self._guard_effective_slack(obs_builder, handle, distance)
            local_mask = obs_builder._build_local_action_mask(handle)
            coordinated_mask = obs_builder._coordination_masks.get(handle, local_mask)
            target_position, target_direction = obs_builder._action_target(handle, 2)
            forward_corridor_len = len(obs_builder._corridor_edges_for_action(handle, 2))
        except Exception:
            return False, scores

        if (
            target_position is None
            or target_direction is None
            or local_mask[2] < 0.5
            or coordinated_mask[2] < 0.5
        ):
            return False, scores
        try:
            if obs_builder._occupied_by_other(target_position, handle):
                return False, scores
        except Exception:
            return False, scores

        on_switch = self._observation_scalar(observation, 7)
        active_fraction = self._observation_scalar(observation, 32)
        stop_proximity = self._observation_scalar(observation, 33)
        time_slack = self._observation_scalar(observation, 35)
        route_distance = self._observation_scalar(observation, 36)
        route_opposing = self._observation_scalar(observation, 37)
        route_same_direction = self._observation_scalar(observation, 38)
        route_other_tighter = self._observation_scalar(observation, 40)
        route_count = self._observation_scalar(observation, 43)
        intersection_own_distance = self._observation_scalar(observation, 44)
        intersection_eta_risk = self._observation_scalar(observation, 46)
        intersection_other_first = self._observation_scalar(observation, 47)
        intersection_other_tighter = self._observation_scalar(observation, 50)
        obs_checks = {
            "deadline_forward_guard_on_switch": on_switch,
            "deadline_forward_guard_active_fraction": active_fraction,
            "deadline_forward_guard_stop_proximity": stop_proximity,
            "deadline_forward_guard_time_slack": time_slack,
            "deadline_forward_guard_route_distance": route_distance,
            "deadline_forward_guard_route_opposing": route_opposing,
            "deadline_forward_guard_route_same_direction": route_same_direction,
            "deadline_forward_guard_route_other_tighter": route_other_tighter,
            "deadline_forward_guard_route_count": route_count,
            "deadline_forward_guard_intersection_own_distance": (
                intersection_own_distance
            ),
            "deadline_forward_guard_intersection_eta_risk": intersection_eta_risk,
            "deadline_forward_guard_intersection_other_first": (
                intersection_other_first
            ),
            "deadline_forward_guard_intersection_other_tighter": (
                intersection_other_tighter
            ),
        }
        for key, value in obs_checks.items():
            if value is None:
                return False, scores
            scores[key] = float(value)
        scores.update(
            {
                "switch_forward_guard_profile": "deadline_forward",
                "deadline_forward_guard_distance": float(distance),
                "deadline_forward_guard_slack": float(slack),
                "deadline_forward_guard_forward_corridor_len": int(
                    forward_corridor_len
                ),
                "deadline_forward_guard_forward_target": str(target_position),
            }
        )

        if (
            not np.isfinite(distance)
            or distance < self.deadline_forward_guard_min_distance
            or distance > self.deadline_forward_guard_max_distance
            or not np.isfinite(slack)
            or slack < self.deadline_forward_guard_min_slack
            or slack > self.deadline_forward_guard_max_slack
            or on_switch < 0.5
            or active_fraction < self.deadline_forward_guard_min_active_fraction
            or active_fraction > self.deadline_forward_guard_max_active_fraction
            or stop_proximity < self.deadline_forward_guard_min_stop_proximity
            or time_slack < self.deadline_forward_guard_min_time_slack
            or time_slack > self.deadline_forward_guard_max_time_slack
            or route_distance < self.deadline_forward_guard_min_route_distance
            or route_distance > self.deadline_forward_guard_max_route_distance
            or route_opposing < 0.5
            or route_same_direction > 0.5
            or route_other_tighter
            > self.deadline_forward_guard_max_route_other_tighter
            or route_count < self.deadline_forward_guard_min_route_count
            or route_count > self.deadline_forward_guard_max_route_count
            or intersection_own_distance
            > self.deadline_forward_guard_max_intersection_own_distance
            or intersection_eta_risk
            < self.deadline_forward_guard_min_intersection_eta_risk
            or intersection_eta_risk
            > self.deadline_forward_guard_max_intersection_eta_risk
            or intersection_other_first > 0.5
            or intersection_other_tighter
            > self.deadline_forward_guard_max_intersection_other_tighter
            or forward_corridor_len
            < self.deadline_forward_guard_min_forward_corridor_len
            or forward_corridor_len
            > self.deadline_forward_guard_max_forward_corridor_len
        ):
            return False, scores
        return True, scores

    def _apply_deadline_forward_guard(
        self,
        output: dict[int, RailEnvActions],
        handles: List[int],
        obs_builder: Any,
        observations_by_handle: dict[int, Any],
        seed: int | None,
        step: int | None,
    ) -> dict[int, RailEnvActions]:
        if not self.deadline_forward_guard_enabled or obs_builder is None:
            return output
        adjusted = dict(output)
        for handle in handles:
            if handle not in adjusted:
                continue
            previous_action = self._action_id(adjusted[handle])
            accepted, scores = self._deadline_forward_guard_scores(
                obs_builder,
                int(handle),
                previous_action,
                step,
                observations_by_handle.get(int(handle)),
            )
            if not accepted:
                continue
            adjusted[handle] = RailEnvActions.MOVE_FORWARD
            self._trace_switch_forward_guard(
                handle=int(handle),
                seed=seed,
                step=step,
                previous_action=previous_action,
                scores=scores,
            )
        return adjusted

    def _inf_right_target_scores(
        self,
        obs_builder: Any,
        handle: int,
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        scores: dict[str, Any] = {}
        try:
            agent = obs_builder.env.agents[handle]
            position = agent.position
            direction = (
                agent.direction
                if agent.direction is not None
                else agent.initial_direction
            )
            if position is None or direction is None:
                return None, scores
            distance_map = obs_builder._get_distance_map(handle)
            current_distance = float(distance_map[position[0], position[1], direction])
            forward_target, forward_direction = obs_builder._action_target(handle, 2)
            right_target, right_direction = obs_builder._action_target(handle, 3)
            if (
                forward_target is None
                or forward_direction is None
                or right_target is None
                or right_direction is None
            ):
                return None, scores
            forward_distance = float(
                distance_map[
                    forward_target[0],
                    forward_target[1],
                    forward_direction,
                ]
            )
            right_distance = float(
                distance_map[right_target[0], right_target[1], right_direction]
            )
            scores.update(
                {
                    "inf_right_guard_current_distance": float(current_distance),
                    "inf_right_guard_forward_distance": float(forward_distance),
                    "inf_right_guard_right_distance": float(right_distance),
                    "inf_right_guard_right_distance_delta": float(
                        right_distance - current_distance
                    )
                    if np.isfinite(right_distance) and np.isfinite(current_distance)
                    else float("inf"),
                    "inf_right_guard_forward_target": str(forward_target),
                    "inf_right_guard_right_target": str(right_target),
                }
            )
            return {
                "current_distance": current_distance,
                "forward_distance": forward_distance,
                "right_distance": right_distance,
                "right_delta": right_distance - current_distance,
                "right_target": right_target,
            }, scores
        except Exception:
            return None, scores

    def _inf_right_guard_scores(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        scores: dict[str, Any] = {}
        if not self.inf_right_guard_enabled:
            return False, scores
        if obs_builder is None or step is None or observation is None:
            return False, scores
        if int(action) != 2:
            return False, scores
        if step < self.inf_right_guard_min_step or step > self.inf_right_guard_max_step:
            return False, scores
        if self.inf_right_guard_allowed_scenes:
            try:
                scene = runtime_context.get().scene
            except Exception:
                scene = None
            if scene not in self.inf_right_guard_allowed_scenes:
                return False, scores
        holds = self._inf_right_counts.get(handle, 0)
        if holds >= self.inf_right_guard_max_holds:
            return False, scores

        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "MOVING"):
                return False, scores
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = self._guard_effective_slack(obs_builder, handle, distance)
            local_mask = obs_builder._build_local_action_mask(handle)
            coordinated_mask = obs_builder._coordination_masks.get(handle, local_mask)
        except Exception:
            return False, scores

        if (
            local_mask[2] < 0.5
            or coordinated_mask[2] < 0.5
            or local_mask[3] >= 0.5
            or coordinated_mask[3] >= 0.5
        ):
            return False, scores

        target_scores, target_trace = self._inf_right_target_scores(
            obs_builder,
            handle,
        )
        scores.update(target_trace)
        if target_scores is None:
            return False, scores
        right_target = target_scores["right_target"]
        try:
            if obs_builder._occupied_by_other(right_target, handle):
                return False, scores
        except Exception:
            return False, scores

        on_switch = self._observation_scalar(observation, 7)
        active_fraction = self._observation_scalar(observation, 32)
        stop_proximity = self._observation_scalar(observation, 33)
        route_distance = self._observation_scalar(observation, 36)
        route_opposing = self._observation_scalar(observation, 37)
        route_same_direction = self._observation_scalar(observation, 38)
        route_other_tighter = self._observation_scalar(observation, 40)
        route_count = self._observation_scalar(observation, 43)
        intersection_own_distance = self._observation_scalar(observation, 44)
        intersection_eta_risk = self._observation_scalar(observation, 46)
        intersection_other_first = self._observation_scalar(observation, 47)
        intersection_other_tighter = self._observation_scalar(observation, 50)
        prefix_conflict_count = self._observation_scalar(observation, 51)
        priority_tighter_fraction = self._observation_scalar(observation, 53)
        obs_checks = {
            "inf_right_guard_on_switch": on_switch,
            "inf_right_guard_active_fraction": active_fraction,
            "inf_right_guard_stop_proximity": stop_proximity,
            "inf_right_guard_route_distance": route_distance,
            "inf_right_guard_route_opposing": route_opposing,
            "inf_right_guard_route_same_direction": route_same_direction,
            "inf_right_guard_route_other_tighter": route_other_tighter,
            "inf_right_guard_route_count": route_count,
            "inf_right_guard_intersection_own_distance": intersection_own_distance,
            "inf_right_guard_intersection_eta_risk": intersection_eta_risk,
            "inf_right_guard_intersection_other_first": intersection_other_first,
            "inf_right_guard_intersection_other_tighter": intersection_other_tighter,
            "inf_right_guard_prefix_conflict_count": prefix_conflict_count,
            "inf_right_guard_priority_tighter_fraction": priority_tighter_fraction,
        }
        for key, value in obs_checks.items():
            if value is None:
                return False, scores
            scores[key] = float(value)
        scores.update(
            {
                "inf_right_guard_distance": float(distance),
                "inf_right_guard_slack": float(slack),
                "inf_right_guard_holds": int(holds),
            }
        )

        right_delta = float(target_scores["right_delta"])
        if (
            not np.isfinite(distance)
            or distance < self.inf_right_guard_min_distance
            or distance > self.inf_right_guard_max_distance
        ):
            return False, scores
        if (
            not np.isfinite(slack)
            or slack < self.inf_right_guard_min_slack
            or slack > self.inf_right_guard_max_slack
        ):
            return False, scores
        if (
            np.isfinite(float(target_scores["forward_distance"]))
            or not np.isfinite(float(target_scores["right_distance"]))
            or not np.isfinite(right_delta)
            or right_delta < self.inf_right_guard_min_right_distance_delta
            or right_delta > self.inf_right_guard_max_right_distance_delta
        ):
            return False, scores
        if (
            on_switch < 0.5
            or active_fraction < self.inf_right_guard_min_active_fraction
            or active_fraction > self.inf_right_guard_max_active_fraction
            or stop_proximity > self.inf_right_guard_max_stop_proximity
            or route_distance < self.inf_right_guard_min_route_distance
            or route_distance > self.inf_right_guard_max_route_distance
            or route_opposing > 0.5
            or route_same_direction < 0.5
            or route_other_tighter < 0.5
            or route_count < 0.5
            or intersection_own_distance
            < self.inf_right_guard_min_intersection_own_distance
            or intersection_own_distance
            > self.inf_right_guard_max_intersection_own_distance
            or intersection_eta_risk < self.inf_right_guard_min_intersection_eta_risk
            or intersection_eta_risk > self.inf_right_guard_max_intersection_eta_risk
            or intersection_other_first < 0.5
            or intersection_other_tighter < 0.5
            or prefix_conflict_count
            < self.inf_right_guard_min_prefix_conflict_count
            or prefix_conflict_count
            > self.inf_right_guard_max_prefix_conflict_count
            or priority_tighter_fraction
            < self.inf_right_guard_min_priority_tighter_fraction
            or priority_tighter_fraction
            > self.inf_right_guard_max_priority_tighter_fraction
        ):
            return False, scores
        return True, scores

    def _apply_inf_right_guard(
        self,
        output: dict[int, RailEnvActions],
        handles: List[int],
        obs_builder: Any,
        observations_by_handle: dict[int, Any],
        seed: int | None,
        step: int | None,
    ) -> dict[int, RailEnvActions]:
        if not self.inf_right_guard_enabled or obs_builder is None:
            return output
        self._reset_inf_right_guard_state(seed, step)
        adjusted = dict(output)
        for handle in handles:
            if handle not in adjusted:
                continue
            previous_action = self._action_id(adjusted[handle])
            accepted, scores = self._inf_right_guard_scores(
                obs_builder,
                int(handle),
                previous_action,
                step,
                observations_by_handle.get(int(handle)),
            )
            if not accepted:
                continue
            adjusted[handle] = RailEnvActions.MOVE_RIGHT
            self._inf_right_counts[int(handle)] = (
                self._inf_right_counts.get(int(handle), 0) + 1
            )
            self._trace_inf_right_guard(
                handle=int(handle),
                seed=seed,
                step=step,
                previous_action=previous_action,
                scores=scores,
            )
        return adjusted

    def _short_block_stop_guard_scores(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
    ) -> tuple[int | None, dict[str, Any]]:
        scores: dict[str, Any] = {}
        if not self.short_block_stop_guard_enabled:
            return None, scores
        if obs_builder is None or step is None or observation is None:
            return None, scores
        if int(action) not in self.short_block_stop_guard_actions:
            return None, scores
        if (
            step < self.short_block_stop_guard_min_step
            or step > self.short_block_stop_guard_max_step
        ):
            return None, scores
        if self.short_block_stop_guard_allowed_scenes:
            try:
                scene = runtime_context.get().scene
            except Exception:
                scene = None
            if scene not in self.short_block_stop_guard_allowed_scenes:
                return None, scores
        holds = self._short_block_stop_counts.get(handle, 0)
        if holds >= self.short_block_stop_guard_max_holds:
            return None, scores

        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "MOVING"):
                return None, scores
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = self._guard_effective_slack(obs_builder, handle, distance)
            local_mask = obs_builder._build_local_action_mask(handle)
            coordinated_mask = obs_builder._coordination_masks.get(handle, local_mask)
            forward_target, _forward_direction = obs_builder._action_target(handle, 2)
            detour_target, _detour_direction = obs_builder._action_target(handle, int(action))
            forward_distance = float(obs_builder._target_distance(handle, 2))
            detour_distance = float(obs_builder._target_distance(handle, int(action)))
        except Exception:
            return None, scores

        if forward_target is None or detour_target is None:
            return None, scores
        try:
            forward_occupied = obs_builder._occupied_by_other(forward_target, handle)
            detour_occupied = obs_builder._occupied_by_other(detour_target, handle)
        except Exception:
            return None, scores
        if not forward_occupied or detour_occupied:
            return None, scores

        if coordinated_mask[4] >= 0.5:
            wait_action = 4
        elif coordinated_mask[0] >= 0.5:
            wait_action = 0
        else:
            return None, scores

        detour_distance_delta = detour_distance - distance
        forward_distance_delta = forward_distance - distance
        route_distance = self._observation_scalar(observation, 36)
        intersection_eta_risk = self._observation_scalar(observation, 46)
        prefix_conflict_count = self._observation_scalar(observation, 51)
        scores.update(
            {
                "short_block_stop_guard_distance": float(distance),
                "short_block_stop_guard_slack": float(slack),
                "short_block_stop_guard_holds": int(holds),
                "short_block_stop_guard_forward_distance": float(forward_distance),
                "short_block_stop_guard_detour_distance": float(detour_distance),
                "short_block_stop_guard_forward_distance_delta": float(
                    forward_distance_delta
                ),
                "short_block_stop_guard_detour_distance_delta": float(
                    detour_distance_delta
                ),
                "short_block_stop_guard_forward_target": str(forward_target),
                "short_block_stop_guard_detour_target": str(detour_target),
            }
        )
        if route_distance is not None:
            scores["short_block_stop_guard_route_distance"] = float(route_distance)
        if intersection_eta_risk is not None:
            scores["short_block_stop_guard_intersection_eta_risk"] = float(
                intersection_eta_risk
            )
        if prefix_conflict_count is not None:
            scores["short_block_stop_guard_prefix_conflict_count"] = float(
                prefix_conflict_count
            )

        if (
            not np.isfinite(distance)
            or not np.isfinite(slack)
            or not np.isfinite(forward_distance)
            or not np.isfinite(detour_distance)
            or distance < self.short_block_stop_guard_min_distance
            or distance > self.short_block_stop_guard_max_distance
            or slack < self.short_block_stop_guard_min_slack
            or slack > self.short_block_stop_guard_max_slack
            or detour_distance_delta
            < self.short_block_stop_guard_min_detour_distance_delta
            or forward_distance_delta
            > self.short_block_stop_guard_max_forward_distance_delta
        ):
            return None, scores
        if (
            route_distance is not None
            and route_distance > self.short_block_stop_guard_max_route_distance
        ):
            return None, scores
        if (
            intersection_eta_risk is not None
            and intersection_eta_risk
            < self.short_block_stop_guard_min_intersection_eta_risk
        ):
            return None, scores
        if (
            prefix_conflict_count is not None
            and prefix_conflict_count
            < self.short_block_stop_guard_min_prefix_conflict_count
        ):
            return None, scores
        return wait_action, scores

    def _apply_short_block_stop_guard(
        self,
        output: dict[int, RailEnvActions],
        handles: List[int],
        obs_builder: Any,
        observations_by_handle: dict[int, Any],
        seed: int | None,
        step: int | None,
    ) -> dict[int, RailEnvActions]:
        if not self.short_block_stop_guard_enabled or obs_builder is None:
            return output
        self._reset_short_block_stop_guard_state(seed, step)
        adjusted = dict(output)
        for handle in handles:
            if handle not in adjusted:
                continue
            previous_action = self._action_id(adjusted[handle])
            candidate_action, scores = self._short_block_stop_guard_scores(
                obs_builder,
                int(handle),
                previous_action,
                step,
                observations_by_handle.get(int(handle)),
            )
            if candidate_action is None:
                continue
            adjusted[handle] = RailEnvActions(candidate_action)
            self._short_block_stop_counts[int(handle)] = (
                self._short_block_stop_counts.get(int(handle), 0) + 1
            )
            self._trace_short_block_stop_guard(
                handle=int(handle),
                seed=seed,
                step=step,
                previous_action=previous_action,
                candidate_action=candidate_action,
                scores=scores,
            )
        return adjusted

    def _early_yield_guard_scores_for_profile(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
        *,
        profile_name: str,
        enabled: bool,
        allowed_scenes: set[str],
        actions: set[int],
        min_step: float,
        max_step: float,
        max_holds: float,
        min_departure_lag: float,
        max_departure_lag: float,
        min_distance: float,
        max_distance: float,
        min_slack: float,
        max_slack: float,
        min_active_fraction: float,
        max_active_fraction: float,
        min_stop_proximity: float,
        max_stop_proximity: float,
        min_route_distance: float,
        max_route_count: float,
        max_prefix_conflict_count: float,
        max_action_distance_delta: float,
    ) -> tuple[bool, dict[str, Any]]:
        scores: dict[str, Any] = {}
        if not enabled:
            return False, scores
        if obs_builder is None or step is None or observation is None:
            return False, scores
        if int(action) not in actions:
            return False, scores
        if step < min_step or step > max_step:
            return False, scores
        if allowed_scenes:
            try:
                scene = runtime_context.get().scene
            except Exception:
                scene = None
            if scene not in allowed_scenes:
                return False, scores
        holds = self._early_yield_counts.get(handle, 0)
        if holds >= max_holds:
            return False, scores

        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "MOVING"):
                return False, scores
            if agent.position is None:
                return False, scores
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = self._guard_effective_slack(obs_builder, handle, distance)
            local_mask = obs_builder._build_local_action_mask(handle)
            coordinated_mask = obs_builder._coordination_masks.get(handle, local_mask)
            departure_lag = float(step - int(agent.earliest_departure))
            action_distance = float(obs_builder._target_distance(handle, int(action)))
            forward_distance = float(obs_builder._target_distance(handle, 2))
        except Exception:
            return False, scores

        if coordinated_mask[4] < 0.5:
            return False, scores
        action_distance_delta = action_distance - distance
        forward_distance_delta = forward_distance - distance
        active_fraction = self._observation_scalar(observation, 32)
        stop_proximity = self._observation_scalar(observation, 33)
        route_distance = self._observation_scalar(observation, 36)
        route_count = self._observation_scalar(observation, 43)
        prefix_conflict_count = self._observation_scalar(observation, 51)
        obs_checks = {
            "early_yield_guard_active_fraction": active_fraction,
            "early_yield_guard_stop_proximity": stop_proximity,
            "early_yield_guard_route_distance": route_distance,
            "early_yield_guard_route_count": route_count,
            "early_yield_guard_prefix_conflict_count": prefix_conflict_count,
        }
        for key, value in obs_checks.items():
            if value is None:
                return False, scores
            scores[key] = float(value)
        scores.update(
            {
                "early_yield_guard_profile": profile_name,
                "early_yield_guard_distance": float(distance),
                "early_yield_guard_slack": float(slack),
                "early_yield_guard_holds": int(holds),
                "early_yield_guard_departure_lag": float(departure_lag),
                "early_yield_guard_action_distance_delta": float(
                    action_distance_delta
                ),
                "early_yield_guard_forward_distance_delta": float(
                    forward_distance_delta
                ),
            }
        )

        if (
            not np.isfinite(distance)
            or not np.isfinite(slack)
            or not np.isfinite(action_distance_delta)
            or distance < min_distance
            or distance > max_distance
            or slack < min_slack
            or slack > max_slack
            or departure_lag < min_departure_lag
            or departure_lag > max_departure_lag
            or active_fraction < min_active_fraction
            or active_fraction > max_active_fraction
            or stop_proximity < min_stop_proximity
            or stop_proximity > max_stop_proximity
            or route_distance < min_route_distance
            or route_count > max_route_count
            or prefix_conflict_count
            > max_prefix_conflict_count
            or action_distance_delta > max_action_distance_delta
        ):
            return False, scores
        return True, scores

    def _early_yield_guard_scores(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        profiles = [
            (
                "primary",
                self.early_yield_guard_enabled,
                self.early_yield_guard_allowed_scenes,
                self.early_yield_guard_actions,
                self.early_yield_guard_min_step,
                self.early_yield_guard_max_step,
                self.early_yield_guard_max_holds,
                self.early_yield_guard_min_departure_lag,
                self.early_yield_guard_max_departure_lag,
                self.early_yield_guard_min_distance,
                self.early_yield_guard_max_distance,
                self.early_yield_guard_min_slack,
                self.early_yield_guard_max_slack,
                self.early_yield_guard_min_active_fraction,
                self.early_yield_guard_max_active_fraction,
                self.early_yield_guard_min_stop_proximity,
                self.early_yield_guard_max_stop_proximity,
                self.early_yield_guard_min_route_distance,
                self.early_yield_guard_max_route_count,
                self.early_yield_guard_max_prefix_conflict_count,
                self.early_yield_guard_max_action_distance_delta,
            ),
            (
                "secondary",
                self.early_yield_guard2_enabled,
                self.early_yield_guard2_allowed_scenes,
                self.early_yield_guard2_actions,
                self.early_yield_guard2_min_step,
                self.early_yield_guard2_max_step,
                self.early_yield_guard2_max_holds,
                self.early_yield_guard2_min_departure_lag,
                self.early_yield_guard2_max_departure_lag,
                self.early_yield_guard2_min_distance,
                self.early_yield_guard2_max_distance,
                self.early_yield_guard2_min_slack,
                self.early_yield_guard2_max_slack,
                self.early_yield_guard2_min_active_fraction,
                self.early_yield_guard2_max_active_fraction,
                self.early_yield_guard2_min_stop_proximity,
                self.early_yield_guard2_max_stop_proximity,
                self.early_yield_guard2_min_route_distance,
                self.early_yield_guard2_max_route_count,
                self.early_yield_guard2_max_prefix_conflict_count,
                self.early_yield_guard2_max_action_distance_delta,
            ),
        ]
        last_scores: dict[str, Any] = {}
        for profile in profiles:
            accepted, scores = self._early_yield_guard_scores_for_profile(
                obs_builder,
                handle,
                action,
                step,
                observation,
                profile_name=profile[0],
                enabled=profile[1],
                allowed_scenes=profile[2],
                actions=profile[3],
                min_step=profile[4],
                max_step=profile[5],
                max_holds=profile[6],
                min_departure_lag=profile[7],
                max_departure_lag=profile[8],
                min_distance=profile[9],
                max_distance=profile[10],
                min_slack=profile[11],
                max_slack=profile[12],
                min_active_fraction=profile[13],
                max_active_fraction=profile[14],
                min_stop_proximity=profile[15],
                max_stop_proximity=profile[16],
                min_route_distance=profile[17],
                max_route_count=profile[18],
                max_prefix_conflict_count=profile[19],
                max_action_distance_delta=profile[20],
            )
            if accepted:
                return True, scores
            if scores:
                last_scores = scores
        return False, last_scores

    def _apply_early_yield_guard(
        self,
        output: dict[int, RailEnvActions],
        handles: List[int],
        obs_builder: Any,
        observations_by_handle: dict[int, Any],
        seed: int | None,
        step: int | None,
    ) -> dict[int, RailEnvActions]:
        if (
            not (self.early_yield_guard_enabled or self.early_yield_guard2_enabled)
            or obs_builder is None
        ):
            return output
        self._reset_early_yield_guard_state(seed, step)
        adjusted = dict(output)
        for handle in handles:
            if handle not in adjusted:
                continue
            previous_action = self._action_id(adjusted[handle])
            accepted, scores = self._early_yield_guard_scores(
                obs_builder,
                int(handle),
                previous_action,
                step,
                observations_by_handle.get(int(handle)),
            )
            if not accepted:
                continue
            adjusted[handle] = RailEnvActions.STOP_MOVING
            self._early_yield_counts[int(handle)] = (
                self._early_yield_counts.get(int(handle), 0) + 1
            )
            self._trace_early_yield_guard(
                handle=int(handle),
                seed=seed,
                step=step,
                previous_action=previous_action,
                scores=scores,
            )
        return adjusted

    def _switch_escape_candidate_actions(
        self,
        obs_builder: Any,
        handle: int,
    ) -> list[tuple[int, float, tuple[int, int] | None]]:
        candidates: list[tuple[int, float, tuple[int, int] | None]] = []
        for action in sorted(self.switch_escape_guard_allowed_actions):
            if action not in (1, 2, 3):
                continue
            try:
                target_position, _target_direction = obs_builder._action_target(
                    handle,
                    action,
                )
            except Exception:
                continue
            if target_position is None:
                continue
            try:
                if obs_builder._occupied_by_other(target_position, handle):
                    continue
            except Exception:
                continue
            distance_delta = self._candidate_distance_delta(
                obs_builder,
                handle,
                action,
            )
            if distance_delta is None or not np.isfinite(distance_delta):
                continue
            if (
                distance_delta < self.switch_escape_guard_min_distance_delta
                or distance_delta > self.switch_escape_guard_max_distance_delta
            ):
                continue
            candidates.append((int(action), float(distance_delta), target_position))
        return sorted(candidates, key=lambda item: (item[1], item[0]))

    def _switch_escape_guard_scores(
        self,
        obs_builder: Any,
        handle: int,
        action: int,
        step: int | None,
        observation: Any | None = None,
    ) -> tuple[int | None, dict[str, Any]]:
        scores: dict[str, Any] = {}
        if not self.switch_escape_guard_enabled:
            return None, scores
        if obs_builder is None or step is None or observation is None:
            return None, scores
        if int(action) not in (0, 4):
            return None, scores
        if (
            step < self.switch_escape_guard_min_step
            or step > self.switch_escape_guard_max_step
        ):
            return None, scores
        if self.switch_escape_guard_allowed_scenes:
            try:
                scene = runtime_context.get().scene
            except Exception:
                scene = None
            if scene not in self.switch_escape_guard_allowed_scenes:
                return None, scores
        holds = self._switch_escape_counts.get(handle, 0)
        if holds >= self.switch_escape_guard_max_holds:
            return None, scores

        try:
            agent = obs_builder.env.agents[handle]
            if not obs_builder._state_matches(agent.state, "MOVING", "STOPPED"):
                return None, scores
            if agent.position is None:
                return None, scores
            distance = float(obs_builder._current_distance_to_waypoint(handle))
            slack = self._guard_effective_slack(obs_builder, handle, distance)
            local_mask = obs_builder._build_local_action_mask(handle)
            coordinated_mask = obs_builder._coordination_masks.get(handle, local_mask)
        except Exception:
            return None, scores

        if np.any(np.asarray(coordinated_mask[1:4], dtype=np.float32) >= 0.5):
            return None, scores
        if local_mask[4] < 0.5 and coordinated_mask[0] < 0.5:
            return None, scores

        on_switch = self._observation_scalar(observation, 7)
        active_fraction = self._observation_scalar(observation, 32)
        stop_proximity = self._observation_scalar(observation, 33)
        route_distance = self._observation_scalar(observation, 36)
        route_opposing = self._observation_scalar(observation, 37)
        intersection_own_distance = self._observation_scalar(observation, 44)
        intersection_eta_risk = self._observation_scalar(observation, 46)
        prefix_conflict_count = self._observation_scalar(observation, 51)
        obs_checks = {
            "switch_escape_guard_on_switch": on_switch,
            "switch_escape_guard_active_fraction": active_fraction,
            "switch_escape_guard_stop_proximity": stop_proximity,
            "switch_escape_guard_route_distance": route_distance,
            "switch_escape_guard_route_opposing": route_opposing,
            "switch_escape_guard_intersection_own_distance": (
                intersection_own_distance
            ),
            "switch_escape_guard_intersection_eta_risk": intersection_eta_risk,
            "switch_escape_guard_prefix_conflict_count": prefix_conflict_count,
        }
        for key, value in obs_checks.items():
            if value is None:
                return None, scores
            scores[key] = float(value)
        scores.update(
            {
                "switch_escape_guard_distance": float(distance),
                "switch_escape_guard_slack": float(slack),
                "switch_escape_guard_holds": int(holds),
            }
        )

        if (
            not np.isfinite(distance)
            or distance < self.switch_escape_guard_min_distance
            or distance > self.switch_escape_guard_max_distance
        ):
            return None, scores
        if (
            not np.isfinite(slack)
            or slack < self.switch_escape_guard_min_slack
            or slack > self.switch_escape_guard_max_slack
        ):
            return None, scores
        if (
            on_switch < 0.5
            or active_fraction < self.switch_escape_guard_min_active_fraction
            or stop_proximity < self.switch_escape_guard_min_stop_proximity
            or route_distance > self.switch_escape_guard_max_route_distance
            or route_opposing < 0.5
            or intersection_own_distance
            > self.switch_escape_guard_max_intersection_own_distance
            or intersection_eta_risk
            < self.switch_escape_guard_min_intersection_eta_risk
            or prefix_conflict_count
            < self.switch_escape_guard_min_prefix_conflict_count
        ):
            return None, scores

        candidates = self._switch_escape_candidate_actions(obs_builder, handle)
        if not candidates:
            return None, scores
        candidate_action, distance_delta, target_position = candidates[0]
        scores["switch_escape_guard_candidate_distance_delta"] = float(distance_delta)
        scores["switch_escape_guard_candidate_target"] = str(target_position)
        return candidate_action, scores

    def _apply_switch_escape_guard(
        self,
        output: dict[int, RailEnvActions],
        handles: List[int],
        obs_builder: Any,
        observations_by_handle: dict[int, Any],
        seed: int | None,
        step: int | None,
    ) -> dict[int, RailEnvActions]:
        if not self.switch_escape_guard_enabled or obs_builder is None:
            return output
        self._reset_switch_escape_guard_state(seed, step)
        adjusted = dict(output)
        for handle in handles:
            if handle not in adjusted:
                continue
            previous_action = self._action_id(adjusted[handle])
            candidate_action, scores = self._switch_escape_guard_scores(
                obs_builder,
                int(handle),
                previous_action,
                step,
                observations_by_handle.get(int(handle)),
            )
            if candidate_action is None:
                continue
            adjusted[handle] = RailEnvActions(candidate_action)
            self._switch_escape_counts[int(handle)] = (
                self._switch_escape_counts.get(int(handle), 0) + 1
            )
            self._trace_switch_escape_guard(
                handle=int(handle),
                seed=seed,
                step=step,
                previous_action=previous_action,
                candidate_action=candidate_action,
                scores=scores,
            )
        return adjusted

    def _candidate_action_lists(
        self,
        handles: List[int],
        observations: List[Any],
        baseline_action_ids: dict[int, int],
        candidate_actions: Dict[int, RailEnvActions],
        extra_candidate_actions: (
            list[tuple[str, Dict[int, RailEnvActions]]] | None
        ) = None,
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
            existing = {action for action, _ in result[handle]}
            for source, actions_by_handle in extra_candidate_actions or []:
                if handle not in actions_by_handle:
                    continue
                action_id = self._action_id(actions_by_handle[handle])
                if action_id in existing:
                    continue
                result[handle].append(
                    (
                        action_id,
                        {"candidate_source": source},
                    )
                )
                existing.add(action_id)

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
        extra_candidate_actions = [
            (source, policy.act_many(handles, observations, **kwargs))
            for source, policy in self.extra_candidate_policies
        ]
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
            extra_candidate_actions,
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
                    accepted = self._extra_candidate_rescue_relax(
                        observations_by_handle.get(handle),
                        baseline_action,
                        candidate_action,
                        step,
                        scores,
                    )
                if not accepted:
                    accepted = self._start_candidate_rescue_relax(
                        observations_by_handle.get(handle),
                        baseline_action,
                        candidate_action,
                        step,
                        scores,
                    )
                if not accepted:
                    accepted = self._deadline_right_rescue_relax(
                        obs_builder,
                        int(handle),
                        observations_by_handle.get(handle),
                        baseline_action,
                        candidate_action,
                        step,
                        scores,
                    )
                if accepted and self._start_relax_active_distance_veto(
                    obs_builder,
                    int(handle),
                    candidate_action,
                    scores,
                ):
                    accepted = False
                if accepted and self._start_relax_opposing_route_veto(
                    observations_by_handle.get(handle),
                    scores,
                ):
                    accepted = False
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
                    baseline_action,
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
        output = self._apply_start_delay_guard(
            output,
            handles,
            obs_builder,
            observations_by_handle,
            seed,
            step,
        )
        output = self._apply_yield_stop_guard(
            output,
            handles,
            obs_builder,
            observations_by_handle,
            seed,
            step,
        )
        output = self._apply_head_on_yield_guard(
            output,
            handles,
            obs_builder,
            observations_by_handle,
            seed,
            step,
        )
        output = self._apply_detour_left_guard(
            output,
            handles,
            obs_builder,
            observations_by_handle,
            seed,
            step,
        )
        output = self._apply_inf_right_guard(
            output,
            handles,
            obs_builder,
            observations_by_handle,
            seed,
            step,
        )
        output = self._apply_short_block_stop_guard(
            output,
            handles,
            obs_builder,
            observations_by_handle,
            seed,
            step,
        )
        output = self._apply_early_yield_guard(
            output,
            handles,
            obs_builder,
            observations_by_handle,
            seed,
            step,
        )
        output = self._apply_switch_forward_guard(
            output,
            handles,
            obs_builder,
            observations_by_handle,
            seed,
            step,
        )
        output = self._apply_right_forward_guard(
            output,
            handles,
            obs_builder,
            observations_by_handle,
            seed,
            step,
        )
        output = self._apply_deadline_forward_guard(
            output,
            handles,
            obs_builder,
            observations_by_handle,
            seed,
            step,
        )
        return self._apply_switch_escape_guard(
            output,
            handles,
            obs_builder,
            observations_by_handle,
            seed,
            step,
        )


MyPolicy = RiskVetoPolicy
