from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from flatland.envs.rail_env_action import RailEnvActions

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
        self.risk_policy = (
            ActorCritic(checkpoint_path=risk_checkpoint)
            if risk_checkpoint and Path(risk_checkpoint).exists()
            else None
        )
        if self.risk_policy is not None:
            self.risk_policy.eval()
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
        self.trace_path = os.environ.get("ECML_RISK_VETO_TRACE_PATH", "").strip()

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, default))
        except Exception:
            return default

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

    def _risk_scores(
        self,
        observation: Any,
        baseline_action: int,
        candidate_action: int,
    ) -> tuple[bool, dict[str, Any]]:
        if self.risk_policy is None:
            return False, {"reject_reason": "missing_risk_head"}
        try:
            with torch.no_grad():
                logits = self.risk_policy.risk_logits(
                    np.asarray(observation, dtype=np.float32)
                )
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
            baseline_risk = float(probs[baseline_action])
            candidate_risk = float(probs[candidate_action])
        except Exception:
            return False, {"reject_reason": "risk_score_error"}

        candidate_minus_baseline = candidate_risk - baseline_risk
        baseline_minus_candidate = baseline_risk - candidate_risk
        scores = {
            "risk_head_baseline": baseline_risk,
            "risk_head_candidate": candidate_risk,
            "risk_head_candidate_minus_baseline": candidate_minus_baseline,
            "risk_head_baseline_minus_candidate": baseline_minus_candidate,
        }
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
        return accepted, scores

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
        row = {
            "seed": seed,
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
        try:
            from submission import runtime_context

            context = runtime_context.get()
            seed = context.seed
            step = context.step
        except Exception:
            pass

        for handle in handles:
            if handle not in candidate_actions or handle not in baseline_actions:
                continue
            baseline_action = self._action_id(baseline_actions[handle])
            candidate_action = self._action_id(candidate_actions[handle])
            if candidate_action == baseline_action:
                continue
            accepted, scores = self._risk_scores(
                observations_by_handle.get(handle),
                baseline_action,
                candidate_action,
            )
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
        return output


MyPolicy = RiskVetoPolicy
