from __future__ import annotations

import argparse
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
from submission.sequence_success_policy import (
    DEFAULT_CANDIDATE_CHECKPOINT_PATHS,
    SequenceSuccessPolicy,
)

try:
    from tools.analyze_policy_action_diffs import (
        diff_row,
        raw_policy_actions,
    )
except Exception:  # pragma: no cover - submission fallback for stripped packages.
    diff_row = None
    raw_policy_actions = None


class EventGateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int, output_dim: int = 1):
        super().__init__()
        if hidden_size <= 0:
            self.net = nn.Linear(input_dim, output_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_dim),
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        output = self.net(features)
        return output.squeeze(-1) if output.shape[-1] == 1 else output


class ExactEventGate:
    def __init__(self, checkpoint_path: str):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        config = checkpoint["config"]
        self.feature_columns = list(config["feature_columns"])
        self.mean = np.asarray(config["mean"], dtype=np.float32)
        self.std = np.asarray(config["std"], dtype=np.float32)
        self.objective = str(config.get("objective", "binary"))
        self.max_bad_probability = float(config.get("max_bad_probability", 0.05))
        self.model = EventGateMLP(
            int(config["input_dim"]),
            int(config["hidden_size"]),
            int(config.get("output_dim", 1)),
        )
        self.model.load_state_dict(checkpoint["model"])
        self.model.eval()

    @staticmethod
    def _float_value(value: Any) -> float:
        try:
            result = float(value)
        except Exception:
            return 0.0
        return result if np.isfinite(result) else 0.0

    def feature_vector(self, row: dict[str, Any]) -> np.ndarray:
        values = np.asarray(
            [self._float_value(row.get(column)) for column in self.feature_columns],
            dtype=np.float32,
        )
        return (values - self.mean) / self.std

    def score(self, row: dict[str, Any]) -> dict[str, float]:
        features = torch.as_tensor(self.feature_vector(row)[None, :], dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(features)
            if self.objective == "multiclass":
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
                return {
                    "event_gate_probability": float(probs[1]),
                    "event_gate_bad_probability": float(probs[2]),
                }
            probability = float(torch.sigmoid(logits).item())
            return {
                "event_gate_probability": probability,
                "event_gate_bad_probability": float("nan"),
            }


class ExactEventGatePolicy:
    """Experimental learned selector for direct PPO event proposals.

    The Sequence policy remains the baseline. A direct PPO/Rerank candidate may
    override it only if a small exact-event gate, trained on forced
    candidate-vs-baseline counterfactual labels, scores the event above the
    configured threshold. This is intentionally experimental and is not the
    default submission policy.
    """

    def __init__(self, checkpoint_path: str | None = None):
        candidate_checkpoint = (
            os.environ.get("ECML_EXACT_EVENT_GATE_CANDIDATE_CHECKPOINT", "").strip()
            or DEFAULT_CANDIDATE_CHECKPOINT_PATHS[-1]
        )
        gate_checkpoint = (
            checkpoint_path
            or os.environ.get("ECML_EXACT_EVENT_GATE_CHECKPOINT", "").strip()
        )
        self.baseline_policy = SequenceSuccessPolicy()
        self.candidate_policy = RerankPolicy(checkpoint_path=candidate_checkpoint)
        self.gate = (
            ExactEventGate(gate_checkpoint)
            if gate_checkpoint and Path(gate_checkpoint).exists()
            else None
        )
        self.risk_policy = self._load_actor(
            os.environ.get("ECML_EXACT_EVENT_GATE_RISK_CHECKPOINT", "").strip()
        )
        self.reward_risk_policy = self._load_actor(
            os.environ.get(
                "ECML_EXACT_EVENT_GATE_REWARD_RISK_CHECKPOINT",
                "",
            ).strip()
        )
        self.value_policy = self._load_actor(
            os.environ.get("ECML_EXACT_EVENT_GATE_VALUE_CHECKPOINT", "").strip()
        )
        self.threshold = self._env_float("ECML_EXACT_EVENT_GATE_THRESHOLD", 0.75)
        self.max_bad_probability = self._env_float(
            "ECML_EXACT_EVENT_GATE_MAX_BAD_PROBABILITY",
            0.05,
        )
        self.max_accepted_events = self._env_int(
            "ECML_EXACT_EVENT_GATE_MAX_ACCEPTED_EVENTS",
            1,
        )
        self.trace_path = os.environ.get("ECML_EXACT_EVENT_GATE_TRACE_PATH", "").strip()
        self._accepted_events = 0
        self._last_seed: int | None = None
        self._last_step: int | None = None

    @staticmethod
    def _load_actor(checkpoint_path: str) -> ActorCritic | None:
        if not checkpoint_path or not Path(checkpoint_path).exists():
            return None
        policy = ActorCritic(checkpoint_path=checkpoint_path)
        policy.eval()
        return policy

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

    def _reset_episode_state(self, seed: int | None, step: int | None) -> None:
        if seed != self._last_seed or (
            step is not None and self._last_step is not None and step < self._last_step
        ):
            self._accepted_events = 0
        self._last_seed = seed
        self._last_step = step

    def _head_scores(
        self,
        observation: Any,
        baseline_action: int,
        candidate_action: int,
    ) -> dict[str, float]:
        scores: dict[str, float] = {}
        obs = np.asarray(observation, dtype=np.float32)
        for prefix, model in (
            ("risk_head", self.risk_policy),
            ("reward_risk_head", self.reward_risk_policy),
            ("value_head", self.value_policy),
        ):
            if model is None:
                continue
            try:
                with torch.no_grad():
                    if prefix == "value_head":
                        values = model.action_value_scores(obs).squeeze(0).cpu().numpy()
                        baseline_score = float(values[baseline_action])
                        candidate_score = float(values[candidate_action])
                        scores[f"{prefix}_baseline"] = baseline_score
                        scores[f"{prefix}_candidate"] = candidate_score
                        scores[f"{prefix}_candidate_minus_baseline"] = (
                            candidate_score - baseline_score
                        )
                    else:
                        probs = torch.sigmoid(model.risk_logits(obs)).squeeze(0).cpu().numpy()
                        baseline_risk = float(probs[baseline_action])
                        candidate_risk = float(probs[candidate_action])
                        scores[f"{prefix}_baseline"] = baseline_risk
                        scores[f"{prefix}_candidate"] = candidate_risk
                        scores[f"{prefix}_candidate_minus_baseline"] = (
                            candidate_risk - baseline_risk
                        )
                        scores[f"{prefix}_baseline_minus_candidate"] = (
                            baseline_risk - candidate_risk
                        )
            except Exception:
                scores[f"{prefix}_score_error"] = 1.0
        return scores

    def _event_row(
        self,
        *,
        seed: int | None,
        env: Any,
        obs_builder: Any,
        handle: int,
        observation: Any,
        baseline_raw_actions: dict[int, int],
        candidate_raw_actions: dict[int, int],
        baseline_actions: dict[int, int],
        candidate_actions: dict[int, int],
    ) -> dict[str, Any] | None:
        if diff_row is None:
            return None
        try:
            row = diff_row(
                argparse.Namespace(),
                int(seed) if seed is not None else -1,
                env,
                obs_builder,
                handle,
                observation,
                self.baseline_policy,
                self.candidate_policy,
                baseline_raw_actions,
                candidate_raw_actions,
                baseline_actions,
                candidate_actions,
            )
        except Exception:
            return None
        baseline_action = int(baseline_actions[handle])
        candidate_action = int(candidate_actions[handle])
        row.update(
            self._head_scores(
                observation,
                baseline_action,
                candidate_action,
            )
        )
        return row

    def _trace(
        self,
        row: dict[str, Any],
        *,
        accepted: bool,
        scores: dict[str, float],
        reject_reason: str,
    ) -> None:
        if not self.trace_path:
            return
        context = runtime_context.get()
        payload = {
            "accepted": bool(accepted),
            "reject_reason": reject_reason,
            "scene": context.scene,
            **{
                key: row.get(key)
                for key in (
                    "seed",
                    "env_time",
                    "agent_id",
                    "baseline_action",
                    "baseline_action_name",
                    "candidate_action",
                    "candidate_action_name",
                    "candidate_future_head_on_risk",
                    "candidate_deadline_conflict_penalty",
                    "risk_head_candidate",
                    "risk_head_candidate_minus_baseline",
                    "reward_risk_head_candidate",
                    "reward_risk_head_candidate_minus_baseline",
                )
            },
            **scores,
        }
        path = Path(self.trace_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def act_many(
        self,
        handles: List[int],
        observations: List[Any],
        **kwargs,
    ) -> Dict[int, RailEnvActions]:
        baseline_actions = self.baseline_policy.act_many(handles, observations, **kwargs)
        if self.gate is None or diff_row is None:
            return baseline_actions
        candidate_actions = self.candidate_policy.act_many(handles, observations, **kwargs)
        context = runtime_context.get()
        env = context.env
        obs_builder = context.obs_builder
        self._reset_episode_state(context.seed, context.step)
        if env is None or obs_builder is None:
            return baseline_actions

        baseline_action_ids = {
            handle: self._action_id(action)
            for handle, action in baseline_actions.items()
        }
        candidate_action_ids = {
            handle: self._action_id(action)
            for handle, action in candidate_actions.items()
        }
        baseline_raw_actions = (
            raw_policy_actions(self.baseline_policy, handles, observations)
            if raw_policy_actions is not None
            else {}
        )
        candidate_raw_actions = (
            raw_policy_actions(self.candidate_policy, handles, observations)
            if raw_policy_actions is not None
            else {}
        )

        output = dict(baseline_actions)
        for handle, observation in zip(handles, observations):
            if handle not in baseline_action_ids or handle not in candidate_action_ids:
                continue
            baseline_action = baseline_action_ids[handle]
            candidate_action = candidate_action_ids[handle]
            if baseline_action == candidate_action:
                continue
            if (
                self.max_accepted_events >= 0
                and self._accepted_events >= self.max_accepted_events
            ):
                continue
            row = self._event_row(
                seed=context.seed,
                env=env,
                obs_builder=obs_builder,
                handle=int(handle),
                observation=observation,
                baseline_raw_actions=baseline_raw_actions,
                candidate_raw_actions=candidate_raw_actions,
                baseline_actions=baseline_action_ids,
                candidate_actions=candidate_action_ids,
            )
            if row is None:
                continue
            scores = self.gate.score(row)
            accepted = scores["event_gate_probability"] >= self.threshold
            if self.gate.objective == "multiclass":
                accepted = (
                    accepted
                    and scores["event_gate_bad_probability"] <= self.max_bad_probability
                )
            reject_reason = "" if accepted else "event_gate_below_threshold"
            self._trace(row, accepted=accepted, scores=scores, reject_reason=reject_reason)
            if accepted:
                output[handle] = RailEnvActions(candidate_action)
                self._accepted_events += 1
        return output


MyPolicy = ExactEventGatePolicy
