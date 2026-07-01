from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from submission import runtime_context


DISPATCH_PAIR_FEATURE_NAMES = (
    "bias",
    "slack_left",
    "slack_right",
    "slack_delta",
    "path_len_left",
    "path_len_right",
    "path_len_delta",
    "wait_streak_left",
    "wait_streak_right",
    "wait_streak_delta",
    "rl_go_advantage_left",
    "rl_go_advantage_right",
    "rl_go_advantage_delta",
    "conflict_degree_left",
    "conflict_degree_right",
    "occupied_target_left",
    "occupied_target_right",
    "target_same",
    "reverse_immediate",
    "shared_edges",
    "opposing_edges",
    "eta_gap",
    "num_agents",
    "max_steps",
    "scene_id",
)
DISPATCH_PAIR_FEATURE_DIM = len(DISPATCH_PAIR_FEATURE_NAMES)


@dataclass(frozen=True)
class DispatchPairConflict:
    target_same: bool
    reverse_immediate: bool
    shared_edges: int
    opposing_edges: int

    @property
    def is_real(self) -> bool:
        return (
            self.target_same
            or self.reverse_immediate
            or self.shared_edges > 0
            or self.opposing_edges > 0
        )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off", ""}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    if not math.isfinite(result):
        return default
    return result


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _node_position(node: Any) -> Any:
    return getattr(node, "position", None)


def _edge(source: Any, target: Any) -> tuple[Any, Any]:
    return (source, target)


def _reverse_edge(edge: tuple[Any, Any]) -> tuple[Any, Any]:
    return (edge[1], edge[0])


def _candidate_edges(candidate: Any, limit: int = 24) -> tuple[tuple[Any, Any], ...]:
    nodes = list(getattr(candidate, "nodes", []) or [])
    edges: list[tuple[Any, Any]] = []
    for source_node, target_node in zip(nodes, nodes[1:]):
        source = _node_position(source_node)
        target = _node_position(target_node)
        if source is None or target is None or source == target:
            continue
        edges.append(_edge(source, target))
        if limit > 0 and len(edges) >= limit:
            break
    return tuple(edges)


def dispatch_pair_conflict(left: Any, right: Any) -> DispatchPairConflict:
    target_same = getattr(left, "target", None) == getattr(right, "target", None)
    reverse_immediate = (
        getattr(left, "target", None) == getattr(right, "source", None)
        and getattr(right, "target", None) == getattr(left, "source", None)
    )
    left_edges = set(_candidate_edges(left))
    right_edges = set(_candidate_edges(right))
    shared_edges = len(left_edges & right_edges)
    opposing_edges = sum(1 for edge in left_edges if _reverse_edge(edge) in right_edges)
    return DispatchPairConflict(
        target_same=bool(target_same),
        reverse_immediate=bool(reverse_immediate),
        shared_edges=int(shared_edges),
        opposing_edges=int(opposing_edges),
    )


def scene_id(scene: str | None) -> float:
    mapping = {
        "scene_1": 0.2,
        "scene_2": 0.4,
        "scene_3": 0.6,
        "scene_4": 0.8,
        "scene_5": 1.0,
    }
    return mapping.get(scene or "", 0.0)


def _num_agents(env: Any) -> int:
    try:
        return int(env.get_num_agents())
    except Exception:
        return 0


def _max_steps(env: Any) -> int:
    try:
        return int(getattr(env, "_max_episode_steps", 0) or 0)
    except Exception:
        return 0


def dispatch_pair_features(
    env: Any,
    left: Any,
    right: Any,
    conflict: DispatchPairConflict | None = None,
) -> list[float]:
    """Return fixed-width, finite features for P(left before right)."""

    conflict = conflict or dispatch_pair_conflict(left, right)
    max_steps = max(1, _max_steps(env))
    agents = max(1, _num_agents(env))
    scene = runtime_context.get().scene

    left_slack = _clip(_finite(getattr(left, "slack", 0.0)), -float(max_steps), float(max_steps))
    right_slack = _clip(
        _finite(getattr(right, "slack", 0.0)),
        -float(max_steps),
        float(max_steps),
    )
    left_path = _clip(_finite(getattr(left, "path_len", 0.0)), 0.0, float(max_steps))
    right_path = _clip(_finite(getattr(right, "path_len", 0.0)), 0.0, float(max_steps))
    left_wait = _clip(_finite(getattr(left, "wait_streak", 0.0)), 0.0, 50.0)
    right_wait = _clip(_finite(getattr(right, "wait_streak", 0.0)), 0.0, 50.0)
    left_adv = _clip(_finite(getattr(left, "rl_go_advantage", 0.0)), -2.0, 2.0)
    right_adv = _clip(_finite(getattr(right, "rl_go_advantage", 0.0)), -2.0, 2.0)
    left_conflict = _clip(_finite(getattr(left, "conflict_degree", 0.0)), 0.0, 20.0)
    right_conflict = _clip(_finite(getattr(right, "conflict_degree", 0.0)), 0.0, 20.0)

    features = [
        1.0,
        left_slack / float(max_steps),
        right_slack / float(max_steps),
        (left_slack - right_slack) / float(max_steps),
        left_path / float(max_steps),
        right_path / float(max_steps),
        (left_path - right_path) / float(max_steps),
        left_wait / 50.0,
        right_wait / 50.0,
        (left_wait - right_wait) / 50.0,
        left_adv / 2.0,
        right_adv / 2.0,
        (left_adv - right_adv) / 2.0,
        left_conflict / 20.0,
        right_conflict / 20.0,
        float(bool(getattr(left, "occupied_target", False))),
        float(bool(getattr(right, "occupied_target", False))),
        float(conflict.target_same),
        float(conflict.reverse_immediate),
        _clip(float(conflict.shared_edges), 0.0, 24.0) / 24.0,
        _clip(float(conflict.opposing_edges), 0.0, 24.0) / 24.0,
        (left_path - right_path) / float(max_steps),
        _clip(float(agents), 0.0, 200.0) / 200.0,
        _clip(float(max_steps), 0.0, 5000.0) / 5000.0,
        scene_id(scene),
    ]
    if len(features) != DISPATCH_PAIR_FEATURE_DIM:
        raise ValueError(
            f"dispatch feature dimension drift: {len(features)} != "
            f"{DISPATCH_PAIR_FEATURE_DIM}"
        )
    return [_finite(value, 0.0) for value in features]


def teacher_pair_probability(env: Any, left: Any, right: Any) -> float:
    """Conservative teacher for BC pretraining.

    Positive means the left candidate should be planned before the right one.
    The teacher encodes dispatch rules that have been stable in local failure
    analysis: urgent trains, already waiting trains, and dense-conflict trains
    get earlier reservations; occupied-target moves are de-prioritized.
    """

    max_steps = max(1, _max_steps(env))
    left_slack = _finite(getattr(left, "slack", max_steps), float(max_steps))
    right_slack = _finite(getattr(right, "slack", max_steps), float(max_steps))
    left_wait = _finite(getattr(left, "wait_streak", 0.0))
    right_wait = _finite(getattr(right, "wait_streak", 0.0))
    left_conflict = _finite(getattr(left, "conflict_degree", 0.0))
    right_conflict = _finite(getattr(right, "conflict_degree", 0.0))
    left_path = _finite(getattr(left, "path_len", max_steps), float(max_steps))
    right_path = _finite(getattr(right, "path_len", max_steps), float(max_steps))
    left_adv = _finite(getattr(left, "rl_go_advantage", 0.0))
    right_adv = _finite(getattr(right, "rl_go_advantage", 0.0))

    score = 0.0
    score += _clip((right_slack - left_slack) / 80.0, -2.0, 2.0)
    score += 0.18 * _clip(left_wait - right_wait, -8.0, 8.0)
    score += 0.12 * _clip(left_conflict - right_conflict, -8.0, 8.0)
    score += 0.35 * (float(bool(getattr(right, "occupied_target", False))) - float(bool(getattr(left, "occupied_target", False))))
    score += 0.10 * _clip(right_path - left_path, -20.0, 20.0) / 10.0
    score += 0.20 * _clip(left_adv - right_adv, -2.0, 2.0)
    return 1.0 / (1.0 + math.exp(-score))


class DispatchPairwiseMLP:
    """Tiny pairwise MLP factory kept import-light for submission runtime."""

    @staticmethod
    def build(input_dim: int = DISPATCH_PAIR_FEATURE_DIM, hidden_size: int = 64) -> Any:
        import torch

        return torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, 1),
        )


class DispatchRanker:
    """High-level RL/BC dispatch ranker over MAPF/SIPP candidates.

    The ranker only returns priority offsets. It never emits Flatland actions.
    If disabled, no compatible model is available, or no confident conflict pair
    exists, it returns an empty dict and the caller's heuristic order remains in
    full control.
    """

    def __init__(self, checkpoint_path: str | None = None):
        self.enabled = _env_bool("ECML_DISPATCH_RANKER_ENABLED", False)
        self.scenes = {
            scene.strip()
            for scene in os.environ.get(
                "ECML_DISPATCH_RANKER_SCENES",
                "scene_3,scene_4",
            ).split(",")
            if scene.strip()
        }
        self.min_agents = _env_int("ECML_DISPATCH_RANKER_MIN_AGENTS", 70)
        self.min_confidence = _env_float("ECML_DISPATCH_RANKER_MIN_CONFIDENCE", 0.55)
        self.weight = _env_float("ECML_DISPATCH_RANKER_WEIGHT", 1.0)
        default_path = Path(__file__).resolve().parent / "models" / "ecml_dispatch_pairwise_ranker_v1.pt"
        self.checkpoint_path = Path(
            checkpoint_path
            or os.environ.get("ECML_DISPATCH_RANKER_MODEL", str(default_path))
        )
        self._load_attempted = False
        self._model: Any | None = None
        self._torch: Any | None = None
        self.last_stats: dict[str, Any] = {}

    @property
    def model_available(self) -> bool:
        self._load_model()
        return self._model is not None and self._torch is not None

    def _load_model(self) -> None:
        if self._load_attempted:
            return
        self._load_attempted = True
        if not self.enabled:
            self.last_stats = {"status": "disabled"}
            return
        if not self.checkpoint_path.exists():
            self.last_stats = {"status": "missing_model", "path": str(self.checkpoint_path)}
            return
        try:
            import torch

            self._torch = torch
            try:
                model = torch.jit.load(str(self.checkpoint_path), map_location="cpu")
            except Exception:
                checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
                state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else checkpoint
                input_dim = int(
                    checkpoint.get("input_dim", DISPATCH_PAIR_FEATURE_DIM)
                    if isinstance(checkpoint, dict)
                    else DISPATCH_PAIR_FEATURE_DIM
                )
                hidden_size = int(
                    checkpoint.get("hidden_size", 64)
                    if isinstance(checkpoint, dict)
                    else 64
                )
                if input_dim != DISPATCH_PAIR_FEATURE_DIM:
                    self.last_stats = {
                        "status": "incompatible_model",
                        "input_dim": input_dim,
                    }
                    return
                model = DispatchPairwiseMLP.build(input_dim=input_dim, hidden_size=hidden_size)
                model.load_state_dict(state_dict)
            model.eval()
            self._model = model
            self.last_stats = {"status": "loaded", "path": str(self.checkpoint_path)}
        except Exception as exc:
            self._model = None
            self._torch = None
            self.last_stats = {
                "status": "load_failed",
                "path": str(self.checkpoint_path),
                "error": str(exc),
            }

    def _should_run(self, env: Any) -> bool:
        if not self.enabled:
            self.last_stats = {"status": "disabled"}
            return False
        scene = runtime_context.get().scene
        if self.scenes and scene not in self.scenes:
            self.last_stats = {"status": "scene_disabled", "scene": scene}
            return False
        agents = _num_agents(env)
        if agents < self.min_agents:
            self.last_stats = {"status": "too_few_agents", "num_agents": agents}
            return False
        if not self.model_available:
            return False
        return True

    def priority_scores(self, env: Any, candidates: Iterable[Any]) -> dict[int, float]:
        candidates = list(candidates)
        if len(candidates) < 2 or not self._should_run(env):
            return {}

        pairs: list[tuple[Any, Any, DispatchPairConflict]] = []
        rows: list[list[float]] = []
        for left_index, left in enumerate(candidates):
            for right in candidates[left_index + 1 :]:
                conflict = dispatch_pair_conflict(left, right)
                if not conflict.is_real:
                    continue
                pairs.append((left, right, conflict))
                rows.append(dispatch_pair_features(env, left, right, conflict))

        if not rows:
            self.last_stats = {"status": "no_conflicts", "num_candidates": len(candidates)}
            return {}
        if self._model is None or self._torch is None:
            return {}

        torch = self._torch
        try:
            with torch.no_grad():
                tensor = torch.tensor(rows, dtype=torch.float32)
                logits = self._model(tensor)
                probabilities = torch.sigmoid(logits.reshape(-1)).cpu().tolist()
        except Exception as exc:
            self.last_stats = {"status": "score_failed", "error": str(exc)}
            return {}

        scores: dict[int, float] = {int(getattr(candidate, "handle")): 0.0 for candidate in candidates}
        used_pairs = 0
        for (left, right, _conflict), probability in zip(pairs, probabilities):
            probability = _clip(_finite(probability, 0.5), 0.0, 1.0)
            confidence = max(probability, 1.0 - probability)
            if confidence < self.min_confidence:
                continue
            left_handle = int(getattr(left, "handle"))
            right_handle = int(getattr(right, "handle"))
            margin = 2.0 * (probability - 0.5)
            scores[left_handle] += margin
            scores[right_handle] -= margin
            used_pairs += 1

        if used_pairs == 0:
            self.last_stats = {
                "status": "low_confidence",
                "num_pairs": len(pairs),
                "min_confidence": self.min_confidence,
            }
            return {}

        for candidate in candidates:
            handle = int(getattr(candidate, "handle"))
            slack = _finite(getattr(candidate, "slack", 0.0), 0.0)
            wait = _finite(getattr(candidate, "wait_streak", 0.0), 0.0)
            conflict_degree = _finite(getattr(candidate, "conflict_degree", 0.0), 0.0)
            urgency = _clip((80.0 - slack) / 80.0, 0.0, 1.0)
            scores[handle] += 0.20 * urgency + 0.03 * wait + 0.03 * conflict_degree

        self.last_stats = {
            "status": "scored",
            "num_candidates": len(candidates),
            "num_pairs": len(pairs),
            "used_pairs": used_pairs,
        }
        return {handle: self.weight * score for handle, score in scores.items() if score != 0.0}


def dispatch_training_rows(
    env: Any,
    candidates: Iterable[Any],
    *,
    seed: int | None = None,
    step: int | None = None,
) -> list[dict[str, Any]]:
    rows = []
    candidates = list(candidates)
    scene = runtime_context.get().scene
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            conflict = dispatch_pair_conflict(left, right)
            if not conflict.is_real:
                continue
            features = dispatch_pair_features(env, left, right, conflict)
            probability = teacher_pair_probability(env, left, right)
            row = {
                "seed": seed,
                "step": step,
                "scene": scene,
                "left_handle": int(getattr(left, "handle")),
                "right_handle": int(getattr(right, "handle")),
                "teacher_probability": probability,
                "label": int(probability >= 0.5),
                "target_same_raw": int(conflict.target_same),
                "reverse_immediate_raw": int(conflict.reverse_immediate),
                "shared_edges_raw": int(conflict.shared_edges),
                "opposing_edges_raw": int(conflict.opposing_edges),
            }
            row.update(dict(zip(DISPATCH_PAIR_FEATURE_NAMES, features)))
            rows.append(row)
    return rows
