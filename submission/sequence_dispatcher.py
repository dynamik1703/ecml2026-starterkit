from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable, Iterable

from submission import runtime_context
from submission.dispatch_ranker import dispatch_pair_conflict, scene_id


SEQUENCE_DISPATCH_FEATURE_NAMES = (
    "bias",
    "slack",
    "urgency",
    "path_len",
    "wait_streak",
    "rl_go_advantage",
    "conflict_degree",
    "occupied_target",
    "action_left",
    "action_forward",
    "action_right",
    "source_row",
    "source_col",
    "target_row",
    "target_col",
    "route_edge_count",
    "route_distinct_cells",
    "elapsed_fraction",
    "num_agents",
    "max_steps",
    "scene_id",
)
SEQUENCE_DISPATCH_FEATURE_DIM = len(SEQUENCE_DISPATCH_FEATURE_NAMES)


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


def _elapsed_steps(env: Any) -> int:
    try:
        return int(getattr(env, "_elapsed_steps", 0) or 0)
    except Exception:
        return 0


def _grid_size(env: Any) -> tuple[int, int]:
    height = int(getattr(env, "height", 0) or 0)
    width = int(getattr(env, "width", 0) or 0)
    rail = getattr(env, "rail", None)
    grid = getattr(rail, "grid", None)
    if (height <= 0 or width <= 0) and getattr(grid, "shape", None) is not None:
        height = int(grid.shape[0])
        width = int(grid.shape[1])
    return max(1, height), max(1, width)


def _position_features(position: Any, height: int, width: int) -> tuple[float, float]:
    if not isinstance(position, tuple) or len(position) < 2:
        return 0.0, 0.0
    row = _clip(_finite(position[0]), 0.0, float(height))
    col = _clip(_finite(position[1]), 0.0, float(width))
    return row / float(height), col / float(width)


def _candidate_route_stats(candidate: Any) -> tuple[float, float]:
    nodes = list(getattr(candidate, "nodes", []) or [])
    positions = [
        getattr(node, "position", None)
        for node in nodes
        if getattr(node, "position", None) is not None
    ]
    edges = 0
    for source, target in zip(positions, positions[1:]):
        if source != target:
            edges += 1
    return float(min(edges, 48)) / 48.0, float(min(len(set(positions)), 48)) / 48.0


def sequence_candidate_features(env: Any, candidate: Any) -> list[float]:
    max_steps = max(1, _max_steps(env))
    num_agents = max(1, _num_agents(env))
    elapsed_steps = _elapsed_steps(env)
    height, width = _grid_size(env)
    source_row, source_col = _position_features(getattr(candidate, "source", None), height, width)
    target_row, target_col = _position_features(getattr(candidate, "target", None), height, width)
    route_edges, route_cells = _candidate_route_stats(candidate)

    slack = _clip(_finite(getattr(candidate, "slack", 0.0)), -float(max_steps), float(max_steps))
    path_len = _clip(_finite(getattr(candidate, "path_len", 0.0)), 0.0, float(max_steps))
    wait_streak = _clip(_finite(getattr(candidate, "wait_streak", 0.0)), 0.0, 50.0)
    rl_advantage = _clip(_finite(getattr(candidate, "rl_go_advantage", 0.0)), -2.0, 2.0)
    conflict_degree = _clip(_finite(getattr(candidate, "conflict_degree", 0.0)), 0.0, 20.0)
    action_id = int(_finite(getattr(candidate, "action_id", 0), 0.0))

    features = [
        1.0,
        slack / float(max_steps),
        _clip((80.0 - slack) / 80.0, 0.0, 1.0),
        path_len / float(max_steps),
        wait_streak / 50.0,
        rl_advantage / 2.0,
        conflict_degree / 20.0,
        float(bool(getattr(candidate, "occupied_target", False))),
        float(action_id == 1),
        float(action_id == 2),
        float(action_id == 3),
        source_row,
        source_col,
        target_row,
        target_col,
        route_edges,
        route_cells,
        _clip(float(elapsed_steps), 0.0, float(max_steps)) / float(max_steps),
        _clip(float(num_agents), 0.0, 200.0) / 200.0,
        _clip(float(max_steps), 0.0, 5000.0) / 5000.0,
        scene_id(runtime_context.get().scene),
    ]
    if len(features) != SEQUENCE_DISPATCH_FEATURE_DIM:
        raise ValueError(
            f"sequence feature dimension drift: {len(features)} != "
            f"{SEQUENCE_DISPATCH_FEATURE_DIM}"
        )
    return [_finite(value, 0.0) for value in features]


def has_real_sequence_conflict(candidates: Iterable[Any]) -> bool:
    candidates = list(candidates)
    for left_index, left in enumerate(candidates):
        for right in candidates[left_index + 1 :]:
            if dispatch_pair_conflict(left, right).is_real:
                return True
    return False


def sequence_dispatch_example(
    env: Any,
    candidates: Iterable[Any],
    priority_key: Callable[[Any], Any],
    *,
    seed: int | None = None,
    step: int | None = None,
    max_candidates: int = 96,
    require_conflict: bool = True,
) -> dict[str, Any] | None:
    candidates = list(candidates)
    if len(candidates) < 2:
        return None
    if require_conflict and not has_real_sequence_conflict(candidates):
        return None

    teacher_ordered = sorted(candidates, key=priority_key)
    teacher_ordered = teacher_ordered[:max_candidates]
    handle_set = {int(getattr(candidate, "handle")) for candidate in teacher_ordered}
    tokens = sorted(
        [candidate for candidate in candidates if int(getattr(candidate, "handle")) in handle_set],
        key=lambda candidate: int(getattr(candidate, "handle")),
    )
    if len(tokens) < 2:
        return None

    rank_by_handle = {
        int(getattr(candidate, "handle")): rank
        for rank, candidate in enumerate(teacher_ordered)
    }
    handles = [int(getattr(candidate, "handle")) for candidate in tokens]
    ranks = [int(rank_by_handle[handle]) for handle in handles]
    n_tokens = len(tokens)
    teacher_scores = [
        (float(n_tokens - 1 - rank) / float(max(1, n_tokens - 1)))
        for rank in ranks
    ]
    return {
        "seed": seed,
        "step": step,
        "scene": runtime_context.get().scene,
        "num_agents": _num_agents(env),
        "max_steps": _max_steps(env),
        "handles": handles,
        "features": [sequence_candidate_features(env, candidate) for candidate in tokens],
        "teacher_ranks": ranks,
        "teacher_scores": teacher_scores,
        "teacher_order": [int(getattr(candidate, "handle")) for candidate in teacher_ordered],
        "feature_names": list(SEQUENCE_DISPATCH_FEATURE_NAMES),
    }


class SequenceDispatchTransformer:
    @staticmethod
    def build(
        input_dim: int = SEQUENCE_DISPATCH_FEATURE_DIM,
        hidden_size: int = 96,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.05,
    ) -> Any:
        import torch

        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.LayerNorm(hidden_size),
                )
                layer = torch.nn.TransformerEncoderLayer(
                    d_model=hidden_size,
                    nhead=num_heads,
                    dim_feedforward=hidden_size * 4,
                    dropout=dropout,
                    batch_first=True,
                    activation="gelu",
                    norm_first=True,
                )
                self.encoder = torch.nn.TransformerEncoder(layer, num_layers=num_layers)
                self.score = torch.nn.Linear(hidden_size, 1)

            def forward(self, x: Any, padding_mask: Any | None = None) -> Any:
                hidden = self.input(x)
                hidden = self.encoder(hidden, src_key_padding_mask=padding_mask)
                return self.score(hidden).squeeze(-1)

        return Model()


class SequenceDispatchRanker:
    """Optional sequence-level BC ranker over MAPF/SIPP candidates.

    This is intentionally disabled by default. It only returns priority offsets,
    never direct Flatland actions.
    """

    def __init__(self, checkpoint_path: str | None = None):
        self.enabled = _env_bool("ECML_SEQUENCE_DISPATCH_ENABLED", False)
        self.scenes = {
            scene.strip()
            for scene in os.environ.get("ECML_SEQUENCE_DISPATCH_SCENES", "").split(",")
            if scene.strip()
        }
        self.min_agents = _env_int("ECML_SEQUENCE_DISPATCH_MIN_AGENTS", 70)
        self.weight = _env_float("ECML_SEQUENCE_DISPATCH_WEIGHT", 1.0)
        self.max_candidates = _env_int("ECML_SEQUENCE_DISPATCH_MAX_CANDIDATES", 96)
        default_path = (
            Path(__file__).resolve().parent
            / "models"
            / "ecml_sequence_dispatch_transformer_v1.pt"
        )
        self.checkpoint_path = Path(
            checkpoint_path
            or os.environ.get("ECML_SEQUENCE_DISPATCH_MODEL", str(default_path))
        )
        self._load_attempted = False
        self._model: Any | None = None
        self._torch: Any | None = None
        self.last_stats: dict[str, Any] = {}

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

            checkpoint = torch.load(str(self.checkpoint_path), map_location="cpu")
            input_dim = int(checkpoint.get("input_dim", SEQUENCE_DISPATCH_FEATURE_DIM))
            hidden_size = int(checkpoint.get("hidden_size", 96))
            num_layers = int(checkpoint.get("num_layers", 2))
            num_heads = int(checkpoint.get("num_heads", 4))
            if input_dim != SEQUENCE_DISPATCH_FEATURE_DIM:
                self.last_stats = {"status": "incompatible_model", "input_dim": input_dim}
                return
            model = SequenceDispatchTransformer.build(
                input_dim=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_heads=num_heads,
                dropout=0.0,
            )
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self._torch = torch
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

    def _should_run(self, env: Any, candidates: list[Any]) -> bool:
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
        if len(candidates) < 2:
            self.last_stats = {"status": "too_few_candidates"}
            return False
        if not has_real_sequence_conflict(candidates):
            self.last_stats = {"status": "no_conflicts"}
            return False
        self._load_model()
        return self._model is not None and self._torch is not None

    def priority_scores(self, env: Any, candidates: Iterable[Any]) -> dict[int, float]:
        candidates = sorted(
            list(candidates)[: self.max_candidates],
            key=lambda candidate: int(getattr(candidate, "handle")),
        )
        if not self._should_run(env, candidates):
            return {}
        if self._model is None or self._torch is None:
            return {}
        try:
            features = [
                sequence_candidate_features(env, candidate)
                for candidate in candidates
            ]
            torch = self._torch
            with torch.no_grad():
                x = torch.tensor([features], dtype=torch.float32)
                scores = self._model(x)[0].cpu().tolist()
        except Exception as exc:
            self.last_stats = {"status": "score_failed", "error": str(exc)}
            return {}

        mean_score = sum(float(score) for score in scores) / float(max(1, len(scores)))
        result = {
            int(getattr(candidate, "handle")): self.weight * (float(score) - mean_score)
            for candidate, score in zip(candidates, scores)
        }
        self.last_stats = {"status": "scored", "num_candidates": len(candidates)}
        return {handle: score for handle, score in result.items() if score != 0.0}


def write_sequence_example(handle: Any, example: dict[str, Any]) -> None:
    handle.write(json.dumps(example, sort_keys=True))
    handle.write("\n")
