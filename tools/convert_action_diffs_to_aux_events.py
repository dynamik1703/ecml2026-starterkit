#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


OUTPUT_COLUMNS = (
    "scene",
    "seed",
    "env_time",
    "agent_id",
    "event_kind",
    "forced_applied",
    "forced_action",
    "baseline_action",
    "candidate_action",
    "reward_delta",
    "success_delta",
    "failed_agents_delta",
    "source_path",
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_scene_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"input must be SCENE=PATH, got {value!r}"
        )
    scene, path = value.split("=", 1)
    scene = scene.strip()
    if scene not in {"scene_1", "scene_2", "scene_3", "scene_4", "scene_5"}:
        raise argparse.ArgumentTypeError(f"invalid scene {scene!r}")
    return scene, Path(path)


def compare_lookup(paths: list[Path]) -> dict[tuple[str, int], dict[str, str]]:
    rows: dict[tuple[str, int], dict[str, str]] = {}
    for path in paths:
        for row in read_csv(path):
            scene = row.get("scene", "").strip()
            seed = int(safe_float(row.get("seed")))
            rows[(scene, seed)] = row
    return rows


def convert_rows(
    scene: str,
    path: Path,
    compare_rows: dict[tuple[str, int], dict[str, str]],
) -> list[dict[str, str]]:
    output_rows: list[dict[str, str]] = []
    for row in read_csv(path):
        if str(row.get("has_diff", "")).lower() not in {"true", "1"}:
            continue
        seed = int(safe_float(row.get("seed")))
        outcome = compare_rows.get((scene, seed), {})
        success_delta = safe_float(outcome.get("success_delta"))
        reward_delta = safe_float(outcome.get("reward_delta"))
        num_agents = max(1.0, safe_float(outcome.get("baseline_num_agents"), 6.0))
        failed_agents_delta = max(0.0, -success_delta * num_agents)
        output_rows.append(
            {
                "scene": scene,
                "seed": str(seed),
                "env_time": str(int(safe_float(row.get("env_time")))),
                "agent_id": str(int(safe_float(row.get("agent_id")))),
                "event_kind": "negative_baseline",
                "forced_applied": "True",
                "forced_action": str(int(safe_float(row.get("baseline_action")))),
                "baseline_action": str(int(safe_float(row.get("baseline_action")))),
                "candidate_action": str(int(safe_float(row.get("candidate_action")))),
                "reward_delta": f"{reward_delta:.12g}",
                "success_delta": f"{success_delta:.12g}",
                "failed_agents_delta": f"{failed_agents_delta:.12g}",
                "source_path": str(path),
            }
        )
    return output_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert policy action-diff rows into negative_baseline aux events "
            "for BC/PPO forbid training."
        )
    )
    parser.add_argument(
        "--input",
        action="append",
        type=parse_scene_input,
        required=True,
        help="Scene-qualified diff CSV in the form SCENE=PATH.",
    )
    parser.add_argument(
        "--compare-csv",
        action="append",
        type=Path,
        default=[],
        help="A/B aggregate CSV providing reward_delta/success_delta by scene/seed.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outcomes = compare_lookup(args.compare_csv)
    rows: list[dict[str, str]] = []
    for scene, path in args.input:
        rows.extend(convert_rows(scene, path, outcomes))

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row["scene"] for row in rows)
    print(
        f"wrote {len(rows)} negative_baseline events to {args.output_csv} "
        f"by_scene={dict(counts)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
