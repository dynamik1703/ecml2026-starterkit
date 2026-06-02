#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluate_sampled import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_POLICY,
    DEFAULT_REWARDS,
    repo_root,
    run_episode,
)


@dataclass(frozen=True)
class Candidate:
    name: str
    policy: str
    checkpoint: Path | None = None


def parse_candidate(value: str) -> Candidate:
    if "=" in value:
        name, spec = value.split("=", maxsplit=1)
    else:
        spec = value
        name = spec.rsplit(".", maxsplit=1)[-1]

    if "@" in spec:
        policy, checkpoint = spec.rsplit("@", maxsplit=1)
        return Candidate(name=name, policy=policy, checkpoint=Path(checkpoint))
    return Candidate(name=name, policy=spec)


def benchmark_args(
    args: argparse.Namespace,
    candidate: Candidate,
    seed: int,
    num_agents: int,
    line_length: int,
    scene: str | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        base_state_pkl=args.base_state_pkl,
        policy=candidate.policy,
        policy_checkpoint=candidate.checkpoint,
        obs_builder=args.obs_builder,
        rewards=args.rewards,
        episodes=1,
        seed=seed,
        num_agents=num_agents,
        line_length=line_length,
        scene=scene,
        output_json=None,
        output_csv=None,
        agent_details=False,
    )


def flatten_row(
    candidate: Candidate,
    config: argparse.Namespace,
    row: dict[str, Any],
) -> dict[str, Any]:
    failed_agents = row.pop("failed_agents", [])
    return {
        "candidate": candidate.name,
        "policy": candidate.policy,
        "policy_checkpoint": str(candidate.checkpoint) if candidate.checkpoint else "",
        "seed": row["seed"],
        "num_agents_config": config.num_agents,
        "line_length_config": config.line_length,
        "scene_config": config.scene or "scene_5",
        "env_time": row["env_time"],
        "success_rate": row["success_rate"],
        "normalized_reward": row["normalized_reward"],
        "num_agents": row["num_agents"],
        "max_episode_steps": row["max_episode_steps"],
        "failed_agents": len(failed_agents),
    }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["candidate"],
            row["policy"],
            row["policy_checkpoint"],
            row["num_agents_config"],
            row["line_length_config"],
            row["scene_config"],
        )
        groups[key].append(row)

    summaries = []
    for (
        candidate,
        policy,
        checkpoint,
        num_agents,
        line_length,
        scene,
    ), group_rows in groups.items():
        rewards = [float(row["normalized_reward"]) for row in group_rows]
        successes = [float(row["success_rate"]) for row in group_rows]
        summaries.append(
            {
                "candidate": candidate,
                "policy": policy,
                "policy_checkpoint": checkpoint,
                "num_agents": num_agents,
                "line_length": line_length,
                "scene": scene,
                "episodes": len(group_rows),
                "reward_mean": sum(rewards) / len(rewards),
                "reward_min": min(rewards),
                "success_rate_mean": sum(successes) / len(successes),
                "success_rate_min": min(successes),
                "failed_agents_total": sum(int(row["failed_agents"]) for row in group_rows),
            }
        )

    return sorted(
        summaries,
        key=lambda row: (row["reward_mean"], row["success_rate_mean"]),
        reverse=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sampled Flatland benchmarks across policy/scenario matrices."
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        help=(
            "Candidate as name=module.Class or name=module.Class@checkpoint.pt. "
            "Can be repeated."
        ),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[10, 11, 12, 13, 14])
    parser.add_argument("--num-agents", type=int, nargs="+", default=[6])
    parser.add_argument("--line-lengths", type=int, nargs="+", default=[2])
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
        default=["scene_5"],
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    candidates = args.candidate or [Candidate("default", DEFAULT_POLICY)]
    rows = []

    for candidate in candidates:
        for num_agents in args.num_agents:
            for line_length in args.line_lengths:
                for scene_name in args.scenes:
                    scene = None if scene_name == "scene_5" else scene_name
                    for seed in args.seeds:
                        config = benchmark_args(
                            args,
                            candidate,
                            seed,
                            num_agents,
                            line_length,
                            scene,
                        )
                        episode_row = run_episode(config, seed)
                        flat = flatten_row(candidate, config, episode_row)
                        rows.append(flat)
                        print(
                            f"{flat['candidate']} seed={seed} "
                            f"agents={num_agents} line={line_length} scene={scene_name} "
                            f"reward={flat['normalized_reward']:.6g} "
                            f"success={flat['success_rate']:.6g}",
                            flush=True,
                        )

    summaries = summarize(rows)
    print("\nSummary:")
    print("candidate,agents,line,scene,episodes,reward_mean,success_rate_mean,reward_min,success_min")
    for summary in summaries:
        print(
            f"{summary['candidate']},{summary['num_agents']},"
            f"{summary['line_length']},{summary['scene']},{summary['episodes']},"
            f"{summary['reward_mean']:.6g},{summary['success_rate_mean']:.6g},"
            f"{summary['reward_min']:.6g},{summary['success_rate_min']:.6g}"
        )

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    if args.summary_csv is not None:
        write_csv(args.summary_csv, summaries)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump({"rows": rows, "summaries": summaries}, handle, indent=2)
            handle.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
