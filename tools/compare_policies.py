#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools.evaluate_sampled import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_REWARDS,
    repo_root,
    run_episode,
)


DEFAULT_BASELINE_POLICY = "submission.hybrid_policy.MyPolicy"
DEFAULT_CANDIDATE_POLICY = "submission.rerank_policy.MyPolicy"


def episode_args(
    args: argparse.Namespace,
    policy: str,
    checkpoint: Path | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        base_state_pkl=args.base_state_pkl,
        policy=policy,
        policy_checkpoint=checkpoint,
        obs_builder=args.obs_builder,
        rewards=args.rewards,
        episodes=1,
        seed=args.seed,
        num_agents=args.num_agents,
        line_length=args.line_length,
        scene=args.scene,
        output_json=None,
        output_csv=None,
        agent_details=False,
    )


def failed_agent_ids(row: dict[str, Any]) -> str:
    return ",".join(str(agent["agent_id"]) for agent in row.get("failed_agents", []))


def compare_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    baseline_args = episode_args(args, args.baseline_policy, args.baseline_checkpoint)
    candidate_args = episode_args(args, args.candidate_policy, args.candidate_checkpoint)
    baseline_row = run_episode(baseline_args, seed)
    candidate_row = run_episode(candidate_args, seed)

    baseline_reward = float(baseline_row["normalized_reward"])
    candidate_reward = float(candidate_row["normalized_reward"])
    baseline_success = float(baseline_row["success_rate"])
    candidate_success = float(candidate_row["success_rate"])
    return {
        "seed": seed,
        "baseline_reward": baseline_reward,
        "candidate_reward": candidate_reward,
        "reward_delta": candidate_reward - baseline_reward,
        "baseline_success": baseline_success,
        "candidate_success": candidate_success,
        "success_delta": candidate_success - baseline_success,
        "baseline_env_time": int(baseline_row["env_time"]),
        "candidate_env_time": int(candidate_row["env_time"]),
        "env_time_delta": int(candidate_row["env_time"]) - int(baseline_row["env_time"]),
        "baseline_failed_agents": len(baseline_row.get("failed_agents", [])),
        "candidate_failed_agents": len(candidate_row.get("failed_agents", [])),
        "failed_agents_delta": (
            len(candidate_row.get("failed_agents", []))
            - len(baseline_row.get("failed_agents", []))
        ),
        "baseline_failed_agent_ids": failed_agent_ids(baseline_row),
        "candidate_failed_agent_ids": failed_agent_ids(candidate_row),
        "num_agents": int(baseline_row["num_agents"]),
        "max_episode_steps": int(baseline_row["max_episode_steps"]),
        "line_length": int(baseline_row["line_length"]),
        "scene": baseline_row["scene"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "episodes": 0,
            "baseline_reward_mean": 0.0,
            "candidate_reward_mean": 0.0,
            "reward_delta_mean": 0.0,
            "baseline_success_mean": 0.0,
            "candidate_success_mean": 0.0,
            "success_delta_mean": 0.0,
            "reward_wins": 0,
            "reward_losses": 0,
            "reward_ties": 0,
            "success_wins": 0,
            "success_losses": 0,
            "success_ties": 0,
        }

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    return {
        "episodes": len(rows),
        "baseline_reward_mean": mean("baseline_reward"),
        "candidate_reward_mean": mean("candidate_reward"),
        "reward_delta_mean": mean("reward_delta"),
        "baseline_success_mean": mean("baseline_success"),
        "candidate_success_mean": mean("candidate_success"),
        "success_delta_mean": mean("success_delta"),
        "reward_wins": sum(int(row["reward_delta"] > 1e-9) for row in rows),
        "reward_losses": sum(int(row["reward_delta"] < -1e-9) for row in rows),
        "reward_ties": sum(int(abs(row["reward_delta"]) <= 1e-9) for row in rows),
        "success_wins": sum(int(row["success_delta"] > 1e-9) for row in rows),
        "success_losses": sum(int(row["success_delta"] < -1e-9) for row in rows),
        "success_ties": sum(int(abs(row["success_delta"]) <= 1e-9) for row in rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, summary: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=2)
        handle.write("\n")


def print_rows(title: str, rows: list[dict[str, Any]]) -> None:
    print(f"\n{title}:")
    print(
        "seed,reward_delta,success_delta,baseline_reward,candidate_reward,"
        "baseline_failed,candidate_failed"
    )
    for row in rows:
        print(
            f"{row['seed']},{row['reward_delta']:.6g},{row['success_delta']:.6g},"
            f"{row['baseline_reward']:.6g},{row['candidate_reward']:.6g},"
            f"{row['baseline_failed_agents']},{row['candidate_failed_agents']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two policies seed-by-seed on sampled ECML-style scenarios."
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--baseline-policy", default=DEFAULT_BASELINE_POLICY)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--candidate-policy", default=DEFAULT_CANDIDATE_POLICY)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = []
    for index in range(args.episodes):
        seed = args.seed + index
        row = compare_seed(args, seed)
        rows.append(row)
        print(
            f"seed={seed} reward_delta={row['reward_delta']:.6g} "
            f"success_delta={row['success_delta']:.6g} "
            f"baseline={row['baseline_reward']:.6g}/{row['baseline_success']:.6g} "
            f"candidate={row['candidate_reward']:.6g}/{row['candidate_success']:.6g}",
            flush=True,
        )

    summary = summarize(rows)
    print(
        "\nSummary: "
        f"episodes={summary['episodes']} "
        f"baseline_reward_mean={summary['baseline_reward_mean']:.6g} "
        f"candidate_reward_mean={summary['candidate_reward_mean']:.6g} "
        f"reward_delta_mean={summary['reward_delta_mean']:.6g} "
        f"baseline_success_mean={summary['baseline_success_mean']:.6g} "
        f"candidate_success_mean={summary['candidate_success_mean']:.6g} "
        f"success_delta_mean={summary['success_delta_mean']:.6g}"
    )
    print(
        "Wins/losses: "
        f"reward={summary['reward_wins']}/{summary['reward_losses']}/"
        f"{summary['reward_ties']} "
        f"success={summary['success_wins']}/{summary['success_losses']}/"
        f"{summary['success_ties']} "
        "(wins/losses/ties)"
    )

    top = max(0, args.top_k)
    if top:
        print_rows(
            "Largest candidate gains",
            sorted(rows, key=lambda row: row["reward_delta"], reverse=True)[:top],
        )
        print_rows(
            "Largest candidate regressions",
            sorted(rows, key=lambda row: row["reward_delta"])[:top],
        )

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    if args.output_json is not None:
        write_json(args.output_json, summary, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
