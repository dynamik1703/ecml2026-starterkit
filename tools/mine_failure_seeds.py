#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Any

from tools.evaluate_sampled import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_REWARDS,
    repo_root,
    run_episode,
)


DEFAULT_POLICY = "submission.rerank_policy.MyPolicy"


def episode_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        base_state_pkl=args.base_state_pkl,
        policy=args.policy,
        policy_checkpoint=args.policy_checkpoint,
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


def seed_list(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(item) for item in args.seeds.split(",") if item.strip()]
    return [args.seed + index for index in range(args.episodes)]


def failed_ids(failed_agents: list[dict[str, Any]]) -> str:
    return ",".join(str(agent["agent_id"]) for agent in failed_agents)


def flatten_row(row: dict[str, Any]) -> dict[str, Any]:
    failed_agents = row.get("failed_agents", [])
    stationary_tails = [
        int(agent.get("stationary_tail") or 0)
        for agent in failed_agents
    ]
    missed_by = [
        int(agent["missed_by"])
        for agent in failed_agents
        if agent.get("missed_by") is not None
    ]
    return {
        "seed": int(row["seed"]),
        "scene": row["scene"],
        "num_agents": int(row["num_agents"]),
        "line_length": int(row["line_length"]),
        "env_time": int(row["env_time"]),
        "max_episode_steps": int(row["max_episode_steps"]),
        "success_rate": float(row["success_rate"]),
        "normalized_reward": float(row["normalized_reward"]),
        "failed_agents": len(failed_agents),
        "failed_agent_ids": failed_ids(failed_agents),
        "max_stationary_tail": max(stationary_tails) if stationary_tails else 0,
        "mean_stationary_tail": (
            sum(stationary_tails) / len(stationary_tails)
            if stationary_tails else 0.0
        ),
        "max_missed_by": max(missed_by) if missed_by else 0,
        "mean_missed_by": sum(missed_by) / len(missed_by) if missed_by else 0.0,
    }


def is_candidate(flat: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.selection_mode == "full-success-low-reward":
        if float(flat["success_rate"]) < 1.0:
            return False
        if int(flat["failed_agents"]) > 0:
            return False
        if (
            args.max_reward is not None
            and float(flat["normalized_reward"]) > args.max_reward
        ):
            return False
        return True

    if float(flat["success_rate"]) < args.success_threshold:
        return True
    if int(flat["failed_agents"]) >= args.min_failed_agents:
        return True
    if (
        args.max_reward is not None
        and float(flat["normalized_reward"]) <= args.max_reward
    ):
        return True
    return False


def ranking_key(
    flat: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[float, float, float, float]:
    if args.selection_mode == "full-success-low-reward":
        return (
            float(flat["normalized_reward"]),
            -float(flat["env_time"]),
            -float(flat["max_episode_steps"]),
            float(flat["seed"]),
        )
    return (
        -float(flat["failed_agents"]),
        float(flat["success_rate"]),
        float(flat["normalized_reward"]),
        -float(flat["max_stationary_tail"]),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(
    path: Path,
    summary: dict[str, Any],
    flat_rows: list[dict[str, Any]],
    detailed_rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(
            {
                "summary": summary,
                "rows": flat_rows,
                "details": detailed_rows,
            },
            handle,
            indent=2,
        )
        handle.write("\n")


def counterfactual_command(args: argparse.Namespace, seeds: list[int]) -> str:
    output = args.counterfactual_output or Path(
        f"/private/tmp/ecml_counterfactual_failure_{seeds[0]}_{len(seeds)}.csv"
    )
    parts = [
        "env",
        "PYTHONPATH=.",
        "PYTHONPYCACHEPREFIX=/private/tmp/ecml_pycache",
        "MPLCONFIGDIR=/private/tmp/ecml_mpl",
        ".venv/bin/python",
        "tools/counterfactual_decision_eval.py",
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--policy",
        args.policy,
        "--num-agents",
        str(args.num_agents),
        "--line-length",
        str(args.line_length),
        "--forced-actions",
        args.forced_actions,
        "--max-decisions-per-seed",
        str(args.max_decisions_per_seed),
        "--max-alternatives-per-decision",
        str(args.max_alternatives_per_decision),
        "--output-csv",
        str(output),
    ]
    if args.scene:
        parts.extend(["--scene", args.scene])
    if args.policy_checkpoint is not None:
        parts.extend(["--policy-checkpoint", str(args.policy_checkpoint)])
    if args.output_json is not None and args.selection_mode == "failures":
        parts.extend(["--focus-failures-json", str(args.output_json)])
    if (
        args.focus_window_before_stationary is not None
        and args.selection_mode == "failures"
    ):
        parts.extend(
            [
                "--focus-window-before-stationary",
                str(args.focus_window_before_stationary),
                "--focus-window-after-stationary",
                str(args.focus_window_after_stationary),
            ]
        )
    if (
        args.focus_window_before_deadline is not None
        and args.selection_mode == "failures"
    ):
        parts.extend(
            [
                "--focus-window-before-deadline",
                str(args.focus_window_before_deadline),
                "--focus-window-after-deadline",
                str(args.focus_window_after_deadline),
            ]
        )
    return " ".join(shlex.quote(part) for part in parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find failure-rich seeds for targeted counterfactual mining."
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--seeds",
        help="Comma-separated exact seed list. Overrides --seed/--episodes when set.",
    )
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--success-threshold", type=float, default=1.0)
    parser.add_argument("--min-failed-agents", type=int, default=1)
    parser.add_argument(
        "--max-reward",
        type=float,
        help=(
            "In failure mode, also include seeds whose normalized reward is at "
            "or below this value. In full-success-low-reward mode, filter "
            "selected successful seeds to this maximum reward."
        ),
    )
    parser.add_argument(
        "--selection-mode",
        choices=["failures", "full-success-low-reward"],
        default="failures",
        help=(
            "Select failure-rich seeds, or completed episodes with the lowest "
            "normalized reward for hard-negative counterfactual mining."
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--print-counterfactual-command",
        action="store_true",
        help="Print a counterfactual_decision_eval command for the selected seeds.",
    )
    parser.add_argument("--counterfactual-output", type=Path)
    parser.add_argument("--forced-actions", default="LEFT,FORWARD,RIGHT")
    parser.add_argument("--max-decisions-per-seed", type=int, default=8)
    parser.add_argument("--max-alternatives-per-decision", type=int, default=2)
    parser.add_argument(
        "--focus-window-before-stationary",
        type=int,
        help=(
            "Include this failed-agent time-window argument in the generated "
            "counterfactual command."
        ),
    )
    parser.add_argument(
        "--focus-window-after-stationary",
        type=int,
        default=30,
        help="After-window used in generated counterfactual commands.",
    )
    parser.add_argument(
        "--focus-window-before-deadline",
        type=int,
        help=(
            "Include this latest-arrival deadline window argument in generated "
            "counterfactual commands."
        ),
    )
    parser.add_argument(
        "--focus-window-after-deadline",
        type=int,
        default=30,
        help="Deadline after-window used in generated counterfactual commands.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    detailed_rows = []
    flat_rows = []
    candidates = []

    for seed in seed_list(args):
        row = run_episode(episode_args(args), seed)
        flat = flatten_row(row)
        detailed_rows.append(row)
        flat_rows.append(flat)
        if is_candidate(flat, args):
            candidates.append(flat)
        print(
            "seed={seed} reward={normalized_reward:.6g} success={success_rate:.6g} "
            "failed={failed_agents} failed_ids={failed_agent_ids}".format(**flat),
            flush=True,
        )

    ranked = sorted(candidates, key=lambda flat: ranking_key(flat, args))
    selected = ranked[: max(0, args.top_k)]
    summary = {
        "policy": args.policy,
        "selection_mode": args.selection_mode,
        "episodes": len(flat_rows),
        "candidate_seeds": len(candidates),
        "selected_seeds": len(selected),
        "reward_mean": (
            sum(float(row["normalized_reward"]) for row in flat_rows) / len(flat_rows)
            if flat_rows else 0.0
        ),
        "success_rate_mean": (
            sum(float(row["success_rate"]) for row in flat_rows) / len(flat_rows)
            if flat_rows else 0.0
        ),
    }

    print(
        "\nSummary: "
        f"episodes={summary['episodes']} "
        f"reward_mean={summary['reward_mean']:.6g} "
        f"success_rate_mean={summary['success_rate_mean']:.6g} "
        f"candidate_seeds={summary['candidate_seeds']} "
        f"selected_seeds={summary['selected_seeds']}"
    )
    if selected:
        selected_label = (
            "failure seeds"
            if args.selection_mode == "failures"
            else "full-success low-reward seeds"
        )
        print(f"\nSelected {selected_label}:")
        print("seed,success_rate,normalized_reward,failed_agents,failed_agent_ids")
        for row in selected:
            print(
                f"{row['seed']},{row['success_rate']:.6g},"
                f"{row['normalized_reward']:.6g},{row['failed_agents']},"
                f"{row['failed_agent_ids']}"
            )
        print("seeds=" + ",".join(str(row["seed"]) for row in selected))

    if args.print_counterfactual_command and selected:
        print("\nCounterfactual command:")
        print(counterfactual_command(args, [int(row["seed"]) for row in selected]))

    if args.output_csv is not None:
        write_csv(args.output_csv, flat_rows)
    if args.output_json is not None:
        write_json(args.output_json, summary, flat_rows, detailed_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
