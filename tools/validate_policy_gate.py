#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from tools.compare_policies import compare_seed, summarize
from tools.evaluate_sampled import DEFAULT_BASE_STATE, DEFAULT_OBS_BUILDER, DEFAULT_REWARDS, repo_root


def parse_seed_blocks(values: list[str] | None) -> list[int]:
    if not values:
        return []
    seeds: list[int] = []
    for value in values:
        for part in value.split(","):
            token = part.strip()
            if not token:
                continue
            if ":" in token:
                start_text, count_text = token.split(":", maxsplit=1)
                start = int(start_text)
                count = int(count_text)
                if count < 0:
                    raise argparse.ArgumentTypeError(f"Negative seed count: {token}")
                seeds.extend(range(start, start + count))
            elif "-" in token:
                start_text, end_text = token.split("-", maxsplit=1)
                start = int(start_text)
                end = int(end_text)
                if end < start:
                    raise argparse.ArgumentTypeError(f"Seed range ends before start: {token}")
                seeds.extend(range(start, end + 1))
            else:
                seeds.append(int(token))
    return list(dict.fromkeys(seeds))


def failed_reasons(summary: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons = []
    epsilon = args.epsilon
    if int(summary["success_losses"]) > args.max_success_losses:
        reasons.append(
            "success_losses "
            f"{summary['success_losses']} > {args.max_success_losses}"
        )
    if int(summary["reward_losses"]) > args.max_reward_losses:
        reasons.append(
            f"reward_losses {summary['reward_losses']} > {args.max_reward_losses}"
        )
    if float(summary["success_delta_mean"]) < args.min_success_delta_mean - epsilon:
        reasons.append(
            "success_delta_mean "
            f"{summary['success_delta_mean']:.9g} < {args.min_success_delta_mean:.9g}"
        )
    if float(summary["reward_delta_mean"]) < args.min_reward_delta_mean - epsilon:
        reasons.append(
            "reward_delta_mean "
            f"{summary['reward_delta_mean']:.9g} < {args.min_reward_delta_mean:.9g}"
        )
    return reasons


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
    rows: list[dict[str, Any]],
    reasons: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(
            {
                "passed": not reasons,
                "failure_reasons": reasons,
                "summary": summary,
                "success_regressions": [
                    row for row in rows if float(row["success_delta"]) < -1e-9
                ],
                "success_gains": [
                    row for row in rows if float(row["success_delta"]) > 1e-9
                ],
                "rows": rows,
            },
            handle,
            indent=2,
        )
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a seedwise validation gate for a candidate policy checkpoint. "
            "The command exits with status 1 when the gate fails."
        )
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--baseline-policy", default="submission.rerank_policy.MyPolicy")
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--candidate-policy", default="submission.rerank_policy.MyPolicy")
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=["100-129"],
        help=(
            "Seed blocks to evaluate. Accepts inclusive ranges like 100-129, "
            "count ranges like 100:30, comma-separated lists, or single seeds."
        ),
    )
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--max-success-losses", type=int, default=0)
    parser.add_argument("--max-reward-losses", type=int, default=10**9)
    parser.add_argument("--min-success-delta-mean", type=float, default=0.0)
    parser.add_argument("--min-reward-delta-mean", type=float, default=0.0)
    parser.add_argument("--epsilon", type=float, default=1e-9)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_blocks(args.blocks)
    if not seeds:
        raise ValueError("No validation seeds selected")
    args.seed = seeds[0]

    rows = []
    for seed in seeds:
        row = compare_seed(args, seed)
        rows.append(row)
        if not args.quiet:
            print(
                f"seed={seed} reward_delta={row['reward_delta']:.6g} "
                f"success_delta={row['success_delta']:.6g} "
                f"baseline={row['baseline_reward']:.6g}/{row['baseline_success']:.6g} "
                f"candidate={row['candidate_reward']:.6g}/{row['candidate_success']:.6g}",
                flush=True,
            )

    summary = summarize(rows)
    reasons = failed_reasons(summary, args)
    print(
        "\nGate summary: "
        f"episodes={summary['episodes']} "
        f"baseline={summary['baseline_reward_mean']:.6g}/"
        f"{summary['baseline_success_mean']:.6g} "
        f"candidate={summary['candidate_reward_mean']:.6g}/"
        f"{summary['candidate_success_mean']:.6g} "
        f"reward={summary['reward_wins']}/{summary['reward_losses']}/"
        f"{summary['reward_ties']} "
        f"success={summary['success_wins']}/{summary['success_losses']}/"
        f"{summary['success_ties']}"
    )
    if reasons:
        print("Gate: FAIL")
        for reason in reasons:
            print(f"  {reason}")
    else:
        print("Gate: PASS")

    success_regressions = [
        row for row in rows if float(row["success_delta"]) < -1e-9
    ]
    if success_regressions:
        print("Success regressions:")
        for row in success_regressions:
            print(
                f"  seed={row['seed']} reward_delta={row['reward_delta']:.6g} "
                f"success_delta={row['success_delta']:.6g}"
            )

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    if args.output_json is not None:
        write_json(args.output_json, summary, rows, reasons)
    return 1 if reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
