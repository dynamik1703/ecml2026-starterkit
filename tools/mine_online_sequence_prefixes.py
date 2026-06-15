#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

from tools.compare_policies import compare_seed, summarize
from tools.convert_sequence_trace_to_prefix_rows import (
    event_detail_from_trace,
    output_row,
    read_trace_rows,
    write_csv as write_prefix_csv,
)
from tools.evaluate_sampled import DEFAULT_BASE_STATE, DEFAULT_REWARDS, repo_root
from tools.evaluate_success_rescue_classifier import json_default
from tools.policy_scoreboard import parse_seed_blocks


DEFAULT_BASELINE_POLICY = "submission.rerank_policy.MyPolicy"
DEFAULT_BASELINE_CHECKPOINT = (
    repo_root()
    / "submission"
    / "models"
    / "ecml_aux_bc_conflict_neg_currentinit_ppo_v4b.pt"
)
DEFAULT_CANDIDATE_POLICY = "submission.sequence_success_policy.MyPolicy"
DEFAULT_OBS_BUILDER = (
    "submission.my_observation_builder.MyActionConflictObservationBuilder"
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def changed(row: dict[str, Any]) -> bool:
    return (
        abs(float(row.get("reward_delta", 0.0))) > 1e-9
        or abs(float(row.get("success_delta", 0.0))) > 1e-9
    )


def compare_args(
    args: argparse.Namespace,
    line_length: int,
    scene: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        base_state_pkl=args.base_state_pkl,
        baseline_policy=args.baseline_policy,
        baseline_checkpoint=args.baseline_checkpoint,
        candidate_policy=args.candidate_policy,
        candidate_checkpoint=args.candidate_checkpoint,
        obs_builder=args.obs_builder,
        rewards=args.rewards,
        seed=0,
        num_agents=args.num_agents,
        line_length=line_length,
        scene=None if scene == "scene_5" else scene,
    )


def trace_path_for(args: argparse.Namespace, line_length: int, scene: str) -> Path:
    return args.output_prefix.with_name(
        f"{args.output_prefix.name}_line{line_length}_{scene}_trace.jsonl"
    )


def compare_json_path_for(args: argparse.Namespace, line_length: int, scene: str) -> Path:
    return args.output_prefix.with_name(
        f"{args.output_prefix.name}_line{line_length}_{scene}_compare.json"
    )


def compare_csv_path_for(args: argparse.Namespace, line_length: int, scene: str) -> Path:
    return args.output_prefix.with_name(
        f"{args.output_prefix.name}_line{line_length}_{scene}_compare.csv"
    )


def mine_config(
    args: argparse.Namespace,
    seeds: list[int],
    line_length: int,
    scene: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    config_args = compare_args(args, line_length, scene)
    trace_path = trace_path_for(args, line_length, scene)
    compare_json_path = compare_json_path_for(args, line_length, scene)
    compare_csv_path = compare_csv_path_for(args, line_length, scene)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()

    old_trace_path = os.environ.get("ECML_SEQUENCE_TRACE_PATH")
    old_trace_all = os.environ.get("ECML_SEQUENCE_TRACE_ALL")
    os.environ["ECML_SEQUENCE_TRACE_PATH"] = str(trace_path)
    os.environ["ECML_SEQUENCE_TRACE_ALL"] = "1" if args.trace_all else "0"
    try:
        compare_rows = []
        for seed in seeds:
            row = compare_seed(config_args, seed)
            compare_rows.append(row)
            if not args.quiet:
                print(
                    f"seed={seed} scene={scene} line={line_length} "
                    f"reward_delta={row['reward_delta']:.6g} "
                    f"success_delta={row['success_delta']:.6g}",
                    flush=True,
                )
    finally:
        if old_trace_path is None:
            os.environ.pop("ECML_SEQUENCE_TRACE_PATH", None)
        else:
            os.environ["ECML_SEQUENCE_TRACE_PATH"] = old_trace_path
        if old_trace_all is None:
            os.environ.pop("ECML_SEQUENCE_TRACE_ALL", None)
        else:
            os.environ["ECML_SEQUENCE_TRACE_ALL"] = old_trace_all

    compare_summary = summarize(compare_rows)
    compare_json_path.parent.mkdir(parents=True, exist_ok=True)
    with compare_json_path.open("w") as handle:
        json.dump(
            {"summary": compare_summary, "rows": compare_rows},
            handle,
            indent=2,
            default=json_default,
        )
        handle.write("\n")
    write_csv(compare_csv_path, compare_rows)

    compare_by_seed = {int(row["seed"]): row for row in compare_rows}
    traces_by_seed: dict[int, list[dict[str, Any]]] = {}
    if trace_path.exists():
        for trace_row in read_trace_rows([trace_path]):
            if not trace_row.get("accepted", False):
                continue
            seed = int(trace_row["seed"])
            traces_by_seed.setdefault(seed, []).append(trace_row)

    prefix_rows = []
    for seed, trace_rows in sorted(traces_by_seed.items()):
        compare_row = compare_by_seed.get(seed)
        if compare_row is None:
            continue
        if args.changed_only and not changed(compare_row):
            continue
        trace_rows = sorted(
            trace_rows,
            key=lambda row: (
                int(row.get("env_time", 0)),
                int(row.get("agent_id", 0)),
            ),
        )
        events = [
            event_detail_from_trace(row, prefix_index=index + 1)
            for index, row in enumerate(trace_rows)
        ]
        prefix_indices = (
            [len(events)] if args.final_prefix_only else range(1, len(events) + 1)
        )
        for prefix_len in prefix_indices:
            prefix_rows.append(
                output_row(
                    compare_row=compare_row,
                    events=events[:prefix_len],
                    final_trace_row=trace_rows[prefix_len - 1],
                )
            )

    metadata = {
        "line_length": line_length,
        "scene": scene,
        "trace_jsonl": str(trace_path),
        "compare_json": str(compare_json_path),
        "compare_csv": str(compare_csv_path),
        "compare_summary": compare_summary,
        "changed_seeds": [row["seed"] for row in compare_rows if changed(row)],
        "accepted_trace_seeds": sorted(traces_by_seed),
        "prefix_rows": len(prefix_rows),
    }
    return compare_rows, prefix_rows, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine online SequenceSuccessPolicy accepted prefixes with the correct "
            "Rerank(v4b)+ActionConflict baseline comparison."
        )
    )
    parser.add_argument("--base-state-pkl", type=Path, default=repo_root() / DEFAULT_BASE_STATE)
    parser.add_argument("--baseline-policy", default=DEFAULT_BASELINE_POLICY)
    parser.add_argument(
        "--baseline-checkpoint",
        type=Path,
        default=DEFAULT_BASELINE_CHECKPOINT,
    )
    parser.add_argument("--candidate-policy", default=DEFAULT_CANDIDATE_POLICY)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--blocks", nargs="+", required=True)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-lengths", nargs="+", type=int, default=[2])
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
        default=["scene_3"],
    )
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--changed-only", action="store_true")
    parser.add_argument("--final-prefix-only", action="store_true")
    parser.add_argument("--trace-all", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_blocks(args.blocks)
    if not seeds:
        raise ValueError("No seeds selected")

    all_compare_rows = []
    all_prefix_rows = []
    configs = []
    for line_length in args.line_lengths:
        for scene in args.scenes:
            compare_rows, prefix_rows, metadata = mine_config(
                args,
                seeds=seeds,
                line_length=line_length,
                scene=scene,
            )
            all_compare_rows.extend(compare_rows)
            all_prefix_rows.extend(prefix_rows)
            configs.append(metadata)

    prefix_categories = Counter(
        str(row.get("outcome_category", "")) for row in all_prefix_rows
    )
    compare_summary = summarize(all_compare_rows)
    output_json = args.output_prefix.with_suffix(".json")
    output_csv = args.output_prefix.with_suffix(".csv")
    output_compare_csv = args.output_prefix.with_name(
        f"{args.output_prefix.name}_compare_all.csv"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w") as handle:
        json.dump(
            {
                "summary": {
                    "seeds": len(seeds),
                    "configs": len(configs),
                    "compare_rows": len(all_compare_rows),
                    "prefix_rows": len(all_prefix_rows),
                    "prefix_categories": dict(prefix_categories),
                    "compare_summary": compare_summary,
                    "changed_only": bool(args.changed_only),
                    "final_prefix_only": bool(args.final_prefix_only),
                },
                "configs": configs,
                "rows": all_prefix_rows,
            },
            handle,
            indent=2,
            default=json_default,
        )
        handle.write("\n")
    write_prefix_csv(output_csv, all_prefix_rows)
    write_csv(output_compare_csv, all_compare_rows)
    print(
        "mined "
        f"compare_rows={len(all_compare_rows)} prefix_rows={len(all_prefix_rows)} "
        f"categories={dict(prefix_categories)} output={output_json}"
    )
    print(
        "compare "
        f"reward={compare_summary['reward_wins']}/{compare_summary['reward_losses']}/"
        f"{compare_summary['reward_ties']} "
        f"success={compare_summary['success_wins']}/{compare_summary['success_losses']}/"
        f"{compare_summary['success_ties']} "
        f"delta={compare_summary['reward_delta_mean']:.6g}/"
        f"{compare_summary['success_delta_mean']:.6g}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
