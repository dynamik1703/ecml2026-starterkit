#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.analyze_policy_action_diffs import (
    action_name,
    diff_row,
    done_all,
    instantiate_policy,
    make_env,
    policy_actions,
    raw_policy_actions,
)
from tools.evaluate_policy_diff_prefixes import (
    compact_events,
    final_result,
    run_policy_episode,
    selected_prefix_lengths,
    selected_seeds,
    summarize_rows,
)
from tools.evaluate_sampled import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_REWARDS,
    observation_list,
    repo_root,
)
from tools.train_counterfactual_gate import outcome_category


PREFIX_SUFFIXES = (
    "prefix_len",
    "prefix_conflict_agents",
    "prefix_cell_intersections",
    "prefix_first_intersection_step",
    "prefix_min_intersection_eta_gap",
    "prefix_same_direction_intersections",
    "prefix_opposing_direction_intersections",
    "prefix_crossing_direction_intersections",
    "prefix_same_edge_conflicts",
    "prefix_head_on_edge_conflicts",
    "prefix_min_head_on_eta_gap",
    "prefix_min_own_deadline_slack",
    "prefix_min_other_deadline_slack",
    "prefix_min_pair_deadline_slack",
    "prefix_own_deadline_miss_conflicts",
    "prefix_other_deadline_miss_conflicts",
    "prefix_pair_deadline_miss_conflicts",
    "prefix_own_tight_deadline_conflicts",
    "prefix_other_tight_deadline_conflicts",
    "prefix_pair_tight_deadline_conflicts",
)

NUMERIC_EVENT_EXCLUDE = {
    "seed",
    "has_diff",
    "agent_id",
    "baseline_action",
    "candidate_action",
    "baseline_raw_action",
    "candidate_raw_action",
}

NON_NUMERIC_DETAIL_KEYS = {
    "baseline_action_name",
    "baseline_raw_action_name",
    "baseline_target",
    "candidate_action_name",
    "candidate_raw_action_name",
    "candidate_target",
    "position",
    "state",
}


def column_slug(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value).strip("_")


def candidate_source_name(policy: Any, checkpoint: Path | str | None = None) -> str:
    if checkpoint is not None:
        source = Path(checkpoint).stem
    else:
        source = str(policy).rsplit(".", maxsplit=1)[-1]
    return column_slug(source).lower() or "unknown"


def candidate_source_features(
    policy: Any,
    checkpoint: Path | str | None = None,
) -> dict[str, float]:
    source = candidate_source_name(policy, checkpoint)
    return {
        "candidate_source_known": 1.0,
        f"candidate_source_{source}": 1.0,
    }


def safe_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def add_delta_features(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    for suffix in PREFIX_SUFFIXES:
        baseline = safe_float(row.get(f"baseline_{suffix}"))
        candidate = safe_float(row.get(f"candidate_{suffix}"))
        enriched[f"candidate_{suffix}_delta"] = candidate - baseline

    for base_name, candidate_name, output_name in (
        (
            "baseline_future_head_on_risk",
            "candidate_future_head_on_risk",
            "candidate_future_head_on_risk_delta",
        ),
        (
            "baseline_deadline_conflict_penalty",
            "candidate_deadline_conflict_penalty",
            "candidate_deadline_conflict_penalty_delta",
        ),
        (
            "baseline_residual_head_on_min_pair_deadline_slack",
            "candidate_residual_head_on_min_pair_deadline_slack",
            "candidate_residual_head_on_min_pair_deadline_slack_delta",
        ),
        (
            "baseline_corridor_len",
            "candidate_corridor_len",
            "candidate_corridor_len_delta",
        ),
    ):
        enriched[output_name] = safe_float(row.get(candidate_name)) - safe_float(
            row.get(base_name)
        )
    return enriched


def numeric_event_columns(events: list[dict[str, Any]]) -> list[str]:
    columns = sorted({key for event in events for key in event})
    result = []
    for column in columns:
        if column in NUMERIC_EVENT_EXCLUDE or column in NON_NUMERIC_DETAIL_KEYS:
            continue
        values = [safe_float(event.get(column)) for event in events]
        if any(math.isfinite(value) for value in values):
            result.append(column)
    return result


def aggregate_event_features(events: list[dict[str, Any]]) -> dict[str, Any]:
    if not events:
        return {
            "event_count": 0,
            "event_unique_agents": 0,
            "event_time_first": "",
            "event_time_last": "",
            "event_time_span": "",
        }

    aggregate: dict[str, Any] = {
        "event_count": len(events),
        "event_unique_agents": len({int(event["agent_id"]) for event in events}),
        "event_time_first": int(events[0]["env_time"]),
        "event_time_last": int(events[-1]["env_time"]),
        "event_time_span": int(events[-1]["env_time"]) - int(events[0]["env_time"]),
        "event_first_agent_id": int(events[0]["agent_id"]),
        "event_last_agent_id": int(events[-1]["agent_id"]),
    }

    transitions = Counter(
        f"{event['baseline_action_name']}->{event['candidate_action_name']}"
        for event in events
    )
    baseline_actions = Counter(event["baseline_action_name"] for event in events)
    candidate_actions = Counter(event["candidate_action_name"] for event in events)
    for transition, count in transitions.items():
        aggregate[f"event_transition_{column_slug(transition)}"] = count
    for action, count in baseline_actions.items():
        aggregate[f"event_baseline_action_{column_slug(action)}"] = count
    for action, count in candidate_actions.items():
        aggregate[f"event_candidate_action_{column_slug(action)}"] = count

    for column in numeric_event_columns(events):
        values = [safe_float(event.get(column)) for event in events]
        finite_values = [value for value in values if math.isfinite(value)]
        if not finite_values:
            continue
        aggregate[f"event_{column}_first"] = finite_values[0]
        aggregate[f"event_{column}_last"] = finite_values[-1]
        aggregate[f"event_{column}_mean"] = float(np.mean(finite_values))
        aggregate[f"event_{column}_min"] = float(np.min(finite_values))
        aggregate[f"event_{column}_max"] = float(np.max(finite_values))
    return aggregate


def run_diff_prefix_episode(
    args: argparse.Namespace,
    seed: int,
    prefix_len: int,
) -> dict[str, Any]:
    env, obs_builder = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    baseline_policy = instantiate_policy(args.baseline_policy, args.baseline_checkpoint)
    candidate_policy = instantiate_policy(args.candidate_policy, args.candidate_checkpoint)
    reward_values: list[float] = []
    positions: dict[int, list[Any]] = defaultdict(list)
    actions_by_agent: dict[int, list[int]] = defaultdict(list)
    events: list[dict[str, Any]] = []
    event_details: list[dict[str, Any]] = []

    while int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
        observations_by_handle = dict(zip(handles, obs_list))
        baseline_raw_actions = raw_policy_actions(baseline_policy, handles, obs_list)
        candidate_raw_actions = raw_policy_actions(candidate_policy, handles, obs_list)
        baseline_actions = policy_actions(baseline_policy, handles, obs_list)
        candidate_actions = policy_actions(candidate_policy, handles, obs_list)
        actions = dict(baseline_actions)

        for handle in handles:
            if len(events) >= prefix_len:
                break
            baseline_action = baseline_actions.get(handle)
            candidate_action = candidate_actions.get(handle)
            if baseline_action is None or candidate_action is None:
                continue
            if baseline_action == candidate_action:
                continue

            detail = diff_row(
                args,
                seed,
                env,
                obs_builder,
                handle,
                observations_by_handle[handle],
                baseline_policy,
                candidate_policy,
                baseline_raw_actions,
                candidate_raw_actions,
                baseline_actions,
                candidate_actions,
            )
            detail.update(
                candidate_source_features(
                    args.candidate_policy,
                    args.candidate_checkpoint,
                )
            )
            detail = add_delta_features(detail)
            detail["prefix_index"] = len(events) + 1
            event_details.append(detail)

            actions[handle] = candidate_action
            events.append(
                {
                    "prefix_index": len(events) + 1,
                    "env_time": int(env._elapsed_steps),
                    "agent_id": int(handle),
                    "baseline_action": int(baseline_action),
                    "baseline_action_name": action_name(baseline_action),
                    "candidate_action": int(candidate_action),
                    "candidate_action_name": action_name(candidate_action),
                }
            )

        observations, rewards_by_agent, dones, _ = env.step(actions)
        for handle, action in actions.items():
            actions_by_agent[handle].append(action)
        for handle in handles:
            positions[handle].append(env.agents[handle].position)
        reward_values.extend(float(rewards_by_agent.get(handle, 0.0)) for handle in handles)
        if done_all(dones):
            break

    result = final_result(args, seed, env, reward_values, positions, actions_by_agent)
    result["events"] = events
    result["event_details"] = event_details
    result["forced_applied"] = len(events)
    return result


def compare_row(
    args: argparse.Namespace,
    seed: int,
    prefix_len: int,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    forced: dict[str, Any],
) -> dict[str, Any]:
    baseline_failed = len(baseline["failed_agents"])
    forced_failed = len(forced["failed_agents"])
    source_features = candidate_source_features(
        args.candidate_policy,
        args.candidate_checkpoint,
    )
    row = {
        "seed": seed,
        "scene": args.scene or "scene_5",
        "num_agents": args.num_agents,
        "line_length": args.line_length,
        "prefix_len": prefix_len,
        "forced_applied": forced["forced_applied"],
        "baseline_reward": baseline["normalized_reward"],
        "forced_reward": forced["normalized_reward"],
        "reward_delta": forced["normalized_reward"] - baseline["normalized_reward"],
        "baseline_success": baseline["success_rate"],
        "forced_success": forced["success_rate"],
        "success_delta": forced["success_rate"] - baseline["success_rate"],
        "baseline_env_time": baseline["env_time"],
        "forced_env_time": forced["env_time"],
        "env_time_delta": forced["env_time"] - baseline["env_time"],
        "baseline_failed_agents": baseline_failed,
        "forced_failed_agents": forced_failed,
        "failed_agents_delta": forced_failed - baseline_failed,
        "baseline_failed_agent_ids": baseline["failed_agent_ids"],
        "forced_failed_agent_ids": forced["failed_agent_ids"],
        "candidate_reward": candidate["normalized_reward"],
        "candidate_success": candidate["success_rate"],
        "candidate_reward_delta": candidate["normalized_reward"]
        - baseline["normalized_reward"],
        "candidate_success_delta": candidate["success_rate"] - baseline["success_rate"],
        "candidate_failed_agent_ids": candidate["failed_agent_ids"],
        "events": compact_events(forced["events"]),
        **source_features,
        **aggregate_event_features(forced["event_details"]),
    }
    row["outcome_category"] = outcome_category(row, args.reward_epsilon)
    row["event_details"] = forced["event_details"]
    return row


def csv_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if key != "event_details"}
        for row in rows
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = csv_rows(rows)
    fieldnames = sorted({key for row in flat_rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_rows)


def write_json(
    path: Path,
    rows: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump({"summary": summaries, "rows": rows}, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Mine adaptive candidate-vs-baseline diff prefixes as sequence-level "
            "training rows with aggregated conflict/deadline features."
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
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--seeds",
        help="Comma-separated exact seed list. Overrides --seed/--episodes when set.",
    )
    parser.add_argument(
        "--prefix-lengths",
        nargs="+",
        type=int,
        help="Diff prefix lengths to evaluate. Defaults to 1..--max-prefix-len.",
    )
    parser.add_argument("--max-prefix-len", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help="Skip rows where no candidate diff was applied.",
    )
    parser.add_argument(
        "--require-full-prefix",
        action="store_true",
        help="Skip rows where fewer than prefix_len candidate diffs were applied.",
    )
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = selected_seeds(args)
    prefix_lengths = selected_prefix_lengths(args)
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        baseline = run_policy_episode(
            args,
            seed,
            args.baseline_policy,
            args.baseline_checkpoint,
        )
        candidate = run_policy_episode(
            args,
            seed,
            args.candidate_policy,
            args.candidate_checkpoint,
        )
        if not args.quiet:
            print(
                f"seed={seed} baseline={baseline['normalized_reward']:.6g}/"
                f"{baseline['success_rate']:.6g} candidate="
                f"{candidate['normalized_reward']:.6g}/{candidate['success_rate']:.6g}",
                flush=True,
            )

        for prefix_len in prefix_lengths:
            forced = run_diff_prefix_episode(args, seed, prefix_len)
            if args.only_changed and forced["forced_applied"] <= 0:
                continue
            if args.require_full_prefix and forced["forced_applied"] < prefix_len:
                continue
            row = compare_row(args, seed, prefix_len, baseline, candidate, forced)
            rows.append(row)
            if not args.quiet:
                print(
                    f"  prefix={prefix_len} applied={row['forced_applied']} "
                    f"forced={row['forced_reward']:.6g}/{row['forced_success']:.6g} "
                    f"delta={row['reward_delta']:.6g}/{row['success_delta']:.6g} "
                    f"label={row['outcome_category']}",
                    flush=True,
                )

    summaries = summarize_rows(rows)
    print("\nSummary:")
    for summary in summaries:
        print(
            f"prefix={summary['prefix_len']} episodes={summary['episodes']} "
            f"applied_mean={summary['forced_applied_mean']:.3g} "
            f"delta={summary['reward_delta_mean']:.6g}/"
            f"{summary['success_delta_mean']:.6g} "
            f"reward={summary['reward_wins']}/{summary['reward_losses']}/"
            f"{summary['reward_ties']} "
            f"success={summary['success_wins']}/{summary['success_losses']}/"
            f"{summary['success_ties']}"
        )

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    if args.output_json is not None:
        write_json(args.output_json, rows, summaries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
