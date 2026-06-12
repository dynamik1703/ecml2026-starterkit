#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.analyze_policy_action_diffs import (
    action_name,
    corridor_len,
    done_all,
    future_risk,
    instantiate_policy,
    planned_prefixes,
    policy_actions,
)
from tools.counterfactual_decision_eval import (
    decision_row,
    final_result,
    is_critical_decision,
    make_env,
    valid_actions_from_mask,
)
from tools.evaluate_policy_diff_prefixes import (
    compact_events,
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
from tools.mine_diff_prefix_dataset import aggregate_event_features
from tools.train_counterfactual_gate import outcome_category


DEFAULT_FORCED_ACTIONS = "LEFT,FORWARD,RIGHT"
DEFAULT_REJECT_TRANSITIONS = "MOVE_RIGHT->MOVE_FORWARD"


def parse_action_token(token: str) -> int:
    normalized = token.strip().upper()
    if not normalized:
        raise argparse.ArgumentTypeError("Empty action token")
    if normalized.isdigit():
        return int(normalized)
    names = {
        "DO_NOTHING": 0,
        "NOTHING": 0,
        "LEFT": 1,
        "MOVE_LEFT": 1,
        "FORWARD": 2,
        "MOVE_FORWARD": 2,
        "RIGHT": 3,
        "MOVE_RIGHT": 3,
        "STOP": 4,
        "STOP_MOVING": 4,
    }
    try:
        return names[normalized]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(f"Unknown action token: {token}") from exc


def parse_actions(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    actions = []
    for part in value.split(","):
        token = part.strip()
        if token:
            actions.append(parse_action_token(token))
    return tuple(dict.fromkeys(actions))


def parse_transition_set(value: str) -> set[tuple[int, int]]:
    transitions: set[tuple[int, int]] = set()
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if "->" not in token:
            raise argparse.ArgumentTypeError(f"Transition must contain ->: {token}")
        left, right = (part.strip() for part in token.split("->", maxsplit=1))
        transitions.add((parse_action_token(left), parse_action_token(right)))
    return transitions


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if np.isfinite(result) else default


def positive_delta(row: dict[str, Any], key: str) -> float:
    return max(0.0, safe_float(row.get(f"baseline_{key}")) - safe_float(row.get(f"forced_{key}")))


def planner_score(row: dict[str, Any], args: argparse.Namespace) -> float:
    distance_penalty = max(0.0, safe_float(row.get("forced_distance_delta")))
    occupied_penalty = float(bool(row.get("forced_target_occupied_by_other")))
    unreachable_penalty = float(not np.isfinite(safe_float(row.get("forced_target_distance"), float("nan"))))
    score = 0.0
    score += args.head_on_weight * positive_delta(row, "prefix_head_on_edge_conflicts")
    score += args.same_edge_weight * positive_delta(row, "prefix_same_edge_conflicts")
    score += args.cell_conflict_weight * positive_delta(row, "prefix_cell_intersections")
    score += args.future_risk_weight * max(0.0, -safe_float(row.get("future_head_on_risk_delta")))
    score += args.deadline_slack_weight * max(
        0.0,
        safe_float(row.get("forced_prefix_min_pair_deadline_slack_delta")),
    )
    score += args.rejoin_weight * safe_float(row.get("prefix_rejoin_found"))
    score -= args.distance_weight * distance_penalty
    score -= args.occupied_weight * occupied_penalty
    score -= args.unreachable_weight * unreachable_penalty
    return float(score)


def normalize_event_detail(row: dict[str, Any], source: str) -> dict[str, Any]:
    detail = dict(row)
    detail["candidate_action"] = detail.get("forced_action")
    detail["candidate_action_name"] = detail.get("forced_action_name")
    detail["candidate_target"] = detail.get("forced_target")
    detail["candidate_target_distance"] = detail.get("forced_target_distance")
    detail["candidate_source_known"] = 1.0
    detail[f"candidate_source_{source}"] = 1.0
    return detail


def compact_planner_events(events: list[dict[str, Any]]) -> str:
    return compact_events(
        [
            {
                "prefix_index": index + 1,
                "env_time": int(event["env_time"]),
                "agent_id": int(event["agent_id"]),
                "baseline_action_name": str(event["baseline_action_name"]),
                "candidate_action_name": str(event["forced_action_name"]),
            }
            for index, event in enumerate(events)
        ]
    )


def collect_planner_events(
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env, obs_builder = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    policy = instantiate_policy(args.policy, args.policy_checkpoint)
    reward_values: list[float] = []
    positions: dict[int, list[Any]] = defaultdict(list)
    actions_by_agent: dict[int, list[int]] = defaultdict(list)
    events: list[dict[str, Any]] = []
    decision_index = 0
    collecting = True

    while int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
        observations_by_handle = dict(zip(handles, obs_list))
        actions = policy_actions(policy, handles, obs_list)
        prefixes = planned_prefixes(policy, obs_builder, actions)
        step_candidates: list[dict[str, Any]] = []

        if collecting:
            for handle in handles:
                if handle not in actions:
                    continue
                observation = observations_by_handle[handle]
                baseline_action = int(actions[handle])
                alternatives = valid_actions_from_mask(
                    observation,
                    baseline_action,
                    args.include_do_nothing,
                    args.forced_action_ids,
                )
                if not alternatives:
                    continue
                baseline_risk = future_risk(
                    policy,
                    obs_builder,
                    handle,
                    baseline_action,
                    prefixes,
                )
                baseline_corridor_len = corridor_len(obs_builder, handle, baseline_action)
                if not is_critical_decision(
                    args,
                    observation,
                    baseline_action,
                    baseline_risk,
                    baseline_corridor_len,
                    alternatives,
                ):
                    continue
                decision_index += 1
                for forced_action in alternatives[: args.max_alternatives_per_decision]:
                    row = decision_row(
                        args,
                        seed,
                        decision_index,
                        env,
                        obs_builder,
                        policy,
                        handle,
                        observation,
                        actions,
                        int(forced_action),
                        prefixes,
                    )
                    row["planner_score"] = planner_score(row, args)
                    if row["planner_score"] >= args.min_planner_score:
                        step_candidates.append(row)

            step_candidates = [
                row
                for row in step_candidates
                if (
                    int(row["baseline_action"]),
                    int(row["forced_action"]),
                )
                not in args.rejected_transition_ids
            ]
            step_candidates.sort(
                key=lambda row: (
                    -safe_float(row.get("planner_score")),
                    safe_float(row.get("forced_distance_delta")),
                    int(row.get("agent_id", 0)),
                    int(row.get("forced_action", 0)),
                )
            )
            for row in step_candidates[: args.max_events_per_step]:
                events.append(row)
                if len(events) >= args.max_candidate_events_per_seed:
                    collecting = False
                    break

        observations, rewards_by_agent, dones, _ = env.step(actions)
        for handle, action in actions.items():
            actions_by_agent[handle].append(action)
        for handle in handles:
            positions[handle].append(env.agents[handle].position)
        reward_values.extend(float(rewards_by_agent.get(handle, 0.0)) for handle in handles)
        if done_all(dones):
            break

    result = final_result(args, seed, env, reward_values, positions, actions_by_agent)
    return result, events


def top_planned_events(
    events: list[dict[str, Any]],
    prefix_len: int,
) -> list[dict[str, Any]]:
    top_events = sorted(
        events,
        key=lambda row: (
            -safe_float(row.get("planner_score")),
            safe_float(row.get("forced_distance_delta")),
            int(row.get("env_time", 0)),
            int(row.get("agent_id", 0)),
        ),
    )[:prefix_len]
    return sorted(
        top_events,
        key=lambda row: (int(row["env_time"]), int(row["agent_id"])),
    )


def mask_allows(observation: Any, action: int) -> bool:
    values = np.asarray(observation, dtype=np.float32)
    if values.shape[0] < 5:
        return True
    mask = values[-5:]
    return 0 <= action < len(mask) and mask[action] >= 0.5


def run_forced_sequence_episode(
    args: argparse.Namespace,
    seed: int,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    env, _obs_builder = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    policy = instantiate_policy(args.policy, args.policy_checkpoint)
    reward_values: list[float] = []
    positions: dict[int, list[Any]] = defaultdict(list)
    actions_by_agent: dict[int, list[int]] = defaultdict(list)
    sorted_events = sorted(
        events,
        key=lambda event: (int(event["env_time"]), int(event["agent_id"])),
    )
    next_event = 0
    applied: list[dict[str, Any]] = []

    while int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
        observations_by_handle = dict(zip(handles, obs_list))
        actions = policy_actions(policy, handles, obs_list)
        current_step = int(env._elapsed_steps)

        while (
            next_event < len(sorted_events)
            and int(sorted_events[next_event]["env_time"]) == current_step
        ):
            event = sorted_events[next_event]
            next_event += 1
            handle = int(event["agent_id"])
            forced_action = int(event["forced_action"])
            if (
                handle in actions
                and handle in observations_by_handle
                and mask_allows(observations_by_handle[handle], forced_action)
            ):
                actions[handle] = forced_action
                applied.append(event)

        observations, rewards_by_agent, dones, _ = env.step(actions)
        for handle, action in actions.items():
            actions_by_agent[handle].append(action)
        for handle in handles:
            positions[handle].append(env.agents[handle].position)
        reward_values.extend(float(rewards_by_agent.get(handle, 0.0)) for handle in handles)
        if done_all(dones):
            break

    result = final_result(args, seed, env, reward_values, positions, actions_by_agent)
    result["forced_applied"] = len(applied)
    result["events"] = applied
    return result


def compare_row(
    args: argparse.Namespace,
    seed: int,
    prefix_len: int,
    baseline: dict[str, Any],
    planned_events: list[dict[str, Any]],
    forced: dict[str, Any],
) -> dict[str, Any]:
    applied_details = [
        normalize_event_detail(event, args.candidate_source)
        for event in forced["events"]
    ]
    baseline_failed = len(baseline["failed_agents"])
    forced_failed = len(forced["failed_agents"])
    row = {
        "seed": seed,
        "scene": args.scene or "scene_5",
        "num_agents": args.num_agents,
        "line_length": args.line_length,
        "prefix_len": prefix_len,
        "planned_events": len(planned_events),
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
        "events": compact_planner_events(forced["events"]),
        "candidate_source_known": 1.0,
        f"candidate_source_{args.candidate_source}": 1.0,
        **aggregate_event_features(applied_details),
    }
    row["outcome_category"] = outcome_category(row, args.reward_epsilon)
    row["event_details"] = applied_details
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
            "Mine short rolling-horizon rescue sequences from conflict-reducing "
            "counterfactual actions along a baseline rollout."
        )
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--policy", default="submission.sequence_success_policy.MyPolicy")
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--seeds",
        help="Comma-separated exact seed list. Overrides --seed/--episodes when set.",
    )
    parser.add_argument("--prefix-lengths", nargs="+", type=int)
    parser.add_argument("--max-prefix-len", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--forced-actions", default=DEFAULT_FORCED_ACTIONS)
    parser.add_argument("--include-do-nothing", action="store_true")
    parser.add_argument("--critical-only", action="store_true", default=True)
    parser.add_argument("--all-decisions", dest="critical_only", action="store_false")
    parser.add_argument("--min-corridor-len", type=int, default=8)
    parser.add_argument("--max-events-per-seed", type=int, default=16)
    parser.add_argument("--max-candidate-events-per-seed", type=int, default=256)
    parser.add_argument("--max-events-per-step", type=int, default=1)
    parser.add_argument("--max-alternatives-per-decision", type=int, default=3)
    parser.add_argument("--min-planner-score", type=float, default=0.01)
    parser.add_argument("--reject-transitions", default=DEFAULT_REJECT_TRANSITIONS)
    parser.add_argument("--head-on-weight", type=float, default=2.0)
    parser.add_argument("--same-edge-weight", type=float, default=0.6)
    parser.add_argument("--cell-conflict-weight", type=float, default=0.25)
    parser.add_argument("--future-risk-weight", type=float, default=2.0)
    parser.add_argument("--deadline-slack-weight", type=float, default=0.002)
    parser.add_argument("--rejoin-weight", type=float, default=0.05)
    parser.add_argument("--distance-weight", type=float, default=0.03)
    parser.add_argument("--occupied-weight", type=float, default=2.0)
    parser.add_argument("--unreachable-weight", type=float, default=5.0)
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument("--candidate-source", default="rolling_sequence_planner")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.forced_action_ids = parse_actions(args.forced_actions)
    args.rejected_transition_ids = parse_transition_set(args.reject_transitions)
    seeds = selected_seeds(args)
    prefix_lengths = selected_prefix_lengths(args)
    rows: list[dict[str, Any]] = []

    for seed in seeds:
        baseline, candidate_events = collect_planner_events(args, seed)
        if not args.quiet:
            print(
                f"seed={seed} baseline={baseline['normalized_reward']:.6g}/"
                f"{baseline['success_rate']:.6g} candidates={len(candidate_events)}",
                flush=True,
            )

        for prefix_len in prefix_lengths:
            if prefix_len > args.max_events_per_seed:
                continue
            prefix_events = top_planned_events(candidate_events, prefix_len)
            if not prefix_events:
                continue
            forced = run_forced_sequence_episode(args, seed, prefix_events)
            if forced["forced_applied"] <= 0:
                continue
            row = compare_row(
                args,
                seed,
                prefix_len,
                baseline,
                candidate_events,
                forced,
            )
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
