#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from tools.analyze_policy_action_diffs import (
    action_name,
    done_all,
    instantiate_policy,
    make_env,
    policy_actions,
)
from tools.evaluate_sampled import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_REWARDS,
    failed_agent_details,
    normalized_reward,
    observation_list,
    repo_root,
)


def final_result(
    args: argparse.Namespace,
    seed: int,
    env: Any,
    reward_values: list[float],
    positions: dict[int, list[Any]],
    actions_by_agent: dict[int, list[int]],
) -> dict[str, Any]:
    failed_agents = failed_agent_details(env, positions, actions_by_agent)
    success_rate = sum(int(agent.state == 6) for agent in env.agents) / env.get_num_agents()
    return {
        "seed": seed,
        "env_time": int(env._elapsed_steps),
        "success_rate": success_rate,
        "normalized_reward": normalized_reward(env, reward_values),
        "failed_agents": failed_agents,
        "failed_agent_ids": ",".join(str(agent["agent_id"]) for agent in failed_agents),
        "num_agents": env.get_num_agents(),
        "max_episode_steps": env._max_episode_steps,
        "line_length": args.line_length,
        "scene": args.scene or "scene_5",
    }


def run_policy_episode(
    args: argparse.Namespace,
    seed: int,
    policy_path: str,
    checkpoint: Path | None,
) -> dict[str, Any]:
    env, _ = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    policy = instantiate_policy(policy_path, checkpoint)
    reward_values: list[float] = []
    positions: dict[int, list[Any]] = defaultdict(list)
    actions_by_agent: dict[int, list[int]] = defaultdict(list)

    while int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        actions = policy_actions(policy, handles, observation_list(observations, handles))
        observations, rewards_by_agent, dones, _ = env.step(actions)
        for handle, action in actions.items():
            actions_by_agent[handle].append(action)
        for handle in handles:
            positions[handle].append(env.agents[handle].position)
        reward_values.extend(float(rewards_by_agent.get(handle, 0.0)) for handle in handles)
        if done_all(dones):
            break

    return final_result(args, seed, env, reward_values, positions, actions_by_agent)


def run_diff_prefix_episode(
    args: argparse.Namespace,
    seed: int,
    prefix_len: int,
) -> dict[str, Any]:
    env, _ = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    baseline_policy = instantiate_policy(args.baseline_policy, args.baseline_checkpoint)
    candidate_policy = instantiate_policy(args.candidate_policy, args.candidate_checkpoint)
    reward_values: list[float] = []
    positions: dict[int, list[Any]] = defaultdict(list)
    actions_by_agent: dict[int, list[int]] = defaultdict(list)
    events: list[dict[str, Any]] = []

    while int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
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
    result["forced_applied"] = len(events)
    return result


def compact_events(events: list[dict[str, Any]]) -> str:
    return ";".join(
        "{prefix_index}@{env_time}:a{agent_id}:{baseline_action_name}->{candidate_action_name}".format(
            **event
        )
        for event in events
    )


def compare_row(
    seed: int,
    prefix_len: int,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    forced: dict[str, Any],
) -> dict[str, Any]:
    return {
        "seed": seed,
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
        "baseline_failed_agents": len(baseline["failed_agents"]),
        "forced_failed_agents": len(forced["failed_agents"]),
        "failed_agents_delta": len(forced["failed_agents"]) - len(baseline["failed_agents"]),
        "baseline_failed_agent_ids": baseline["failed_agent_ids"],
        "forced_failed_agent_ids": forced["failed_agent_ids"],
        "candidate_reward": candidate["normalized_reward"],
        "candidate_success": candidate["success_rate"],
        "candidate_reward_delta": candidate["normalized_reward"] - baseline["normalized_reward"],
        "candidate_success_delta": candidate["success_rate"] - baseline["success_rate"],
        "candidate_failed_agent_ids": candidate["failed_agent_ids"],
        "events": compact_events(forced["events"]),
        "event_details": forced["events"],
    }


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["prefix_len"])].append(row)

    summaries = []
    for prefix_len in sorted(grouped):
        prefix_rows = grouped[prefix_len]

        def mean(key: str) -> float:
            return sum(float(row[key]) for row in prefix_rows) / len(prefix_rows)

        summaries.append(
            {
                "prefix_len": prefix_len,
                "episodes": len(prefix_rows),
                "forced_applied_mean": mean("forced_applied"),
                "reward_delta_mean": mean("reward_delta"),
                "success_delta_mean": mean("success_delta"),
                "reward_wins": sum(int(row["reward_delta"] > 1e-9) for row in prefix_rows),
                "reward_losses": sum(int(row["reward_delta"] < -1e-9) for row in prefix_rows),
                "reward_ties": sum(int(abs(row["reward_delta"]) <= 1e-9) for row in prefix_rows),
                "success_wins": sum(int(row["success_delta"] > 1e-9) for row in prefix_rows),
                "success_losses": sum(int(row["success_delta"] < -1e-9) for row in prefix_rows),
                "success_ties": sum(int(abs(row["success_delta"]) <= 1e-9) for row in prefix_rows),
            }
        )
    return summaries


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
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0].keys()))
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
            "Replay adaptive prefixes of candidate-vs-baseline policy action "
            "differences and measure final episode outcomes."
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
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def selected_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(item) for item in args.seeds.split(",") if item.strip()]
    return [args.seed + index for index in range(args.episodes)]


def selected_prefix_lengths(args: argparse.Namespace) -> list[int]:
    if args.prefix_lengths is not None:
        lengths = args.prefix_lengths
    else:
        lengths = list(range(1, args.max_prefix_len + 1))
    unique_lengths = sorted({int(length) for length in lengths})
    if not unique_lengths or unique_lengths[0] <= 0:
        raise ValueError("Prefix lengths must be positive")
    return unique_lengths


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
            row = compare_row(seed, prefix_len, baseline, candidate, forced)
            rows.append(row)
            if not args.quiet:
                print(
                    f"  prefix={prefix_len} applied={row['forced_applied']} "
                    f"forced={row['forced_reward']:.6g}/{row['forced_success']:.6g} "
                    f"delta={row['reward_delta']:.6g}/{row['success_delta']:.6g}",
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
