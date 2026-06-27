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
    policy_actions,
)
from tools.counterfactual_decision_eval import final_result, make_env, set_runtime_context
from tools.evaluate_sampled import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_REWARDS,
    observation_list,
    repo_root,
)


def read_events(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows.append(
                    {
                        "seed": int(float(row["seed"])),
                        "env_time": int(float(row["env_time"])),
                        "agent_id": int(float(row["agent_id"])),
                        "forced_action": int(float(row["forced_action"])),
                        "source": row.get("source", path.name),
                        "schedule_id": row.get("schedule_id", row.get("source", path.name)),
                    }
                )
    return rows


def expand_holds(seed: int, hold_specs: list[str]) -> list[dict[str, Any]]:
    events = []
    for spec in hold_specs:
        parts = [item.strip() for item in spec.split(":")]
        if len(parts) != 4:
            raise ValueError(
                "--hold must be formatted as agent:start:end:action, "
                f"got {spec!r}"
            )
        agent_id, start, end, action = (int(float(part)) for part in parts)
        if end < start:
            raise ValueError(f"--hold end before start in {spec!r}")
        for step in range(start, end + 1):
            events.append(
                {
                    "seed": int(seed),
                    "env_time": step,
                    "agent_id": agent_id,
                    "forced_action": action,
                    "source": f"hold:{spec}",
                    "schedule_id": f"hold:{spec}",
                }
            )
    return events


def events_by_schedule_seed(
    events: list[dict[str, Any]],
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        grouped[(str(event.get("schedule_id", "default")), int(event["seed"]))].append(event)
    return {
        key: sorted(seed_events, key=lambda row: (row["env_time"], row["agent_id"]))
        for key, seed_events in grouped.items()
    }


def run_episode(
    args: argparse.Namespace,
    seed: int,
    events: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    set_runtime_context(args, seed)
    env, _obs_builder = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    policy = instantiate_policy(args.policy, args.policy_checkpoint)
    reward_values: list[float] = []
    positions: dict[int, list[Any]] = defaultdict(list)
    actions_by_agent: dict[int, list[int]] = defaultdict(list)
    event_index = 0
    sorted_events = sorted(events or [], key=lambda row: (row["env_time"], row["agent_id"]))
    applied: list[dict[str, Any]] = []

    while int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
        actions = policy_actions(policy, handles, obs_list)
        current_step = int(env._elapsed_steps)

        while event_index < len(sorted_events):
            event = sorted_events[event_index]
            if int(event["env_time"]) != current_step:
                break
            event_index += 1
            handle = int(event["agent_id"])
            if handle not in actions:
                continue
            baseline_action = int(actions[handle])
            forced_action = int(event["forced_action"])
            actions[handle] = forced_action
            applied.append(
                {
                    **event,
                    "baseline_action": baseline_action,
                    "baseline_action_name": action_name(baseline_action),
                    "forced_action_name": action_name(forced_action),
                }
            )

        observations, rewards_by_agent, dones, _ = env.step(actions)
        for handle, action in actions.items():
            actions_by_agent[handle].append(int(action))
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
    schedule_id: str,
    seed: int,
    baseline: dict[str, Any],
    forced: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schedule_id": schedule_id,
        "seed": seed,
        "baseline_reward": baseline["normalized_reward"],
        "forced_reward": forced["normalized_reward"],
        "reward_delta": forced["normalized_reward"] - baseline["normalized_reward"],
        "baseline_success": baseline["success_rate"],
        "forced_success": forced["success_rate"],
        "success_delta": forced["success_rate"] - baseline["success_rate"],
        "baseline_env_time": baseline["env_time"],
        "forced_env_time": forced["env_time"],
        "baseline_failed_agents": len(baseline["failed_agents"]),
        "forced_failed_agents": len(forced["failed_agents"]),
        "failed_agents_delta": len(forced["failed_agents"]) - len(baseline["failed_agents"]),
        "baseline_failed_agent_ids": baseline["failed_agent_ids"],
        "forced_failed_agent_ids": forced["failed_agent_ids"],
        "forced_applied": forced["forced_applied"],
        "events": ";".join(
            "{env_time}:a{agent_id}:{baseline_action_name}->{forced_action_name}".format(
                **event
            )
            for event in forced["events"]
        ),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "episodes": 0,
            "reward_delta_mean": 0.0,
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
        "reward_delta_mean": mean("reward_delta"),
        "success_delta_mean": mean("success_delta"),
        "reward_wins": sum(float(row["reward_delta"]) > 1e-9 for row in rows),
        "reward_losses": sum(float(row["reward_delta"]) < -1e-9 for row in rows),
        "reward_ties": sum(abs(float(row["reward_delta"])) <= 1e-9 for row in rows),
        "success_wins": sum(float(row["success_delta"]) > 1e-9 for row in rows),
        "success_losses": sum(float(row["success_delta"]) < -1e-9 for row in rows),
        "success_ties": sum(abs(float(row["success_delta"])) <= 1e-9 for row in rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Force exact action schedules and compare against a baseline policy."
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--policy", default="submission.risk_veto_policy.MyPolicy")
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--events-csv", nargs="*", type=Path, default=[])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--hold",
        nargs="*",
        default=[],
        help="Repeat forced actions as agent:start:end:action.",
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events = read_events(args.events_csv)
    events.extend(expand_holds(args.seed, args.hold))
    grouped_events = events_by_schedule_seed(events)
    if not grouped_events:
        raise ValueError("No forced events configured. Use --events-csv or --hold.")

    rows = []
    baseline_by_seed = {}
    for (schedule_id, seed), seed_events in grouped_events.items():
        if seed not in baseline_by_seed:
            baseline_by_seed[seed] = run_episode(args, seed)
        baseline = baseline_by_seed[seed]
        forced = run_episode(args, seed, seed_events)
        row = compare_row(schedule_id, seed, baseline, forced)
        rows.append(row)
        print(
            "schedule={schedule} seed={seed} applied={applied} reward_delta={reward:.6g} "
            "success_delta={success:.6g} failed={base_failed}->{forced_failed}".format(
                schedule=schedule_id,
                seed=seed,
                applied=row["forced_applied"],
                reward=row["reward_delta"],
                success=row["success_delta"],
                base_failed=row["baseline_failed_agents"],
                forced_failed=row["forced_failed_agents"],
            ),
            flush=True,
        )

    summary = summarize(rows)
    print(
        "summary episodes={episodes} delta={reward:.6g}/{success:.6g} "
        "reward={rw}/{rl}/{rt} success={sw}/{sl}/{st}".format(
            episodes=summary["episodes"],
            reward=summary["reward_delta_mean"],
            success=summary["success_delta_mean"],
            rw=summary["reward_wins"],
            rl=summary["reward_losses"],
            rt=summary["reward_ties"],
            sw=summary["success_wins"],
            sl=summary["success_losses"],
            st=summary["success_ties"],
        )
    )
    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump({"summary": summary, "rows": rows}, handle, indent=2)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
