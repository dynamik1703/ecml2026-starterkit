#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.analyze_policy_action_diffs import (
    action_name,
    corridor_len,
    direction_features,
    distance_and_slack,
    done_all,
    future_risk,
    instantiate_policy,
    logit_features,
    mask_values,
    observation_list,
    observation_scalar_features,
    planned_prefixes,
    policy_actions,
    prefix_conflict_features,
    raw_logits,
    route_prefix_for_action,
    state_name,
    target_metrics,
)
from tools.evaluate_sampled import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_REWARDS,
    failed_agent_details,
    load_sampling_env_generator,
    load_symbol,
    normalized_reward,
    repo_root,
)
from flatland.envs.persistence import RailEnvPersister


MOVE_ACTIONS = (1, 2, 3)
DEFAULT_ACTIONS = (1, 2, 3, 4)


def make_env(args: argparse.Namespace) -> tuple[Any, Any]:
    obs_builder = load_symbol(args.obs_builder)()
    rewards = load_symbol(args.rewards)()
    env, _ = RailEnvPersister.load_new(
        args.base_state_pkl,
        obs_builder=obs_builder,
        rewards=rewards,
    )
    if args.num_agents is not None:
        env.number_of_agents = args.num_agents
    env = load_sampling_env_generator()(env, line_length=args.line_length, scene=args.scene)
    return env, obs_builder


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


def valid_actions_from_mask(
    observation: Any,
    baseline_action: int,
    include_do_nothing: bool,
    forced_actions: tuple[int, ...] | None,
) -> list[int]:
    values = np.asarray(observation, dtype=np.float32)
    mask = values[-5:] if values.shape[0] >= 5 else np.ones(5, dtype=np.float32)
    if forced_actions is not None:
        action_pool = forced_actions
    else:
        action_pool = (0, *DEFAULT_ACTIONS) if include_do_nothing else DEFAULT_ACTIONS
    return [
        action
        for action in action_pool
        if action != baseline_action and action < len(mask) and mask[action] >= 0.5
    ]


def has_opposing_ahead(observation: Any) -> bool:
    values = np.asarray(observation, dtype=np.float32)
    return bool(values.shape[0] >= 18 and np.max(values[14:18]) >= 0.5)


def has_same_direction_ahead(observation: Any) -> bool:
    values = np.asarray(observation, dtype=np.float32)
    return bool(values.shape[0] >= 22 and np.max(values[18:22]) >= 0.5)


def is_critical_decision(
    args: argparse.Namespace,
    observation: Any,
    baseline_action: int,
    baseline_risk: float,
    baseline_corridor_len: int,
    alternatives: list[int],
) -> bool:
    if not args.critical_only:
        return True
    if not alternatives:
        return False
    values = np.asarray(observation, dtype=np.float32)
    on_switch = values.shape[0] >= 10 and bool(np.max(values[7:10]) >= 0.5)
    if on_switch:
        return True
    if baseline_risk > 0.0:
        return True
    if baseline_corridor_len >= args.min_corridor_len:
        return True
    if baseline_action == 4 and any(action in MOVE_ACTIONS for action in alternatives):
        return True
    if has_opposing_ahead(observation) or has_same_direction_ahead(observation):
        return True
    return False


def forced_action_score(
    obs_builder: Any,
    handle: int,
    baseline_target: dict[str, Any],
    forced_action: int,
    baseline_risk: float,
    forced_risk: float,
) -> tuple[float, float, float, int]:
    forced_target = target_metrics(obs_builder, handle, forced_action)
    forced_distance = float(forced_target["target_distance"])
    baseline_distance = float(baseline_target["target_distance"])
    distance_delta = forced_distance - baseline_distance
    occupied_penalty = 1 if forced_target["target_occupied_by_other"] else 0
    unreachable_penalty = 1 if not np.isfinite(forced_distance) else 0
    risk_delta = forced_risk - baseline_risk
    action_order = {2: 0, 1: 1, 3: 2, 4: 3, 0: 4}.get(forced_action, 9)
    return (
        float(unreachable_penalty),
        float(occupied_penalty),
        float(risk_delta),
        float(distance_delta),
        action_order,
    )


def decision_row(
    args: argparse.Namespace,
    seed: int,
    decision_index: int,
    env: Any,
    obs_builder: Any,
    policy: Any,
    handle: int,
    observation: Any,
    baseline_actions: dict[int, int],
    forced_action: int,
    baseline_prefixes: dict[int, Any],
) -> dict[str, Any]:
    agent = env.agents[handle]
    baseline_action = baseline_actions[handle]
    distance, slack = distance_and_slack(obs_builder, handle)
    baseline_target = target_metrics(obs_builder, handle, baseline_action)
    forced_target = target_metrics(obs_builder, handle, forced_action)
    logits = raw_logits(policy, observation)
    baseline_risk = future_risk(
        policy,
        obs_builder,
        handle,
        baseline_action,
        baseline_prefixes,
    )
    forced_risk = future_risk(
        policy,
        obs_builder,
        handle,
        forced_action,
        baseline_prefixes,
    )
    baseline_prefix = route_prefix_for_action(
        policy,
        obs_builder,
        handle,
        baseline_action,
    )
    forced_prefix = route_prefix_for_action(
        policy,
        obs_builder,
        handle,
        forced_action,
    )
    baseline_prefix_features = prefix_conflict_features(
        "baseline",
        baseline_prefix,
        baseline_prefixes,
        handle,
        env,
    )
    forced_prefix_features = prefix_conflict_features(
        "forced",
        forced_prefix,
        baseline_prefixes,
        handle,
        env,
    )
    prefix_delta_features = {}
    for suffix in (
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
    ):
        prefix_delta_features[f"forced_{suffix}_delta"] = (
            forced_prefix_features[f"forced_{suffix}"]
            - baseline_prefix_features[f"baseline_{suffix}"]
        )
    return {
        "seed": seed,
        "decision_index": decision_index,
        "env_time": int(env._elapsed_steps),
        "agent_id": int(handle),
        "state": state_name(agent.state),
        "position": str(agent.position),
        "direction": agent.direction,
        "speed": float(agent.speed_counter.speed),
        "distance": distance,
        "slack": slack,
        **observation_scalar_features(observation),
        **mask_values(observation),
        "baseline_action": baseline_action,
        "baseline_action_name": action_name(baseline_action),
        "forced_action": forced_action,
        "forced_action_name": action_name(forced_action),
        "baseline_target": baseline_target["target"],
        "baseline_target_direction": baseline_target["target_direction"],
        "baseline_target_distance": baseline_target["target_distance"],
        "baseline_target_occupied_by_other": baseline_target["target_occupied_by_other"],
        **direction_features(
            observation,
            "baseline_direction",
            baseline_target["target_direction"],
        ),
        "forced_target": forced_target["target"],
        "forced_target_direction": forced_target["target_direction"],
        "forced_target_distance": forced_target["target_distance"],
        "forced_target_occupied_by_other": forced_target["target_occupied_by_other"],
        **direction_features(
            observation,
            "forced_direction",
            forced_target["target_direction"],
        ),
        "forced_distance_delta": (
            forced_target["target_distance"] - baseline_target["target_distance"]
        ),
        **logit_features("policy", logits, baseline_action, forced_action),
        "baseline_corridor_len": corridor_len(obs_builder, handle, baseline_action),
        "forced_corridor_len": corridor_len(obs_builder, handle, forced_action),
        "baseline_future_head_on_risk": baseline_risk,
        "forced_future_head_on_risk": forced_risk,
        "future_head_on_risk_delta": forced_risk - baseline_risk,
        **baseline_prefix_features,
        **forced_prefix_features,
        **prefix_delta_features,
    }


def collect_decisions(
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env, obs_builder = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    policy = instantiate_policy(args.policy, args.policy_checkpoint)
    reward_values: list[float] = []
    positions: dict[int, list[Any]] = defaultdict(list)
    actions_by_agent: dict[int, list[int]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    decision_count = 0

    while int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
        observations_by_handle = dict(zip(handles, obs_list))
        actions = policy_actions(policy, handles, obs_list)
        prefixes = planned_prefixes(policy, obs_builder, actions)

        for handle in handles:
            if len(rows) >= args.max_decisions_per_seed * args.max_alternatives_per_decision:
                break
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

            baseline_target = target_metrics(obs_builder, handle, baseline_action)
            alternatives = sorted(
                alternatives,
                key=lambda action: forced_action_score(
                    obs_builder,
                    handle,
                    baseline_target,
                    action,
                    baseline_risk,
                    future_risk(policy, obs_builder, handle, action, prefixes),
                ),
            )
            decision_count += 1
            for forced_action in alternatives[: args.max_alternatives_per_decision]:
                rows.append(
                    decision_row(
                        args,
                        seed,
                        decision_count,
                        env,
                        obs_builder,
                        policy,
                        handle,
                        observation,
                        actions,
                        forced_action,
                        prefixes,
                    )
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
    return result, rows


def run_forced_episode(
    args: argparse.Namespace,
    seed: int,
    event: dict[str, Any],
) -> dict[str, Any]:
    env, _ = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    policy = instantiate_policy(args.policy, args.policy_checkpoint)
    reward_values: list[float] = []
    positions: dict[int, list[Any]] = defaultdict(list)
    actions_by_agent: dict[int, list[int]] = defaultdict(list)
    force_step = int(event["env_time"])
    force_handle = int(event["agent_id"])
    force_action = int(event["forced_action"])
    forced_applied = False

    while int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
        actions = policy_actions(policy, handles, obs_list)
        if int(env._elapsed_steps) == force_step and force_handle in actions:
            actions[force_handle] = force_action
            forced_applied = True

        observations, rewards_by_agent, dones, _ = env.step(actions)
        for handle, action in actions.items():
            actions_by_agent[handle].append(action)
        for handle in handles:
            positions[handle].append(env.agents[handle].position)
        reward_values.extend(float(rewards_by_agent.get(handle, 0.0)) for handle in handles)
        if done_all(dones):
            break

    result = final_result(args, seed, env, reward_values, positions, actions_by_agent)
    result["forced_applied"] = forced_applied
    return result


def annotate_outcome(
    row: dict[str, Any],
    baseline: dict[str, Any],
    forced: dict[str, Any],
) -> dict[str, Any]:
    baseline_failed = len(baseline["failed_agents"])
    forced_failed = len(forced["failed_agents"])
    return {
        **row,
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
        "forced_applied": forced["forced_applied"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "reward_wins": sum(int(row["reward_delta"] > 1e-9) for row in rows),
        "reward_losses": sum(int(row["reward_delta"] < -1e-9) for row in rows),
        "reward_ties": sum(int(abs(row["reward_delta"]) <= 1e-9) for row in rows),
        "success_wins": sum(int(row["success_delta"] > 1e-9) for row in rows),
        "success_losses": sum(int(row["success_delta"] < -1e-9) for row in rows),
        "success_ties": sum(int(abs(row["success_delta"]) <= 1e-9) for row in rows),
        "forced_not_applied": sum(int(not row["forced_applied"]) for row in rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one-step counterfactual actions at critical policy decisions."
        )
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--policy", default="submission.rerank_policy.MyPolicy")
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--episodes", type=int, default=5)
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
    parser.add_argument("--max-decisions-per-seed", type=int, default=8)
    parser.add_argument("--max-alternatives-per-decision", type=int, default=2)
    parser.add_argument("--min-corridor-len", type=int, default=4)
    parser.add_argument("--critical-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-do-nothing", action="store_true")
    parser.add_argument(
        "--forced-actions",
        help=(
            "Comma-separated action IDs or names to consider as alternatives, "
            "for example 1,2,3 or LEFT,FORWARD,RIGHT."
        ),
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    args.forced_action_ids = parse_forced_actions(args.forced_actions)
    return args


def parse_forced_actions(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    mapping = {
        "N": 0,
        "DO_NOTHING": 0,
        "LEFT": 1,
        "MOVE_LEFT": 1,
        "L": 1,
        "FORWARD": 2,
        "MOVE_FORWARD": 2,
        "F": 2,
        "RIGHT": 3,
        "MOVE_RIGHT": 3,
        "R": 3,
        "STOP": 4,
        "STOP_MOVING": 4,
        "S": 4,
    }
    actions = []
    for item in value.split(","):
        token = item.strip().upper()
        if not token:
            continue
        if token.isdigit():
            action = int(token)
        else:
            if token not in mapping:
                raise argparse.ArgumentTypeError(f"Unknown action token: {item}")
            action = mapping[token]
        if action < 0 or action > 4:
            raise argparse.ArgumentTypeError(f"Action out of range 0..4: {item}")
        actions.append(action)
    return tuple(dict.fromkeys(actions))


def main() -> int:
    args = parse_args()
    seeds = (
        [int(item) for item in args.seeds.split(",") if item.strip()]
        if args.seeds
        else [args.seed + index for index in range(args.episodes)]
    )
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        baseline, events = collect_decisions(args, seed)
        print(
            f"seed={seed} baseline={baseline['normalized_reward']:.6g}/"
            f"{baseline['success_rate']:.6g} events={len(events)}",
            flush=True,
        )
        for event in events:
            forced = run_forced_episode(args, seed, event)
            row = annotate_outcome(event, baseline, forced)
            rows.append(row)
            print(
                "  step={env_time} agent={agent_id} "
                "{baseline_action_name}->{forced_action_name} "
                "reward_delta={reward_delta:.6g} success_delta={success_delta:.6g} "
                "failed_delta={failed_agents_delta}".format(**row),
                flush=True,
            )

    summary = summarize(rows)
    print(
        "\nSummary: "
        f"rows={summary['rows']} "
        f"reward={summary['reward_wins']}/{summary['reward_losses']}/"
        f"{summary['reward_ties']} "
        f"success={summary['success_wins']}/{summary['success_losses']}/"
        f"{summary['success_ties']} "
        f"forced_not_applied={summary['forced_not_applied']}"
    )
    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    if args.output_json is not None:
        write_json(args.output_json, rows, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
