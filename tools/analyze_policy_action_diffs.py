#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env_action import RailEnvActions

from tools.evaluate_sampled import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_REWARDS,
    load_sampling_env_generator,
    load_symbol,
    normalized_reward,
    observation_list,
    repo_root,
)


ACTION_COLUMNS = ("N", "L", "F", "R", "S")


def instantiate_policy(policy_path: str, checkpoint: Path | None) -> Any:
    policy_cls = load_symbol(policy_path)
    if checkpoint is None:
        return policy_cls()
    try:
        return policy_cls(checkpoint_path=str(checkpoint))
    except TypeError as exc:
        raise TypeError(
            f"{policy_path} does not accept a checkpoint_path constructor argument."
        ) from exc


def action_id(action: Any) -> int:
    if hasattr(action, "value"):
        return int(action.value)
    return int(action)


def action_name(action: int | None) -> str:
    if action is None:
        return ""
    try:
        return RailEnvActions(int(action)).name
    except ValueError:
        return str(action)


def done_all(dones: Any) -> bool:
    if isinstance(dones, dict):
        return bool(dones.get("__all__", False))
    return bool(dones)


def state_name(state: Any) -> str:
    return getattr(state, "name", str(state))


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


def policy_actions(policy: Any, handles: list[int], observations: list[Any]) -> dict[int, int]:
    return {
        handle: action_id(action)
        for handle, action in policy.act_many(handles, observations).items()
    }


def raw_policy_actions(policy: Any, handles: list[int], observations: list[Any]) -> dict[int, int]:
    raw_policy = getattr(policy, "rl_policy", None)
    if raw_policy is None:
        return {}
    return {
        handle: action_id(action)
        for handle, action in raw_policy.act_many(handles, observations).items()
    }


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def distance_and_slack(obs_builder: Any, handle: int) -> tuple[float, float]:
    try:
        distance = float(obs_builder._current_distance_to_waypoint(handle))
        slack = float(obs_builder._deadline_slack(handle, distance))
    except Exception:
        return float("nan"), float("nan")
    return distance, slack


def target_metrics(
    obs_builder: Any,
    handle: int,
    action: int,
) -> dict[str, Any]:
    try:
        target, target_direction = obs_builder._action_target(handle, action)
    except Exception:
        target, target_direction = None, None
    target_distance = float("nan")
    occupied = False
    if target is not None and target_direction is not None:
        try:
            distance_map = obs_builder._get_distance_map(handle)
            target_distance = float(distance_map[target[0], target[1], target_direction])
        except Exception:
            target_distance = float("nan")
        try:
            occupied = bool(obs_builder._occupied_by_other(target, handle))
        except Exception:
            occupied = False
    return {
        "target": str(target),
        "target_direction": target_direction,
        "target_distance": target_distance,
        "target_occupied_by_other": occupied,
    }


def corridor_len(obs_builder: Any, handle: int, action: int) -> int:
    try:
        return len(obs_builder._corridor_edges_for_action(handle, action))
    except Exception:
        return 0


def planned_prefixes(policy: Any, obs_builder: Any, actions: dict[int, int]) -> dict[int, Any]:
    if not hasattr(policy, "_route_prefix_for_action"):
        return {}
    lookahead = int(getattr(policy, "FUTURE_RERANK_LOOKAHEAD_CELLS", 45))
    return {
        handle: policy._route_prefix_for_action(obs_builder, handle, action, lookahead)
        for handle, action in actions.items()
    }


def future_risk(
    policy: Any,
    obs_builder: Any,
    handle: int,
    action: int,
    prefixes: dict[int, Any],
) -> float:
    if not hasattr(policy, "_future_head_on_risk") or not prefixes:
        return float("nan")
    try:
        return float(policy._future_head_on_risk(obs_builder, handle, action, prefixes))
    except Exception:
        return float("nan")


def mask_values(observation: Any) -> dict[str, float]:
    values = np.asarray(observation, dtype=np.float32)[-len(ACTION_COLUMNS) :]
    return {
        f"mask_{name}": float(value)
        for name, value in zip(ACTION_COLUMNS, values)
    }


def diff_row(
    args: argparse.Namespace,
    seed: int,
    env: Any,
    obs_builder: Any,
    handle: int,
    observation: Any,
    baseline_policy: Any,
    candidate_policy: Any,
    baseline_raw_actions: dict[int, int],
    candidate_raw_actions: dict[int, int],
    baseline_actions: dict[int, int],
    candidate_actions: dict[int, int],
) -> dict[str, Any]:
    agent = env.agents[handle]
    baseline_action = baseline_actions[handle]
    candidate_action = candidate_actions[handle]
    distance, slack = distance_and_slack(obs_builder, handle)
    baseline_target = target_metrics(obs_builder, handle, baseline_action)
    candidate_target = target_metrics(obs_builder, handle, candidate_action)
    prefixes = planned_prefixes(baseline_policy, obs_builder, baseline_actions)
    return {
        "seed": seed,
        "env_time": int(env._elapsed_steps),
        "has_diff": True,
        "agent_id": int(handle),
        "state": state_name(agent.state),
        "position": str(agent.position),
        "direction": agent.direction,
        "speed": safe_float(agent.speed_counter.speed),
        "distance": distance,
        "slack": slack,
        **mask_values(observation),
        "baseline_raw_action": baseline_raw_actions.get(handle),
        "baseline_raw_action_name": action_name(baseline_raw_actions.get(handle)),
        "candidate_raw_action": candidate_raw_actions.get(handle),
        "candidate_raw_action_name": action_name(candidate_raw_actions.get(handle)),
        "baseline_action": baseline_action,
        "baseline_action_name": action_name(baseline_action),
        "candidate_action": candidate_action,
        "candidate_action_name": action_name(candidate_action),
        "baseline_target": baseline_target["target"],
        "baseline_target_direction": baseline_target["target_direction"],
        "baseline_target_distance": baseline_target["target_distance"],
        "baseline_target_occupied_by_other": baseline_target["target_occupied_by_other"],
        "candidate_target": candidate_target["target"],
        "candidate_target_direction": candidate_target["target_direction"],
        "candidate_target_distance": candidate_target["target_distance"],
        "candidate_target_occupied_by_other": candidate_target["target_occupied_by_other"],
        "candidate_distance_delta": (
            candidate_target["target_distance"] - baseline_target["target_distance"]
        ),
        "baseline_corridor_len": corridor_len(obs_builder, handle, baseline_action),
        "candidate_corridor_len": corridor_len(obs_builder, handle, candidate_action),
        "baseline_future_head_on_risk": future_risk(
            baseline_policy,
            obs_builder,
            handle,
            baseline_action,
            prefixes,
        ),
        "candidate_future_head_on_risk": future_risk(
            baseline_policy,
            obs_builder,
            handle,
            candidate_action,
            prefixes,
        ),
    }


def no_diff_row(seed: int, env: Any, reward_values: list[float]) -> dict[str, Any]:
    success_rate = sum(int(agent.state == 6) for agent in env.agents) / env.get_num_agents()
    return {
        "seed": seed,
        "env_time": int(env._elapsed_steps),
        "has_diff": False,
        "agent_id": "",
        "state": "",
        "position": "",
        "direction": "",
        "speed": "",
        "distance": "",
        "slack": "",
        "success_rate": success_rate,
        "normalized_reward": normalized_reward(env, reward_values),
    }


def analyze_seed(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    env, obs_builder = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    baseline_policy = instantiate_policy(args.baseline_policy, args.baseline_checkpoint)
    candidate_policy = instantiate_policy(args.candidate_policy, args.candidate_checkpoint)
    reward_values: list[float] = []

    while int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
        baseline_raw_actions = raw_policy_actions(baseline_policy, handles, obs_list)
        candidate_raw_actions = raw_policy_actions(candidate_policy, handles, obs_list)
        baseline_actions = policy_actions(baseline_policy, handles, obs_list)
        candidate_actions = policy_actions(candidate_policy, handles, obs_list)

        for handle, observation in zip(handles, obs_list):
            if baseline_actions.get(handle) != candidate_actions.get(handle):
                return diff_row(
                    args,
                    seed,
                    env,
                    obs_builder,
                    handle,
                    observation,
                    baseline_policy,
                    candidate_policy,
                    baseline_raw_actions,
                    candidate_raw_actions,
                    baseline_actions,
                    candidate_actions,
                )

        observations, rewards_by_agent, dones, _ = env.step(baseline_actions)
        reward_values.extend(
            float(rewards_by_agent.get(handle, 0.0))
            for handle in handles
        )
        if done_all(dones):
            break
    return no_diff_row(seed, env, reward_values)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the first synchronized action difference between two policies."
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
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = (
        [int(item) for item in args.seeds.split(",") if item.strip()]
        if args.seeds
        else [args.seed + index for index in range(args.episodes)]
    )
    rows = []
    for seed in seeds:
        row = analyze_seed(args, seed)
        rows.append(row)
        if row["has_diff"]:
            print(
                "seed={seed} step={env_time} agent={agent_id} "
                "baseline={baseline_action_name} candidate={candidate_action_name} "
                "raw={baseline_raw_action_name}->{candidate_raw_action_name} "
                "slack={slack:.6g} dist_delta={candidate_distance_delta:.6g}".format(
                    **row
                ),
                flush=True,
            )
        else:
            print(
                f"seed={seed} no_diff reward={row['normalized_reward']:.6g} "
                f"success={row['success_rate']:.6g}",
                flush=True,
            )

    changed = [row for row in rows if row["has_diff"]]
    print(
        f"\nSummary: episodes={len(rows)} changed={len(changed)} "
        f"unchanged={len(rows) - len(changed)}"
    )
    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    if args.output_json is not None:
        write_json(args.output_json, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
