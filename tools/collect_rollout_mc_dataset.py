#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env_action import RailEnvActions

from submission import runtime_context
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


def action_id(action: Any) -> int:
    if hasattr(action, "value"):
        return int(action.value)
    return int(action)


def action_name(action: int) -> str:
    try:
        return RailEnvActions(int(action)).name
    except Exception:
        return str(action)


def state_name(state: Any) -> str:
    return getattr(state, "name", str(state))


def parse_seed_list(value: str | None) -> list[int]:
    if not value:
        return []
    seeds = []
    for token in value.split(","):
        item = token.strip()
        if item:
            seeds.append(int(item))
    return list(dict.fromkeys(seeds))


def instantiate_policy(policy_path: str, checkpoint: Path | None) -> Any:
    policy_cls = load_symbol(policy_path)
    if checkpoint is None:
        return policy_cls()
    try:
        return policy_cls(checkpoint_path=str(checkpoint))
    except TypeError as exc:
        raise TypeError(
            f"{policy_path} does not accept --policy-checkpoint."
        ) from exc


def mask_values(observation: Any, n_actions: int) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape[0] >= n_actions:
        return obs[-n_actions:].astype(np.float32)
    return np.ones(n_actions, dtype=np.float32)


def distance_and_slack(obs_builder: Any, handle: int) -> tuple[float, float]:
    try:
        distance = float(obs_builder._current_distance_to_waypoint(handle))
        slack = float(obs_builder._deadline_slack(handle, distance))
        return distance, slack
    except Exception:
        return float("nan"), float("nan")


def target_distance(obs_builder: Any, handle: int, action: int) -> float:
    try:
        target, target_direction = obs_builder._action_target(handle, action)
        if target is None or target_direction is None:
            return float("nan")
        distance_map = obs_builder._get_distance_map(handle)
        return float(distance_map[target[0], target[1], target_direction])
    except Exception:
        return float("nan")


def done_all(dones: Any) -> bool:
    if isinstance(dones, dict):
        return bool(dones.get("__all__", False))
    return bool(dones)


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


def observation_features(observation: Any, args: argparse.Namespace) -> dict[str, float]:
    if not args.include_observation_features:
        return {}
    obs = np.asarray(observation, dtype=np.float32)
    limit = min(args.max_observation_features, int(obs.shape[0]))
    return {f"obs_{index:03d}": float(obs[index]) for index in range(limit)}


def collect_episode(args: argparse.Namespace, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env, obs_builder = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    runtime_context.set_seed(seed)
    policy = instantiate_policy(args.policy, args.policy_checkpoint)
    episode_rows: list[dict[str, Any]] = []
    reward_values: list[float] = []
    done = False

    while not done and int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
        actions = {
            handle: action_id(action)
            for handle, action in policy.act_many(handles, obs_list).items()
        }
        step_rows: list[dict[str, Any]] = []
        elapsed = int(env._elapsed_steps)

        for handle, observation in zip(handles, obs_list):
            if handle not in actions:
                continue
            agent = env.agents[handle]
            action = int(actions[handle])
            masks = mask_values(observation, args.n_actions)
            distance, slack = distance_and_slack(obs_builder, handle)
            row: dict[str, Any] = {
                "seed": seed,
                "scene": args.scene or "scene_5",
                "line_length": args.line_length,
                "env_time": elapsed,
                "agent_id": int(handle),
                "state": state_name(agent.state),
                "position": str(agent.position),
                "direction": int(agent.direction) if agent.direction is not None else "",
                "speed": float(agent.speed_counter.speed),
                "distance": distance,
                "slack": slack,
                "action": action,
                "action_name": action_name(action),
                "action_valid": float(0 <= action < len(masks) and masks[action] >= 0.5),
                "target_distance": target_distance(obs_builder, handle, action),
            }
            for index, name in enumerate(ACTION_COLUMNS[: args.n_actions]):
                row[f"mask_{name}"] = float(masks[index]) if index < len(masks) else 0.0
            row.update(observation_features(observation, args))
            step_rows.append(row)

        observations, rewards_by_agent, dones, _ = env.step(actions)
        for row in step_rows:
            handle = int(row["agent_id"])
            row["immediate_reward"] = float(rewards_by_agent.get(handle, 0.0))
        reward_values.extend(float(rewards_by_agent.get(handle, 0.0)) for handle in handles)
        episode_rows.extend(step_rows)
        done = done_all(dones)

    agent_success = {
        handle: float(agent.state == 6 or getattr(agent.state, "name", "") == "DONE")
        for handle, agent in enumerate(env.agents)
    }
    team_success = sum(agent_success.values()) / max(1, len(agent_success))
    episode_reward = normalized_reward(env, reward_values)
    failed_agents = [handle for handle, success in agent_success.items() if success < 0.5]
    episode_summary = {
        "seed": seed,
        "scene": args.scene or "scene_5",
        "line_length": args.line_length,
        "env_time": int(env._elapsed_steps),
        "max_episode_steps": int(env._max_episode_steps),
        "normalized_reward": episode_reward,
        "team_success": team_success,
        "failed_agents": failed_agents,
        "failed_agents_count": len(failed_agents),
        "rows": len(episode_rows),
    }
    for row in episode_rows:
        handle = int(row["agent_id"])
        row["agent_success"] = agent_success.get(handle, 0.0)
        row["agent_failure"] = 1.0 - row["agent_success"]
        row["team_success"] = team_success
        row["episode_normalized_reward"] = episode_reward
        row["episode_env_time"] = episode_summary["env_time"]
        row["failed_agents_count"] = episode_summary["failed_agents_count"]
    return episode_rows, episode_summary


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect Monte-Carlo rollout action rows. Each chosen action is "
            "labeled with full-episode agent/team success and normalized reward."
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
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", help="Comma-separated exact seed list.")
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--include-observation-features", action="store_true")
    parser.add_argument("--max-observation-features", type=int, default=96)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = parse_seed_list(args.seeds) if args.seeds else [
        args.seed + index for index in range(args.episodes)
    ]
    rows: list[dict[str, Any]] = []
    episodes = []
    for seed in seeds:
        episode_rows, summary = collect_episode(args, seed)
        rows.extend(episode_rows)
        episodes.append(summary)
        print(
            f"seed={seed} rows={summary['rows']} "
            f"reward={summary['normalized_reward']:.6g} "
            f"success={summary['team_success']:.6g} "
            f"failed={summary['failed_agents_count']}",
            flush=True,
        )

    write_csv(args.output_csv, rows)
    action_counts = Counter(str(row.get("action_name", "")) for row in rows)
    summary = {
        "config": vars(args),
        "episodes": len(episodes),
        "rows": len(rows),
        "reward_mean": (
            sum(float(item["normalized_reward"]) for item in episodes) / len(episodes)
            if episodes
            else 0.0
        ),
        "team_success_mean": (
            sum(float(item["team_success"]) for item in episodes) / len(episodes)
            if episodes
            else 0.0
        ),
        "action_counts": dict(sorted(action_counts.items())),
        "episode_summaries": episodes,
        "output_csv": str(args.output_csv),
    }
    print(
        "summary "
        f"episodes={summary['episodes']} rows={summary['rows']} "
        f"reward_mean={summary['reward_mean']:.6g} "
        f"team_success_mean={summary['team_success_mean']:.6g} "
        f"actions={summary['action_counts']}"
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(summary, handle, indent=2, default=json_default)
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
