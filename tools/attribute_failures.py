#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env_action import RailEnvActions

from evaluate_sampled import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_POLICY,
    DEFAULT_REWARDS,
    action_id,
    load_sampling_env_generator,
    load_symbol,
    normalized_reward,
    observation_list,
    repo_root,
    state_name,
)


WAIT_ACTIONS = {0, 4}
MOVE_ACTIONS = {1, 2, 3}
ACTION_NAMES = {
    int(action.value): action.name
    for action in RailEnvActions
}


@dataclass
class AgentAttribution:
    seed: int
    agent_id: int
    latest_arrival: int | None
    state_counts: Counter[str] = field(default_factory=Counter)
    action_counts: Counter[int] = field(default_factory=Counter)
    off_map_wait_steps: int = 0
    malfunction_steps: int = 0
    policy_wait_steps: int = 0
    forced_wait_steps: int = 0
    masked_no_move_steps: int = 0
    movement_steps: int = 0
    detour_steps: int = 0
    large_detour_steps: int = 0
    total_distance_increase: float = 0.0
    max_distance_increase: float = 0.0
    min_slack: float | None = None
    final_slack: float | None = None
    final_distance: float | None = None
    last_position: Any = None
    last_position_change_time: int = 0

    def update_slack(self, slack: float) -> None:
        if not np.isfinite(slack):
            return
        self.final_slack = float(slack)
        if self.min_slack is None or slack < self.min_slack:
            self.min_slack = float(slack)

    def update_distance(self, distance: float) -> None:
        if np.isfinite(distance):
            self.final_distance = float(distance)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute sampled Flatland failures to malfunctions, waits, masks, "
            "detours and deadline pressure."
        )
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--only-failed", action="store_true")
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def make_env(args: argparse.Namespace, seed: int) -> tuple[Any, dict[int, Any], Any]:
    obs_builder = load_symbol(args.obs_builder)()
    rewards = load_symbol(args.rewards)()
    env, _ = RailEnvPersister.load_new(
        args.base_state_pkl,
        obs_builder=obs_builder,
        rewards=rewards,
    )
    if args.num_agents is not None:
        env.number_of_agents = args.num_agents
    env = load_sampling_env_generator()(
        env,
        line_length=args.line_length,
        scene=args.scene,
    )
    observations, _ = env.reset(random_seed=seed)
    return env, observations, obs_builder


def mask_values(observation: Any) -> np.ndarray:
    if observation is None:
        return np.zeros(5, dtype=np.float32)
    values = np.asarray(observation, dtype=np.float32)
    if values.shape[0] < 5:
        return np.zeros(5, dtype=np.float32)
    return values[-5:]


def distance_and_slack(obs_builder: Any, handle: int) -> tuple[float, float]:
    try:
        distance = float(obs_builder._current_distance_to_waypoint(handle))
        slack = float(obs_builder._deadline_slack(handle, distance))
        return distance, slack
    except Exception:
        return float("nan"), float("nan")


def action_distance_delta(
    obs_builder: Any,
    handle: int,
    action: int,
    current_distance: float,
) -> float:
    if action not in MOVE_ACTIONS or not np.isfinite(current_distance):
        return 0.0
    try:
        target, target_direction = obs_builder._action_target(handle, action)
        if target is None or target_direction is None:
            return 0.0
        distance_map = obs_builder._get_distance_map(handle)
        new_distance = distance_map[target[0], target[1], target_direction]
        if not np.isfinite(new_distance):
            return 0.0
        return float(new_distance - current_distance)
    except Exception:
        return 0.0


def update_agent_metrics(
    metrics: AgentAttribution,
    env: Any,
    obs_builder: Any,
    handle: int,
    observation: Any,
    action: int,
) -> None:
    agent = env.agents[handle]
    current_state = state_name(agent.state)
    metrics.state_counts[current_state] += 1
    metrics.action_counts[action] += 1

    position_key = tuple(agent.position) if agent.position is not None else None
    if position_key != metrics.last_position:
        metrics.last_position = position_key
        metrics.last_position_change_time = int(env._elapsed_steps)

    distance, slack = distance_and_slack(obs_builder, handle)
    metrics.update_distance(distance)
    metrics.update_slack(slack)

    if current_state in {"WAITING", "READY_TO_DEPART", "MALFUNCTION_OFF_MAP"}:
        metrics.off_map_wait_steps += 1
    if "MALFUNCTION" in current_state and current_state != "MALFUNCTION_OFF_MAP":
        metrics.malfunction_steps += 1

    mask = mask_values(observation)
    movement_available = bool(np.any(mask[list(MOVE_ACTIONS)] >= 0.5))
    if not movement_available:
        metrics.masked_no_move_steps += 1

    if action in WAIT_ACTIONS and current_state in {"MOVING", "STOPPED"}:
        if movement_available:
            metrics.policy_wait_steps += 1
        else:
            metrics.forced_wait_steps += 1

    if action in MOVE_ACTIONS:
        metrics.movement_steps += 1
        delta = action_distance_delta(obs_builder, handle, action, distance)
        if delta > 0:
            metrics.detour_steps += 1
            metrics.total_distance_increase += delta
            metrics.max_distance_increase = max(metrics.max_distance_increase, delta)
            if delta > 10:
                metrics.large_detour_steps += 1


def diagnosis(row: dict[str, Any]) -> str:
    if row["success"]:
        return "success"
    missed_by = row["missed_by"] or 0
    if row["malfunction_steps"] >= max(1, missed_by):
        return "malfunction_limited"
    if row["large_detour_steps"] > 0:
        return "large_detour_or_wrong_branch"
    if row["policy_wait_steps"] >= max(1, missed_by):
        return "policy_wait_limited"
    if row["forced_wait_steps"] >= max(1, missed_by):
        return "blocked_or_mask_limited"
    if row["final_distance"] and row["final_distance"] > 0:
        return "route_or_deadline_limited"
    return "unknown"


def flatten_agent_row(
    env: Any,
    seed: int,
    episode_index: int,
    metrics: AgentAttribution,
) -> dict[str, Any]:
    agent = env.agents[metrics.agent_id]
    success = bool(agent.state == 6)
    latest_arrival = metrics.latest_arrival
    missed_by = (
        max(0, int(env._elapsed_steps) - int(latest_arrival))
        if latest_arrival is not None and not success
        else 0
    )
    row = {
        "episode_index": episode_index,
        "seed": seed,
        "agent_id": metrics.agent_id,
        "success": success,
        "final_state": state_name(agent.state),
        "env_time": int(env._elapsed_steps),
        "latest_arrival": latest_arrival,
        "missed_by": missed_by,
        "final_position": str(agent.position),
        "last_position": str(metrics.last_position),
        "stationary_tail": max(
            0,
            int(env._elapsed_steps) - metrics.last_position_change_time,
        ),
        "final_distance": metrics.final_distance,
        "final_slack": metrics.final_slack,
        "min_slack": metrics.min_slack,
        "off_map_wait_steps": metrics.off_map_wait_steps,
        "malfunction_steps": metrics.malfunction_steps,
        "policy_wait_steps": metrics.policy_wait_steps,
        "forced_wait_steps": metrics.forced_wait_steps,
        "masked_no_move_steps": metrics.masked_no_move_steps,
        "movement_steps": metrics.movement_steps,
        "detour_steps": metrics.detour_steps,
        "large_detour_steps": metrics.large_detour_steps,
        "total_distance_increase": round(metrics.total_distance_increase, 6),
        "max_distance_increase": round(metrics.max_distance_increase, 6),
        "action_counts": {
            ACTION_NAMES.get(action, str(action)): count
            for action, count in sorted(metrics.action_counts.items())
        },
        "state_counts": dict(sorted(metrics.state_counts.items())),
    }
    row["diagnosis"] = diagnosis(row)
    return row


def run_episode(
    args: argparse.Namespace,
    seed: int,
    episode_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env, observations, obs_builder = make_env(args, seed)
    policy = load_symbol(args.policy)()
    handles = list(env.get_agent_handles())
    agent_metrics = {
        handle: AgentAttribution(
            seed=seed,
            agent_id=handle,
            latest_arrival=env.agents[handle].latest_arrival,
        )
        for handle in handles
    }
    reward_values: list[float] = []
    done = False

    while not done and env._elapsed_steps < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
        actions = {
            handle: action_id(action)
            for handle, action in policy.act_many(handles, obs_list).items()
        }
        for handle, observation in zip(handles, obs_list):
            update_agent_metrics(
                agent_metrics[handle],
                env,
                obs_builder,
                handle,
                observation,
                actions.get(handle, 0),
            )

        observations, rewards_by_agent, dones, _ = env.step(actions)
        reward_values.extend(
            float(rewards_by_agent.get(handle, 0.0))
            for handle in handles
        )
        done = bool(dones.get("__all__", False))

    episode_row = {
        "episode_index": episode_index,
        "seed": seed,
        "env_time": int(env._elapsed_steps),
        "success_rate": sum(int(agent.state == 6) for agent in env.agents)
        / env.get_num_agents(),
        "normalized_reward": normalized_reward(env, reward_values),
        "num_agents": env.get_num_agents(),
        "max_episode_steps": env._max_episode_steps,
        "line_length": args.line_length,
        "scene": args.scene or "scene_5",
    }
    agent_rows = [
        flatten_agent_row(env, seed, episode_index, agent_metrics[handle])
        for handle in sorted(agent_metrics)
    ]
    if args.only_failed:
        agent_rows = [row for row in agent_rows if not row["success"]]
    return episode_row, agent_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    episode_rows = []
    agent_rows = []

    for episode_index in range(args.episodes):
        seed = args.seed + episode_index
        episode_row, rows = run_episode(args, seed, episode_index)
        episode_rows.append(episode_row)
        agent_rows.extend(rows)
        print(
            f"seed={seed} reward={episode_row['normalized_reward']:.6g} "
            f"success={episode_row['success_rate']:.6g} "
            f"failed={sum(not row['success'] for row in rows)}",
            flush=True,
        )
        for row in rows:
            if row["success"] and args.only_failed:
                continue
            print(
                "  "
                f"agent={row['agent_id']} success={row['success']} "
                f"diagnosis={row['diagnosis']} final_state={row['final_state']} "
                f"missed_by={row['missed_by']} final_distance={row['final_distance']} "
                f"malf={row['malfunction_steps']} policy_wait={row['policy_wait_steps']} "
                f"forced_wait={row['forced_wait_steps']} "
                f"detours={row['detour_steps']} max_detour={row['max_distance_increase']}",
                flush=True,
            )

    failed_rows = [row for row in agent_rows if not row["success"]]
    diagnosis_counts = Counter(row["diagnosis"] for row in failed_rows)
    reward_mean = float(np.mean([row["normalized_reward"] for row in episode_rows]))
    success_mean = float(np.mean([row["success_rate"] for row in episode_rows]))
    print(
        "\nSummary: "
        f"episodes={len(episode_rows)} reward_mean={reward_mean:.6g} "
        f"success_rate_mean={success_mean:.6g} failed_agents={len(failed_rows)} "
        f"diagnoses={dict(sorted(diagnosis_counts.items()))}",
        flush=True,
    )

    if args.output_csv is not None:
        write_csv(args.output_csv, agent_rows)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(
                {
                    "episodes": episode_rows,
                    "agents": agent_rows,
                },
                handle,
                indent=2,
            )
            handle.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
