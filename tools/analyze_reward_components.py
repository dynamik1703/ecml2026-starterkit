#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rewards import DefaultPenalties
from flatland.envs.step_utils.states import TrainState

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay sampled Flatland episodes and explain ECML2026 reward "
            "components per agent and intermediate waypoint."
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
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument(
        "--disable-temporal-locks",
        action="store_true",
        help="Set HybridPolicy temporal corridor threshold very high for A/B runs.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-agent-csv", type=Path)
    parser.add_argument("--output-stop-csv", type=Path)
    return parser.parse_args()


def instantiate_policy(args: argparse.Namespace) -> Any:
    policy = load_symbol(args.policy)()
    if args.disable_temporal_locks and hasattr(policy, "TEMPORAL_CORRIDOR_MIN_EDGES"):
        policy.TEMPORAL_CORRIDOR_MIN_EDGES = 1_000_000
    return policy


def make_env(args: argparse.Namespace) -> tuple[Any, dict[int, Any], Any, Any]:
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
    return env, obs_builder, rewards, instantiate_policy(args)


def done_all(dones: Any) -> bool:
    if isinstance(dones, dict):
        return bool(dones.get("__all__", False))
    return bool(dones)


def waypoint_key(waypoint: Any) -> str:
    return f"{tuple(waypoint.position)}/{waypoint.direction}"


def train_state_names(states: set[Any]) -> list[str]:
    return sorted(state_name(state) for state in states)


def final_components(env: Any, rewards: Any, handle: int) -> dict[str, float]:
    proxy = rewards._proxy
    components = proxy.empty()
    proxy._agent_done_or_max_episode_steps_reward(
        env.agents[handle],
        env.distance_map,
        int(env._elapsed_steps),
        components,
    )
    return {key: float(value) for key, value in components.items()}


def best_stop_service(
    rewards: Any,
    handle: int,
    alternatives: list[Any],
    latest_arrival: int,
    earliest_departure: int,
) -> dict[str, Any]:
    proxy = rewards._proxy
    arrivals_by_wp = proxy.arrivals[handle]
    departures_by_wp = proxy.departures[handle]
    states_by_wp = proxy.states[handle]

    alternative_rows = []
    best_service = None
    any_arrival = False
    any_stopped = False

    for waypoint in alternatives:
        arrivals = list(arrivals_by_wp.get(waypoint, []))
        departures = list(departures_by_wp.get(waypoint, []))
        states = set(states_by_wp.get(waypoint, set()))
        stopped = TrainState.STOPPED in states
        any_arrival = any_arrival or bool(arrivals)
        any_stopped = any_stopped or stopped

        services = []
        for arrival, departure in zip(arrivals, departures + [None]):
            late_penalty = 0.5 * min(latest_arrival - arrival, 0)
            early_penalty = (
                0.5 * min(departure - earliest_departure, 0)
                if departure is not None
                else 0.0
            )
            service = {
                "arrival": arrival,
                "departure": departure,
                "late_penalty": float(late_penalty),
                "early_departure_penalty": float(early_penalty),
                "total_penalty": float(late_penalty + early_penalty),
            }
            services.append(service)
            if stopped and (
                best_service is None
                or service["total_penalty"] > best_service["total_penalty"]
            ):
                best_service = {
                    **service,
                    "waypoint": waypoint_key(waypoint),
                }

        alternative_rows.append(
            {
                "waypoint": waypoint_key(waypoint),
                "arrivals": arrivals,
                "departures": departures,
                "states": train_state_names(states),
                "stopped": stopped,
                "services": services,
            }
        )

    served = best_service is not None
    if served:
        diagnosis = "served"
    elif any_arrival and not any_stopped:
        diagnosis = "passed_without_stop"
    else:
        diagnosis = "not_reached"

    return {
        "served": served,
        "diagnosis": diagnosis,
        "best_service": best_service,
        "alternatives": alternative_rows,
    }


def intermediate_rows(
    seed: int,
    episode_index: int,
    env: Any,
    rewards: Any,
    handle: int,
) -> list[dict[str, Any]]:
    agent = env.agents[handle]
    rows = []
    for waypoint_index, (alternatives, latest_arrival, earliest_departure) in enumerate(
        zip(
            agent.waypoints[1:-1],
            agent.waypoints_latest_arrival[1:-1],
            agent.waypoints_earliest_departure[1:-1],
        ),
        start=1,
    ):
        service = best_stop_service(
            rewards,
            handle,
            alternatives,
            latest_arrival,
            earliest_departure,
        )
        best = service["best_service"] or {}
        rows.append(
            {
                "episode_index": episode_index,
                "seed": seed,
                "agent_id": handle,
                "waypoint_index": waypoint_index,
                "latest_arrival": latest_arrival,
                "earliest_departure": earliest_departure,
                "served": service["served"],
                "diagnosis": service["diagnosis"],
                "best_waypoint": best.get("waypoint"),
                "best_arrival": best.get("arrival"),
                "best_departure": best.get("departure"),
                "late_penalty": best.get("late_penalty", 0.0),
                "early_departure_penalty": best.get(
                    "early_departure_penalty",
                    0.0,
                ),
                "total_penalty": best.get("total_penalty", -50.0),
                "alternatives": service["alternatives"],
            }
        )
    return rows


def agent_row(
    seed: int,
    episode_index: int,
    env: Any,
    rewards: Any,
    cumulative_reward: dict[int, float],
    handle: int,
) -> dict[str, Any]:
    agent = env.agents[handle]
    components = final_components(env, rewards, handle)
    component_sum = sum(components.values())
    cumulative = float(cumulative_reward[handle])
    step_transition_penalty = cumulative - component_sum
    collision_factor = float(getattr(rewards, "collision_factor", 0.0) or 0.0)
    return {
        "episode_index": episode_index,
        "seed": seed,
        "agent_id": handle,
        "state": state_name(agent.state),
        "success": bool(agent.state == 6 or state_name(agent.state) in {"DONE", "DONE_REMOVED"}),
        "arrival_time": agent.arrival_time,
        "latest_arrival": agent.latest_arrival,
        "env_time": int(env._elapsed_steps),
        "cumulative_reward": cumulative,
        "final_component_sum": float(component_sum),
        "step_transition_penalty": float(step_transition_penalty),
        "estimated_collision_stop_events": (
            float(-step_transition_penalty / collision_factor)
            if collision_factor > 0.0
            else None
        ),
        **components,
    }


def run_episode(args: argparse.Namespace, seed: int, episode_index: int) -> dict[str, Any]:
    env, obs_builder, rewards, policy = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    handles = list(env.get_agent_handles())
    cumulative_reward: dict[int, float] = defaultdict(float)
    reward_values: list[float] = []

    while int(env._elapsed_steps) < env._max_episode_steps:
        actions = {
            handle: action_id(action)
            for handle, action in policy.act_many(
                handles,
                observation_list(observations, handles),
            ).items()
        }
        observations, rewards_by_agent, dones, _ = env.step(actions)
        for handle in handles:
            reward = float(rewards_by_agent.get(handle, 0.0))
            cumulative_reward[handle] += reward
            reward_values.append(reward)
        if done_all(dones):
            break

    agent_rows = [
        agent_row(seed, episode_index, env, rewards, cumulative_reward, handle)
        for handle in handles
    ]
    stop_rows = [
        row
        for handle in handles
        for row in intermediate_rows(seed, episode_index, env, rewards, handle)
    ]
    return {
        "episode": {
            "episode_index": episode_index,
            "seed": seed,
            "env_time": int(env._elapsed_steps),
            "success_rate": sum(int(row["success"]) for row in agent_rows)
            / env.get_num_agents(),
            "normalized_reward": normalized_reward(env, reward_values),
            "max_episode_steps": env._max_episode_steps,
            "num_agents": env.get_num_agents(),
        },
        "agents": agent_rows,
        "stops": stop_rows,
    }


def compact_stop_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key != "alternatives"
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    episodes = [
        run_episode(args, args.seed + index, index)
        for index in range(args.episodes)
    ]
    episode_rows = [episode["episode"] for episode in episodes]
    agent_rows = [row for episode in episodes for row in episode["agents"]]
    stop_rows = [row for episode in episodes for row in episode["stops"]]

    total_reward = sum(row["normalized_reward"] for row in episode_rows)
    total_success = sum(row["success_rate"] for row in episode_rows)
    print(
        "Summary: episodes={episodes} reward_mean={reward:.6f} "
        "success_rate_mean={success:.6f}".format(
            episodes=len(episode_rows),
            reward=total_reward / max(1, len(episode_rows)),
            success=total_success / max(1, len(episode_rows)),
        )
    )

    for episode in episode_rows:
        print(
            "seed={seed} reward={normalized_reward:.6f} "
            "success={success_rate:.6f}".format(**episode)
        )
        rows = [row for row in agent_rows if row["seed"] == episode["seed"]]
        for row in rows:
            nonzero = {
                penalty.value: row[penalty.value]
                for penalty in DefaultPenalties
                if abs(row[penalty.value]) > 1e-9
            }
            if abs(row["step_transition_penalty"]) > 1e-9:
                nonzero["STEP_TRANSITION_PENALTY"] = row[
                    "step_transition_penalty"
                ]
            print(
                "  agent={agent_id} state={state} success={success} "
                "cum={cumulative_reward:.1f} components={components} "
                "stop_events~={estimated_collision_stop_events}".format(
                    components=nonzero,
                    **row,
                )
            )

    result = {
        "episodes": episode_rows,
        "agents": agent_rows,
        "stops": stop_rows,
    }
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.output_agent_csv is not None:
        write_csv(args.output_agent_csv, agent_rows)
    if args.output_stop_csv is not None:
        write_csv(args.output_stop_csv, [compact_stop_row(row) for row in stop_rows])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
