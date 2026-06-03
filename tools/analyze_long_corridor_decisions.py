#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from flatland.envs.persistence import RailEnvPersister

from tools.evaluate_sampled import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_POLICY,
    DEFAULT_REWARDS,
    action_id,
    instantiate_policy,
    load_sampling_env_generator,
    load_symbol,
    normalized_reward,
    observation_list,
    repo_root,
    state_name,
)


ACTION_NAMES = {
    0: "DO_NOTHING",
    1: "MOVE_LEFT",
    2: "MOVE_FORWARD",
    3: "MOVE_RIGHT",
    4: "STOP_MOVING",
}
MOVE_ACTIONS = (1, 2, 3)
WAIT_ACTIONS = (0, 4)


def direction_between(
    source: tuple[int, int],
    target: tuple[int, int],
) -> int | None:
    dr = target[0] - source[0]
    dc = target[1] - source[1]
    if dr == -1 and dc == 0:
        return 0
    if dr == 0 and dc == 1:
        return 1
    if dr == 1 and dc == 0:
        return 2
    if dr == 0 and dc == -1:
        return 3
    return None


def agent_direction(agent: Any) -> int | None:
    return agent.direction if agent.direction is not None else agent.initial_direction


def relation_for(path_direction: int | None, other_direction: int | None) -> str:
    if path_direction is None or other_direction is None:
        return "unknown"
    if path_direction == other_direction:
        return "same"
    if (other_direction - path_direction) % 4 == 2:
        return "opposing"
    return "crossing"


def optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def finite_float(value: Any, default: float = 1e9) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def mask_value(mask: np.ndarray | None, action: int) -> float:
    if mask is None or action >= len(mask):
        return 0.0
    return float(mask[action])


def mask_from_observation(observation: Any) -> np.ndarray:
    if observation is None:
        return np.zeros(5, dtype=np.float32)
    values = np.asarray(observation, dtype=np.float32)
    if values.shape[0] < 5:
        return np.zeros(5, dtype=np.float32)
    return values[-5:]


def distance_and_slack(obs_builder: Any, handle: int) -> tuple[float, float]:
    try:
        _, slack, distance, _ = obs_builder._priority_key(handle)
        return finite_float(distance), finite_float(slack)
    except Exception:
        return 1e9, 1e9


def is_success(agent: Any) -> bool:
    return bool(agent.state == 6 or state_name(agent.state) in {"DONE", "DONE_REMOVED"})


def position_to_text(position: Any) -> str:
    return "" if position is None else str(tuple(position))


def first_corridor_conflict(
    obs_builder: Any,
    handle: int,
    action: int,
    lookahead: int,
) -> dict[str, Any] | None:
    try:
        edges = obs_builder._corridor_edges_for_action(
            handle,
            action,
            max_cells=lookahead,
        )
    except Exception:
        return None

    for distance, (source, target) in enumerate(edges, start=1):
        other = obs_builder._agent_at(target)
        if other == -1 or other == handle:
            continue
        path_direction = direction_between(source, target)
        other_agent = obs_builder.env.agents[other]
        other_direction = agent_direction(other_agent)
        return {
            "conflict_distance": int(distance),
            "conflict_source": source,
            "conflict_position": target,
            "path_direction": optional_int(path_direction),
            "other_agent": int(other),
            "other_position": other_agent.position,
            "other_direction": optional_int(other_direction),
            "relation": relation_for(path_direction, other_direction),
            "corridor_length_scanned": len(edges),
        }
    return None


def relation_matches(relation: str, mode: str) -> bool:
    if mode == "any":
        return True
    if mode == "non_same":
        return relation != "same"
    return relation == mode


def candidate_actions(
    chosen_action: int,
    local_mask: np.ndarray,
) -> list[int]:
    if chosen_action in MOVE_ACTIONS:
        return [chosen_action]
    return [action for action in MOVE_ACTIONS if mask_value(local_mask, action) >= 0.5]


def route_feature_values(obs_builder: Any, observation: Any) -> dict[str, float]:
    if not getattr(obs_builder, "with_route_conflict_features", False):
        return {}
    values = np.asarray(observation, dtype=np.float32)
    if values.shape[0] < 52:
        return {}
    return {
        "obs_route_train_distance": float(values[36]),
        "obs_route_train_opposing": float(values[37]),
        "obs_route_train_same": float(values[38]),
        "obs_route_other_has_priority": float(values[40]),
        "obs_route_side_detour": float(values[41]),
        "obs_future_head_on": float(values[48]),
        "obs_future_crossing": float(values[49]),
    }


def collect_events_before_step(
    args: argparse.Namespace,
    obs_builder: Any,
    observations_by_handle: dict[int, Any],
    actions: dict[int, int],
    episode_index: int,
    seed: int,
) -> list[dict[str, Any]]:
    env = obs_builder.env
    events = []
    for handle in env.get_agent_handles():
        agent = env.agents[handle]
        if is_success(agent) or agent.position is None:
            continue

        chosen_action = int(actions.get(handle, 0))
        observation = observations_by_handle.get(handle)
        observation_mask = mask_from_observation(observation)
        try:
            local_mask = obs_builder._build_local_action_mask(handle)
        except Exception:
            local_mask = observation_mask
        coordination_mask = getattr(obs_builder, "_coordination_masks", {}).get(
            handle,
            local_mask,
        )

        distance, slack = distance_and_slack(obs_builder, handle)
        for candidate in candidate_actions(chosen_action, local_mask):
            conflict = first_corridor_conflict(
                obs_builder,
                handle,
                candidate,
                args.lookahead,
            )
            if conflict is None:
                continue
            if conflict["conflict_distance"] < args.min_conflict_distance:
                continue
            if conflict["conflict_distance"] > args.max_conflict_distance:
                continue
            if not relation_matches(conflict["relation"], args.relation):
                continue

            other = conflict["other_agent"]
            other_distance, other_slack = distance_and_slack(obs_builder, other)
            event = {
                "episode_index": episode_index,
                "seed": seed,
                "env_time": int(env._elapsed_steps),
                "agent_id": int(handle),
                "agent_state": state_name(agent.state),
                "agent_position": position_to_text(agent.position),
                "agent_direction": optional_int(agent_direction(agent)),
                "agent_distance": distance,
                "agent_slack": slack,
                "chosen_action": chosen_action,
                "chosen_action_name": ACTION_NAMES.get(chosen_action, str(chosen_action)),
                "candidate_action": int(candidate),
                "candidate_action_name": ACTION_NAMES.get(candidate, str(candidate)),
                "candidate_was_chosen": bool(candidate == chosen_action),
                "chosen_was_wait": bool(chosen_action in WAIT_ACTIONS),
                "local_allowed": mask_value(local_mask, candidate),
                "coordination_allowed": mask_value(coordination_mask, candidate),
                "observation_allowed": mask_value(observation_mask, candidate),
                "conflict_distance": conflict["conflict_distance"],
                "conflict_position": position_to_text(conflict["conflict_position"]),
                "conflict_source": position_to_text(conflict["conflict_source"]),
                "path_direction": optional_int(conflict["path_direction"]),
                "relation": conflict["relation"],
                "corridor_length_scanned": conflict["corridor_length_scanned"],
                "other_agent": other,
                "other_position": position_to_text(conflict["other_position"]),
                "other_direction": optional_int(conflict["other_direction"]),
                "other_state": state_name(env.agents[other].state),
                "other_distance": other_distance,
                "other_slack": other_slack,
                "other_has_priority": bool(other_slack < slack),
            }
            event.update(route_feature_values(obs_builder, observation))
            events.append(event)
    return events


def add_episode_outcomes(
    env: Any,
    events: list[dict[str, Any]],
    normalized_episode_reward: float,
) -> None:
    success_rate = sum(int(is_success(agent)) for agent in env.agents) / env.get_num_agents()
    for event in events:
        agent = env.agents[event["agent_id"]]
        other = env.agents[event["other_agent"]]
        event["episode_success_rate"] = float(success_rate)
        event["episode_normalized_reward"] = float(normalized_episode_reward)
        event["agent_success"] = is_success(agent)
        event["other_success"] = is_success(other)
        event["agent_failed"] = not event["agent_success"]
        event["other_failed"] = not event["other_success"]
        event["both_failed"] = bool(event["agent_failed"] and event["other_failed"])
        event["agent_final_state"] = state_name(agent.state)
        event["other_final_state"] = state_name(other.state)
        event["agent_final_position"] = position_to_text(agent.position)
        event["other_final_position"] = position_to_text(other.position)
        event["agent_missed_by"] = missed_by(env, agent)
        event["other_missed_by"] = missed_by(env, other)


def missed_by(env: Any, agent: Any) -> int | None:
    latest_arrival = getattr(agent, "latest_arrival", None)
    if latest_arrival is None or is_success(agent):
        return None
    return int(max(0, env._elapsed_steps - latest_arrival))


def run_episode(
    args: argparse.Namespace,
    episode_index: int,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    obs_builder = load_symbol(args.obs_builder)()
    rewards = load_symbol(args.rewards)()
    policy = instantiate_policy(args)
    env, _ = RailEnvPersister.load_new(
        args.base_state_pkl,
        obs_builder=obs_builder,
        rewards=rewards,
    )
    if args.num_agents is not None:
        env.number_of_agents = args.num_agents
    env = load_sampling_env_generator()(env, line_length=args.line_length, scene=args.scene)
    observations, _ = env.reset(random_seed=seed)

    reward_values: list[float] = []
    events: list[dict[str, Any]] = []
    done = False
    while not done and env._elapsed_steps < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        observations_by_handle = dict(zip(handles, observation_list(observations, handles)))
        actions = {
            handle: action_id(action)
            for handle, action in policy.act_many(
                handles,
                observation_list(observations, handles),
            ).items()
        }
        events.extend(
            collect_events_before_step(
                args,
                obs_builder,
                observations_by_handle,
                actions,
                episode_index,
                seed,
            )
        )
        observations, rewards_by_agent, dones, _ = env.step(actions)
        reward_values.extend(float(rewards_by_agent.get(handle, 0.0)) for handle in handles)
        done = bool(dones.get("__all__", False))

    episode_reward = normalized_reward(env, reward_values)
    add_episode_outcomes(env, events, episode_reward)
    success_rate = sum(int(is_success(agent)) for agent in env.agents) / env.get_num_agents()
    row = {
        "episode_index": episode_index,
        "seed": seed,
        "env_time": int(env._elapsed_steps),
        "success_rate": float(success_rate),
        "normalized_reward": float(episode_reward),
        "num_agents": int(env.get_num_agents()),
        "event_count": len(events),
        "failed_agent_event_count": sum(int(event["agent_failed"]) for event in events),
        "move_into_conflict_count": sum(
            int(event["candidate_was_chosen"]) for event in events
        ),
        "wait_at_conflict_count": sum(int(event["chosen_was_wait"]) for event in events),
    }
    return row, events


def compact_float(value: Any) -> str:
    if isinstance(value, bool):
        return str(value)
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "inf"
    return f"{number:.4g}"


def print_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    reward_mean = sum(row["normalized_reward"] for row in rows) / len(rows) if rows else 0.0
    success_mean = sum(row["success_rate"] for row in rows) / len(rows) if rows else 0.0
    action_counts = Counter(event["chosen_action_name"] for event in events)
    relation_counts = Counter(event["relation"] for event in events)
    failed_events = [event for event in events if event["agent_failed"]]
    move_events = [event for event in events if event["candidate_was_chosen"]]
    wait_events = [event for event in events if event["chosen_was_wait"]]

    print(
        "Totals: "
        f"episodes={len(rows)}, "
        f"reward_mean={reward_mean:.6g}, "
        f"success_rate_mean={success_mean:.6g}, "
        f"events={len(events)}, "
        f"failed_agent_events={len(failed_events)}, "
        f"move_into_conflict_events={len(move_events)}, "
        f"wait_at_conflict_events={len(wait_events)}"
    )
    print(f"Chosen actions: {dict(sorted(action_counts.items()))}")
    print(f"Relations: {dict(sorted(relation_counts.items()))}")

    if not events:
        return

    print(
        "\nseed,t,agent,chosen,candidate,conflict_dist,other,relation,"
        "agent_slack,other_slack,coord_allowed,agent_success,other_success,reward"
    )
    ranked = sorted(
        events,
        key=lambda event: (
            not event["both_failed"],
            not event["agent_failed"],
            event["candidate_was_chosen"] is False,
            event["conflict_distance"],
            event["seed"],
            event["env_time"],
        ),
    )
    for event in ranked[: args.top_k]:
        print(
            f"{event['seed']},{event['env_time']},{event['agent_id']},"
            f"{event['chosen_action_name']},{event['candidate_action_name']},"
            f"{event['conflict_distance']},{event['other_agent']},"
            f"{event['relation']},{compact_float(event['agent_slack'])},"
            f"{compact_float(event['other_slack'])},"
            f"{compact_float(event['coordination_allowed'])},"
            f"{event['agent_success']},{event['other_success']},"
            f"{compact_float(event['episode_normalized_reward'])}"
        )


def write_outputs(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    summary = {
        "policy": args.policy,
        "policy_checkpoint": (
            str(args.policy_checkpoint) if args.policy_checkpoint is not None else None
        ),
        "obs_builder": args.obs_builder,
        "episodes": len(rows),
        "rows": rows,
        "events": events,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for event in events for key in event.keys()})
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(events)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Trace policy decisions where a move candidate leads into a long "
            "corridor occupied by another train."
        )
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument(
        "--policy-checkpoint",
        type=Path,
        help="Instantiate policies that accept checkpoint_path with this checkpoint.",
    )
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--lookahead", type=int, default=45)
    parser.add_argument("--min-conflict-distance", type=int, default=2)
    parser.add_argument("--max-conflict-distance", type=int, default=45)
    parser.add_argument(
        "--relation",
        choices=["opposing", "crossing", "same", "non_same", "any"],
        default="opposing",
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTHONPATH", str(repo_root()))

    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    for episode_index in range(args.episodes):
        seed = args.seed + episode_index
        row, episode_events = run_episode(args, episode_index, seed)
        rows.append(row)
        events.extend(episode_events)

    print("seed,env_time,success_rate,normalized_reward,event_count,failed_agent_events")
    for row in rows:
        print(
            f"{row['seed']},{row['env_time']},{row['success_rate']:.6g},"
            f"{row['normalized_reward']:.6g},{row['event_count']},"
            f"{row['failed_agent_event_count']}"
        )
    print_summary(args, rows, events)
    write_outputs(args, rows, events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
