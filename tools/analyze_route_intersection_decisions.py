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
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.persistence import RailEnvPersister

from tools.analyze_long_corridor_decisions import (
    ACTION_NAMES,
    MOVE_ACTIONS,
    compact_float,
    distance_and_slack,
    is_success,
    mask_from_observation,
    mask_value,
    optional_int,
    position_to_text,
)
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


def speed(agent: Any) -> float:
    value = float(agent.speed_counter.speed)
    return value if value > 0 else 1.0


def agent_direction(agent: Any) -> int | None:
    return agent.direction if agent.direction is not None else agent.initial_direction


def route_start(agent: Any) -> tuple[tuple[int, int] | None, int | None]:
    if agent.position is None:
        return agent.initial_position, agent.initial_direction
    return agent.position, agent_direction(agent)


def is_moving_state(agent: Any) -> bool:
    return state_name(agent.state) == "MOVING"


def is_wait_decision(agent: Any, action: int) -> bool:
    if action == 4:
        return True
    return action == 0 and not is_moving_state(agent)


def valid_candidate_actions(
    agent: Any,
    chosen_action: int,
    local_mask: np.ndarray,
) -> list[int]:
    if chosen_action in MOVE_ACTIONS or (
        chosen_action == 0 and is_moving_state(agent)
    ):
        return [chosen_action]
    return [action for action in MOVE_ACTIONS if mask_value(local_mask, action) >= 0.5]


def route_prefix_for_action(
    obs_builder: Any,
    handle: int,
    action: int,
    lookahead: int,
) -> list[dict[str, Any]]:
    env = obs_builder.env
    agent = env.agents[handle]
    if is_success(agent):
        return []

    position, direction = route_start(agent)
    if position is None or direction is None:
        return []

    prefix: list[dict[str, Any]] = []
    distance_map = obs_builder._get_distance_map(handle)

    if action == 0 and is_moving_state(agent):
        transitions = env.rail.get_transitions((position, direction))
        target_direction = obs_builder._best_progress_direction(
            transitions,
            position,
            distance_map,
        )
        if target_direction is None:
            return []
        target_position = get_new_position(position, target_direction)
        if not obs_builder._is_in_bounds(target_position):
            return []
        prefix.append(
            {
                "step": 1,
                "prev_position": position,
                "position": target_position,
                "direction": target_direction,
                "action": action,
            }
        )
        current_position = target_position
        current_direction = target_direction
        start_step = 2
    elif action in MOVE_ACTIONS:
        target_position, target_direction = obs_builder._action_target(handle, action)
        if target_position is None or target_direction is None:
            return []
        if not obs_builder._is_in_bounds(target_position):
            return []
        prefix.append(
            {
                "step": 1,
                "prev_position": agent.position,
                "position": target_position,
                "direction": target_direction,
                "action": action,
            }
        )
        current_position = target_position
        current_direction = target_direction
        start_step = 2
    elif is_wait_decision(agent, action) and agent.position is not None:
        current_position = agent.position
        current_direction = direction
        prefix.append(
            {
                "step": 1,
                "prev_position": None,
                "position": current_position,
                "direction": current_direction,
                "action": action,
            }
        )
        start_step = 2
    else:
        return []

    seen = {
        (
            prefix[-1]["position"],
            prefix[-1]["direction"],
        )
    }
    for step_index in range(start_step, lookahead + 1):
        transitions = env.rail.get_transitions((current_position, current_direction))
        next_direction = obs_builder._best_progress_direction(
            transitions,
            current_position,
            distance_map,
        )
        if next_direction is None:
            break
        next_position = get_new_position(current_position, next_direction)
        if not obs_builder._is_in_bounds(next_position):
            break
        state = (next_position, next_direction)
        if state in seen:
            break
        seen.add(state)
        prefix.append(
            {
                "step": step_index,
                "prev_position": current_position,
                "position": next_position,
                "direction": next_direction,
                "action": action,
            }
        )
        current_position = next_position
        current_direction = next_direction

    return prefix


def index_prefix(prefix: list[dict[str, Any]]) -> tuple[dict[Any, Any], dict[Any, Any]]:
    positions: dict[tuple[int, int], dict[str, Any]] = {}
    edges: dict[tuple[tuple[int, int], tuple[int, int]], dict[str, Any]] = {}
    for node in prefix:
        positions.setdefault(node["position"], node)
        previous = node["prev_position"]
        if previous is not None and previous != node["position"]:
            edges.setdefault((previous, node["position"]), node)
    return positions, edges


def same_position_kind(own_node: dict[str, Any], other_node: dict[str, Any]) -> str:
    own_direction = own_node["direction"]
    other_direction = other_node["direction"]
    if own_direction == other_direction:
        return "same_direction"
    if (
        own_direction is not None
        and other_direction is not None
        and (other_direction - own_direction) % 4 == 2
    ):
        return "opposing_cell"
    return "crossing"


def kind_matches(kind: str, mode: str) -> bool:
    if mode == "any":
        return True
    if mode == "non_same":
        return kind != "same_direction"
    return kind == mode


def best_intersection(
    args: argparse.Namespace,
    obs_builder: Any,
    handle: int,
    candidate_action: int,
    planned_prefixes: dict[int, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    own_prefix = route_prefix_for_action(
        obs_builder,
        handle,
        candidate_action,
        args.lookahead,
    )
    if not own_prefix:
        return None

    own_positions, own_edges = index_prefix(own_prefix)
    env = obs_builder.env
    best = None
    for other, other_prefix in planned_prefixes.items():
        if other == handle or not other_prefix:
            continue
        other_agent = env.agents[other]
        own_agent = env.agents[handle]
        own_speed = speed(own_agent)
        other_speed = speed(other_agent)

        for other_node in other_prefix:
            other_prev = other_node["prev_position"]
            if other_prev is not None:
                own_node = own_edges.get((other_node["position"], other_prev))
                if own_node is not None:
                    candidate = intersection_candidate(
                        args,
                        kind="head_on",
                        handle=handle,
                        other=other,
                        own_node=own_node,
                        other_node=other_node,
                        own_speed=own_speed,
                        other_speed=other_speed,
                    )
                    best = better_candidate(best, candidate)

            own_node = own_positions.get(other_node["position"])
            if own_node is None:
                continue
            kind = same_position_kind(own_node, other_node)
            candidate = intersection_candidate(
                args,
                kind=kind,
                handle=handle,
                other=other,
                own_node=own_node,
                other_node=other_node,
                own_speed=own_speed,
                other_speed=other_speed,
            )
            best = better_candidate(best, candidate)

    return best


def intersection_candidate(
    args: argparse.Namespace,
    kind: str,
    handle: int,
    other: int,
    own_node: dict[str, Any],
    other_node: dict[str, Any],
    own_speed: float,
    other_speed: float,
) -> dict[str, Any] | None:
    own_step = int(own_node["step"])
    other_step = int(other_node["step"])
    if own_step < args.min_own_step or other_step < args.min_other_step:
        return None
    own_eta = own_step / own_speed
    other_eta = other_step / other_speed
    eta_gap = abs(own_eta - other_eta)
    if eta_gap > args.max_eta_gap:
        return None
    if not kind_matches(kind, args.conflict_kind):
        return None
    return {
        "kind": kind,
        "agent_id": int(handle),
        "other_agent": int(other),
        "conflict_position": own_node["position"],
        "own_step": own_step,
        "other_step": other_step,
        "own_eta": float(own_eta),
        "other_eta": float(other_eta),
        "eta_gap": float(eta_gap),
        "eta_overlap": float(max(0.0, 1.0 - eta_gap / max(1.0, args.lookahead))),
        "other_reaches_first": bool(other_eta < own_eta),
        "own_direction": optional_int(own_node["direction"]),
        "other_direction": optional_int(other_node["direction"]),
        "sort_key": (
            0 if kind == "head_on" else 1,
            eta_gap,
            own_step,
            other_step,
            other,
        ),
    }


def better_candidate(
    current: dict[str, Any] | None,
    candidate: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if candidate is None:
        return current
    if current is None or candidate["sort_key"] < current["sort_key"]:
        return candidate
    return current


def route_feature_values(obs_builder: Any, observation: Any) -> dict[str, float]:
    if not getattr(obs_builder, "with_route_conflict_features", False):
        return {}
    values = np.asarray(observation, dtype=np.float32)
    if values.shape[0] < 52:
        return {}
    return {
        "obs_future_own_distance": float(values[44]),
        "obs_future_other_distance": float(values[45]),
        "obs_future_eta_overlap": float(values[46]),
        "obs_future_other_first": float(values[47]),
        "obs_future_head_on": float(values[48]),
        "obs_future_crossing": float(values[49]),
        "obs_future_other_has_priority": float(values[50]),
        "obs_future_intersection_count": float(values[51]),
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
    planned_prefixes = {
        handle: route_prefix_for_action(
            obs_builder,
            handle,
            int(actions.get(handle, 0)),
            args.lookahead,
        )
        for handle in env.get_agent_handles()
    }

    events = []
    for handle in env.get_agent_handles():
        agent = env.agents[handle]
        if is_success(agent):
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
        for candidate_action in valid_candidate_actions(agent, chosen_action, local_mask):
            intersection = best_intersection(
                args,
                obs_builder,
                handle,
                candidate_action,
                planned_prefixes,
            )
            if intersection is None:
                continue
            other = intersection["other_agent"]
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
                "candidate_action": int(candidate_action),
                "candidate_action_name": ACTION_NAMES.get(
                    candidate_action,
                    str(candidate_action),
                ),
                "candidate_was_chosen": bool(candidate_action == chosen_action),
                "chosen_was_wait": bool(is_wait_decision(agent, chosen_action)),
                "local_allowed": mask_value(local_mask, candidate_action),
                "coordination_allowed": mask_value(coordination_mask, candidate_action),
                "observation_allowed": mask_value(observation_mask, candidate_action),
                "kind": intersection["kind"],
                "conflict_position": position_to_text(intersection["conflict_position"]),
                "own_step": intersection["own_step"],
                "other_step": intersection["other_step"],
                "own_eta": intersection["own_eta"],
                "other_eta": intersection["other_eta"],
                "eta_gap": intersection["eta_gap"],
                "eta_overlap": intersection["eta_overlap"],
                "other_reaches_first": intersection["other_reaches_first"],
                "own_direction": intersection["own_direction"],
                "other_agent": other,
                "other_state": state_name(env.agents[other].state),
                "other_position": position_to_text(env.agents[other].position),
                "other_direction": intersection["other_direction"],
                "other_distance": other_distance,
                "other_slack": other_slack,
                "other_has_priority": bool(other_slack < slack),
            }
            event.update(route_feature_values(obs_builder, observation))
            events.append(event)
    return events


def missed_by(env: Any, agent: Any) -> int | None:
    latest_arrival = getattr(agent, "latest_arrival", None)
    if latest_arrival is None or is_success(agent):
        return None
    return int(max(0, env._elapsed_steps - latest_arrival))


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
        "chosen_event_count": sum(int(event["candidate_was_chosen"]) for event in events),
        "wait_event_count": sum(int(event["chosen_was_wait"]) for event in events),
    }
    return row, events


def print_summary(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    reward_mean = sum(row["normalized_reward"] for row in rows) / len(rows) if rows else 0.0
    success_mean = sum(row["success_rate"] for row in rows) / len(rows) if rows else 0.0
    kind_counts = Counter(event["kind"] for event in events)
    action_counts = Counter(event["chosen_action_name"] for event in events)
    failed_events = [event for event in events if event["agent_failed"]]
    chosen_events = [event for event in events if event["candidate_was_chosen"]]
    wait_events = [event for event in events if event["chosen_was_wait"]]
    print(
        "Totals: "
        f"episodes={len(rows)}, "
        f"reward_mean={reward_mean:.6g}, "
        f"success_rate_mean={success_mean:.6g}, "
        f"events={len(events)}, "
        f"failed_agent_events={len(failed_events)}, "
        f"chosen_conflict_events={len(chosen_events)}, "
        f"wait_conflict_events={len(wait_events)}"
    )
    print(f"Kinds: {dict(sorted(kind_counts.items()))}")
    print(f"Chosen actions: {dict(sorted(action_counts.items()))}")

    if not events:
        return

    print(
        "\nseed,t,agent,chosen,candidate,kind,own_step,other,other_step,"
        "eta_gap,agent_slack,other_slack,coord_allowed,agent_success,other_success,reward"
    )
    ranked = sorted(
        events,
        key=lambda event: (
            not event["both_failed"],
            not event["agent_failed"],
            event["candidate_was_chosen"] is False,
            event["eta_gap"],
            event["own_step"],
            event["seed"],
            event["env_time"],
        ),
    )
    for event in ranked[: args.top_k]:
        print(
            f"{event['seed']},{event['env_time']},{event['agent_id']},"
            f"{event['chosen_action_name']},{event['candidate_action_name']},"
            f"{event['kind']},{event['own_step']},{event['other_agent']},"
            f"{event['other_step']},{compact_float(event['eta_gap'])},"
            f"{compact_float(event['agent_slack'])},"
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
            "Trace policy decisions whose candidate route prefix intersects "
            "another train's planned route prefix with a small ETA gap."
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
    parser.add_argument("--min-own-step", type=int, default=2)
    parser.add_argument("--min-other-step", type=int, default=1)
    parser.add_argument("--max-eta-gap", type=float, default=8.0)
    parser.add_argument(
        "--conflict-kind",
        choices=[
            "head_on",
            "opposing_cell",
            "crossing",
            "same_direction",
            "non_same",
            "any",
        ],
        default="non_same",
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
