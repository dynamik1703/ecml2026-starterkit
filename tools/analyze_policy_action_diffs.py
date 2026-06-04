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


def route_prefix_for_action(
    policy: Any,
    obs_builder: Any,
    handle: int,
    action: int,
    lookahead: int | None = None,
) -> list[dict[str, Any]]:
    if not hasattr(policy, "_route_prefix_for_action"):
        return []
    if lookahead is None:
        lookahead = int(getattr(policy, "FUTURE_RERANK_LOOKAHEAD_CELLS", 45))
    try:
        return policy._route_prefix_for_action(obs_builder, handle, action, lookahead)
    except Exception:
        return []


def planned_prefixes(policy: Any, obs_builder: Any, actions: dict[int, int]) -> dict[int, Any]:
    if not hasattr(policy, "_route_prefix_for_action"):
        return {}
    lookahead = int(getattr(policy, "FUTURE_RERANK_LOOKAHEAD_CELLS", 45))
    return {
        handle: route_prefix_for_action(policy, obs_builder, handle, action, lookahead)
        for handle, action in actions.items()
    }


def prefix_edges(
    prefix: list[dict[str, Any]],
) -> dict[tuple[tuple[int, int], tuple[int, int]], dict[str, Any]]:
    edges = {}
    for node in prefix:
        previous = node.get("prev_position")
        position = node.get("position")
        if previous is None or position is None or previous == position:
            continue
        edges[(previous, position)] = node
    return edges


def _agent_eta(env: Any, handle: int, step: int) -> float:
    try:
        speed = float(env.agents[handle].speed_counter.speed)
    except Exception:
        speed = 1.0
    if speed <= 0.0:
        speed = 1.0
    return float(step) / speed


def _agent_deadline_slack_at(env: Any, handle: int, step: int) -> float:
    try:
        latest_arrival = env.agents[handle].latest_arrival
    except Exception:
        latest_arrival = None
    if latest_arrival is None:
        return 999.0
    return float(latest_arrival) - float(env._elapsed_steps) - _agent_eta(env, handle, step)


def _direction_relation(own_direction: Any, other_direction: Any) -> str:
    if own_direction is None or other_direction is None:
        return "crossing"
    try:
        own = int(own_direction)
        other = int(other_direction)
    except Exception:
        return "crossing"
    if own == other:
        return "same"
    if (own + 2) % 4 == other:
        return "opposing"
    return "crossing"


def prefix_conflict_features(
    prefix_name: str,
    prefix: list[dict[str, Any]],
    prefixes: dict[int, list[dict[str, Any]]],
    handle: int,
    env: Any,
) -> dict[str, float]:
    own_positions: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for node in prefix:
        position = node.get("position")
        if position is None:
            continue
        own_positions.setdefault(position, []).append(node)

    own_edges = prefix_edges(prefix)
    conflict_agents: set[int] = set()
    cell_intersections = 0
    same_direction_intersections = 0
    opposing_direction_intersections = 0
    crossing_direction_intersections = 0
    same_edge_conflicts = 0
    head_on_edge_conflicts = 0
    first_intersection_step = 0
    min_intersection_eta_gap = float("inf")
    min_head_on_eta_gap = float("inf")
    min_own_deadline_slack = float("inf")
    min_other_deadline_slack = float("inf")
    min_pair_deadline_slack = float("inf")
    own_deadline_miss_conflicts = 0
    other_deadline_miss_conflicts = 0
    pair_deadline_miss_conflicts = 0
    own_tight_deadline_conflicts = 0
    other_tight_deadline_conflicts = 0
    pair_tight_deadline_conflicts = 0

    for other, other_prefix in prefixes.items():
        if other == handle or not other_prefix:
            continue
        other_positions: dict[tuple[int, int], list[dict[str, Any]]] = {}
        for other_node in other_prefix:
            position = other_node.get("position")
            if position is None:
                continue
            other_positions.setdefault(position, []).append(other_node)

        for position, own_nodes in own_positions.items():
            for other_node in other_positions.get(position, []):
                for own_node in own_nodes:
                    conflict_agents.add(other)
                    cell_intersections += 1
                    own_step = int(own_node.get("step", 0))
                    other_step = int(other_node.get("step", 0))
                    if first_intersection_step == 0 or own_step < first_intersection_step:
                        first_intersection_step = own_step
                    eta_gap = abs(
                        _agent_eta(env, handle, own_step)
                        - _agent_eta(env, other, other_step)
                    )
                    min_intersection_eta_gap = min(min_intersection_eta_gap, eta_gap)
                    own_slack = _agent_deadline_slack_at(env, handle, own_step)
                    other_slack = _agent_deadline_slack_at(env, other, other_step)
                    pair_slack = min(own_slack, other_slack)
                    min_own_deadline_slack = min(min_own_deadline_slack, own_slack)
                    min_other_deadline_slack = min(min_other_deadline_slack, other_slack)
                    min_pair_deadline_slack = min(min_pair_deadline_slack, pair_slack)
                    own_deadline_miss_conflicts += int(own_slack < 0.0)
                    other_deadline_miss_conflicts += int(other_slack < 0.0)
                    pair_deadline_miss_conflicts += int(pair_slack < 0.0)
                    own_tight_deadline_conflicts += int(own_slack < 20.0)
                    other_tight_deadline_conflicts += int(other_slack < 20.0)
                    pair_tight_deadline_conflicts += int(pair_slack < 20.0)
                    relation = _direction_relation(
                        own_node.get("direction"),
                        other_node.get("direction"),
                    )
                    if relation == "same":
                        same_direction_intersections += 1
                    elif relation == "opposing":
                        opposing_direction_intersections += 1
                    else:
                        crossing_direction_intersections += 1

        other_edges = prefix_edges(other_prefix)
        for source, target in own_edges:
            if (source, target) in other_edges:
                same_edge_conflicts += 1
                conflict_agents.add(other)
            head_on_node = other_edges.get((target, source))
            if head_on_node is None:
                continue
            head_on_edge_conflicts += 1
            conflict_agents.add(other)
            own_node = own_edges[(source, target)]
            eta_gap = abs(
                _agent_eta(env, handle, int(own_node.get("step", 0)))
                - _agent_eta(env, other, int(head_on_node.get("step", 0)))
            )
            min_head_on_eta_gap = min(min_head_on_eta_gap, eta_gap)
            own_slack = _agent_deadline_slack_at(
                env,
                handle,
                int(own_node.get("step", 0)),
            )
            other_slack = _agent_deadline_slack_at(
                env,
                other,
                int(head_on_node.get("step", 0)),
            )
            pair_slack = min(own_slack, other_slack)
            min_own_deadline_slack = min(min_own_deadline_slack, own_slack)
            min_other_deadline_slack = min(min_other_deadline_slack, other_slack)
            min_pair_deadline_slack = min(min_pair_deadline_slack, pair_slack)
            own_deadline_miss_conflicts += int(own_slack < 0.0)
            other_deadline_miss_conflicts += int(other_slack < 0.0)
            pair_deadline_miss_conflicts += int(pair_slack < 0.0)
            own_tight_deadline_conflicts += int(own_slack < 20.0)
            other_tight_deadline_conflicts += int(other_slack < 20.0)
            pair_tight_deadline_conflicts += int(pair_slack < 20.0)

    if not np.isfinite(min_intersection_eta_gap):
        min_intersection_eta_gap = 999.0
    if not np.isfinite(min_head_on_eta_gap):
        min_head_on_eta_gap = 999.0
    if not np.isfinite(min_own_deadline_slack):
        min_own_deadline_slack = 999.0
    if not np.isfinite(min_other_deadline_slack):
        min_other_deadline_slack = 999.0
    if not np.isfinite(min_pair_deadline_slack):
        min_pair_deadline_slack = 999.0

    return {
        f"{prefix_name}_prefix_len": float(len(prefix)),
        f"{prefix_name}_prefix_conflict_agents": float(len(conflict_agents)),
        f"{prefix_name}_prefix_cell_intersections": float(cell_intersections),
        f"{prefix_name}_prefix_first_intersection_step": float(first_intersection_step),
        f"{prefix_name}_prefix_min_intersection_eta_gap": float(min_intersection_eta_gap),
        f"{prefix_name}_prefix_same_direction_intersections": float(same_direction_intersections),
        f"{prefix_name}_prefix_opposing_direction_intersections": float(opposing_direction_intersections),
        f"{prefix_name}_prefix_crossing_direction_intersections": float(crossing_direction_intersections),
        f"{prefix_name}_prefix_same_edge_conflicts": float(same_edge_conflicts),
        f"{prefix_name}_prefix_head_on_edge_conflicts": float(head_on_edge_conflicts),
        f"{prefix_name}_prefix_min_head_on_eta_gap": float(min_head_on_eta_gap),
        f"{prefix_name}_prefix_min_own_deadline_slack": float(min_own_deadline_slack),
        f"{prefix_name}_prefix_min_other_deadline_slack": float(min_other_deadline_slack),
        f"{prefix_name}_prefix_min_pair_deadline_slack": float(min_pair_deadline_slack),
        f"{prefix_name}_prefix_own_deadline_miss_conflicts": float(own_deadline_miss_conflicts),
        f"{prefix_name}_prefix_other_deadline_miss_conflicts": float(other_deadline_miss_conflicts),
        f"{prefix_name}_prefix_pair_deadline_miss_conflicts": float(pair_deadline_miss_conflicts),
        f"{prefix_name}_prefix_own_tight_deadline_conflicts": float(own_tight_deadline_conflicts),
        f"{prefix_name}_prefix_other_tight_deadline_conflicts": float(other_tight_deadline_conflicts),
        f"{prefix_name}_prefix_pair_tight_deadline_conflicts": float(pair_tight_deadline_conflicts),
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


def observation_scalar_features(observation: Any) -> dict[str, float]:
    values = np.asarray(observation, dtype=np.float32)
    if values.shape[0] < 36:
        return {}
    return {
        "obs_ready_to_depart": float(values[4]),
        "obs_active": float(values[5]),
        "obs_done": float(values[6]),
        "obs_on_switch": float(values[7]),
        "obs_before_switch": float(values[8]),
        "obs_near_switch": float(values[9]),
        "obs_current_distance": float(values[30]),
        "obs_elapsed_fraction": float(values[31]),
        "obs_active_fraction": float(values[32]),
        "obs_stop_proximity": float(values[33]),
        "obs_waypoint_progress": float(values[34]),
        "obs_time_slack": float(values[35]),
    }


def direction_features(
    observation: Any,
    prefix: str,
    direction: int | None,
) -> dict[str, float]:
    values = np.asarray(observation, dtype=np.float32)
    if direction is None or direction < 0 or direction > 3 or values.shape[0] < 30:
        return {
            f"{prefix}_leads_closer": float("nan"),
            f"{prefix}_transition_exists": float("nan"),
            f"{prefix}_opposing_ahead": float("nan"),
            f"{prefix}_same_direction_ahead": float("nan"),
            f"{prefix}_switch_ahead": float("nan"),
            f"{prefix}_normalized_distance": float("nan"),
        }
    return {
        f"{prefix}_leads_closer": float(values[direction]),
        f"{prefix}_transition_exists": float(values[10 + direction]),
        f"{prefix}_opposing_ahead": float(values[14 + direction]),
        f"{prefix}_same_direction_ahead": float(values[18 + direction]),
        f"{prefix}_switch_ahead": float(values[22 + direction]),
        f"{prefix}_normalized_distance": float(values[26 + direction]),
    }


def raw_logits(policy: Any, observation: Any) -> list[float]:
    raw_policy = getattr(policy, "rl_policy", None)
    if raw_policy is None or not hasattr(raw_policy, "masked_logits"):
        return []
    try:
        logits = raw_policy.masked_logits(np.asarray([observation], dtype=np.float32))
        return [float(value) for value in logits[0].detach().cpu().tolist()]
    except Exception:
        return []


def logit_features(
    prefix: str,
    logits: list[float],
    baseline_action: int,
    candidate_action: int,
) -> dict[str, float]:
    if not logits:
        return {
            f"{prefix}_baseline_logit": float("nan"),
            f"{prefix}_candidate_logit": float("nan"),
            f"{prefix}_candidate_minus_baseline_logit": float("nan"),
            f"{prefix}_top_logit_margin": float("nan"),
        }
    finite_logits = [value for value in logits if np.isfinite(value)]
    top_margin = float("nan")
    if len(finite_logits) >= 2:
        top_two = sorted(finite_logits, reverse=True)[:2]
        top_margin = top_two[0] - top_two[1]
    baseline_logit = logits[baseline_action] if baseline_action < len(logits) else float("nan")
    candidate_logit = logits[candidate_action] if candidate_action < len(logits) else float("nan")
    return {
        f"{prefix}_baseline_logit": baseline_logit,
        f"{prefix}_candidate_logit": candidate_logit,
        f"{prefix}_candidate_minus_baseline_logit": candidate_logit - baseline_logit,
        f"{prefix}_top_logit_margin": top_margin,
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
    baseline_logits = raw_logits(baseline_policy, observation)
    candidate_logits = raw_logits(candidate_policy, observation)
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
        **observation_scalar_features(observation),
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
        **direction_features(
            observation,
            "baseline_direction",
            baseline_target["target_direction"],
        ),
        "candidate_target": candidate_target["target"],
        "candidate_target_direction": candidate_target["target_direction"],
        "candidate_target_distance": candidate_target["target_distance"],
        "candidate_target_occupied_by_other": candidate_target["target_occupied_by_other"],
        **direction_features(
            observation,
            "candidate_direction",
            candidate_target["target_direction"],
        ),
        "candidate_distance_delta": (
            candidate_target["target_distance"] - baseline_target["target_distance"]
        ),
        **logit_features(
            "baseline_raw",
            baseline_logits,
            baseline_action,
            candidate_action,
        ),
        **logit_features(
            "candidate_raw",
            candidate_logits,
            baseline_action,
            candidate_action,
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
