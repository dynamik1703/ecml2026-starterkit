#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env_action import RailEnvActions


DEFAULT_POLICY = "submission.hybrid_policy.MyPolicy"
DEFAULT_OBS_BUILDER = "submission.my_observation_builder.MyObservationBuilder"
DEFAULT_REWARDS = "flatland.envs.rewards.ECML2026Rewards"
DEFAULT_BASE_STATE = "reinforcement-learning/sampling/level_0_scenario_1.pkl"

MOVE_ACTIONS = (1, 2, 3)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_symbol(path: str) -> Any:
    module_name, symbol_name = path.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def load_sampling_env_generator() -> Any:
    path = repo_root() / "reinforcement-learning" / "sampling" / "sampling_env_generator.py"
    spec = importlib.util.spec_from_file_location("ecml_sampling_env_generator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sampling generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sampling_env_generator


def instantiate_policy(args: argparse.Namespace) -> Any:
    policy_cls = load_symbol(args.policy)
    if args.policy_checkpoint is None:
        return policy_cls()
    try:
        return policy_cls(checkpoint_path=str(args.policy_checkpoint))
    except TypeError as exc:
        raise TypeError(
            f"{args.policy} does not accept --policy-checkpoint."
        ) from exc


def action_id(action: Any) -> int:
    if hasattr(action, "value"):
        return int(action.value)
    return int(action)


def action_name(action: int) -> str:
    try:
        return RailEnvActions(action).name
    except ValueError:
        return str(action)


def observation_list(observations: dict[int, Any], handles: list[int]) -> list[Any]:
    return [observations[handle] for handle in handles]


def normalized_reward(env: Any, rewards: list[float]) -> float:
    if not rewards:
        return 0.0
    return float(
        env.rewards.normalize(
            *rewards,
            max_episode_steps=env._max_episode_steps,
            num_agents=env.get_num_agents(),
        )
    )


def state_name(state: Any) -> str:
    return getattr(state, "name", str(state))


def is_done_state(state: Any) -> bool:
    name = state_name(state)
    return name in {"DONE", "DONE_REMOVED"} or state == 6


def direction_between(source: tuple[int, int], target: tuple[int, int]) -> int | None:
    row_delta = target[0] - source[0]
    col_delta = target[1] - source[1]
    if row_delta == -1 and col_delta == 0:
        return 0
    if row_delta == 0 and col_delta == 1:
        return 1
    if row_delta == 1 and col_delta == 0:
        return 2
    if row_delta == 0 and col_delta == -1:
        return 3
    return None


def agent_direction(agent: Any) -> int | None:
    return agent.direction if agent.direction is not None else agent.initial_direction


def relation_to_path(path_direction: int | None, other_direction: int | None) -> str:
    if path_direction is None or other_direction is None:
        return "unknown"
    if (other_direction - path_direction) % 4 == 2:
        return "opposing"
    if other_direction == path_direction:
        return "same"
    return "crossing"


def position_key(position: Any) -> str | None:
    return None if position is None else str(tuple(position))


def make_env(args: argparse.Namespace, obs_builder: Any, rewards: Any) -> Any:
    env, _ = RailEnvPersister.load_new(
        args.base_state_pkl,
        obs_builder=obs_builder,
        rewards=rewards,
    )
    if args.num_agents is not None:
        env.number_of_agents = args.num_agents
    env = load_sampling_env_generator()(env, line_length=args.line_length, scene=args.scene)
    env.reset(random_seed=args.seed)
    return env


def valid_action_targets(obs_builder: Any, handle: int) -> dict[int, dict[str, Any]]:
    env = obs_builder.env
    agent = env.agents[handle]
    position = agent.position
    direction = agent_direction(agent)
    if position is None or direction is None:
        return {}

    possible_transitions = env.rail.get_transitions((position, direction))
    targets = {}
    for action in MOVE_ACTIONS:
        new_direction = obs_builder._action_to_direction(action, direction)
        if new_direction is None or not possible_transitions[new_direction]:
            continue
        target = get_new_position(position, new_direction)
        other = obs_builder._agent_at(target)
        targets[action] = {
            "action": action_name(action),
            "target": position_key(target),
            "target_direction": new_direction,
            "target_occupant": None if other == -1 else int(other),
        }
    return targets


def first_corridor_blocker(
    obs_builder: Any,
    handle: int,
    action: int,
) -> dict[str, Any] | None:
    for step, (source, target) in enumerate(
        obs_builder._corridor_edges_for_action(handle, action),
        start=1,
    ):
        other = obs_builder._agent_at(target)
        if other == -1 or other == handle:
            continue
        other_agent = obs_builder.env.agents[other]
        path_direction = direction_between(source, target)
        other_direction = agent_direction(other_agent)
        return {
            "step": step,
            "agent": int(other),
            "position": position_key(target),
            "path_direction": path_direction,
            "other_direction": other_direction,
            "relation": relation_to_path(path_direction, other_direction),
        }
    return None


def connected_components(edges: dict[int, set[int]]) -> list[list[int]]:
    remaining = set(edges)
    for neighbours in edges.values():
        remaining.update(neighbours)

    components = []
    while remaining:
        root = remaining.pop()
        stack = [root]
        component = {root}
        while stack:
            node = stack.pop()
            for neighbour in edges.get(node, set()):
                if neighbour in component:
                    continue
                component.add(neighbour)
                remaining.discard(neighbour)
                stack.append(neighbour)
            for candidate, neighbours in edges.items():
                if node not in neighbours or candidate in component:
                    continue
                component.add(candidate)
                remaining.discard(candidate)
                stack.append(candidate)
        components.append(sorted(component))
    return sorted(components, key=lambda item: (len(item), item), reverse=True)


def analyze_episode(args: argparse.Namespace) -> dict[str, Any]:
    obs_builder = load_symbol(args.obs_builder)()
    rewards = load_symbol(args.rewards)()
    policy = instantiate_policy(args)
    env = make_env(args, obs_builder, rewards)

    observations, _ = env.reset(random_seed=args.seed)
    handles = list(env.get_agent_handles())
    reward_values: list[float] = []
    positions: dict[int, list[Any]] = defaultdict(list)
    actions_by_agent: dict[int, list[int]] = defaultdict(list)

    max_steps = args.max_steps or env._max_episode_steps
    while int(env._elapsed_steps) < max_steps:
        actions = {
            handle: action_id(action)
            for handle, action in policy.act_many(
                handles,
                observation_list(observations, handles),
            ).items()
        }
        for handle, action in actions.items():
            actions_by_agent[handle].append(action)

        observations, rewards_by_agent, dones, _ = env.step(actions)
        for handle in handles:
            positions[handle].append(env.agents[handle].position)
        reward_values.extend(float(rewards_by_agent.get(handle, 0.0)) for handle in handles)
        if bool(dones.get("__all__", False)):
            break

    observations = env._get_observations()
    failed = [handle for handle, agent in enumerate(env.agents) if not is_done_state(agent.state)]
    failed_set = set(failed)
    blocked_edges: dict[int, set[int]] = defaultdict(set)
    agent_rows = []

    for handle in failed:
        agent = env.agents[handle]
        distance = float(obs_builder._current_distance_to_waypoint(handle))
        slack = float(obs_builder._deadline_slack(handle, distance))
        local_mask = obs_builder._build_local_action_mask(handle)
        coordinated_mask = obs_builder._coordination_masks.get(handle, local_mask)
        targets = valid_action_targets(obs_builder, handle)
        action_rows = []

        for action, target_info in targets.items():
            target_occupant = target_info["target_occupant"]
            if target_occupant in failed_set:
                blocked_edges[handle].add(int(target_occupant))
            corridor_blocker = first_corridor_blocker(obs_builder, handle, action)
            if corridor_blocker is not None and corridor_blocker["agent"] in failed_set:
                blocked_edges[handle].add(int(corridor_blocker["agent"]))
            action_rows.append(
                {
                    **target_info,
                    "local_allowed": bool(local_mask[action] >= 0.5),
                    "coordinated_allowed": bool(coordinated_mask[action] >= 0.5),
                    "corridor_blocker": corridor_blocker,
                }
            )

        agent_positions = positions.get(handle, [])
        last_change_time = 0
        previous_position = object()
        for step, position in enumerate(agent_positions, start=1):
            if position != previous_position:
                last_change_time = step
                previous_position = position

        agent_rows.append(
            {
                "agent": int(handle),
                "state": state_name(agent.state),
                "position": position_key(agent.position),
                "direction": agent.direction,
                "latest_arrival": agent.latest_arrival,
                "missed_by": (
                    max(0, int(env._elapsed_steps) - agent.latest_arrival)
                    if agent.latest_arrival is not None
                    else None
                ),
                "stationary_tail": max(0, int(env._elapsed_steps) - last_change_time),
                "distance": distance,
                "slack": slack,
                "actions": {
                    action_name(action): actions_by_agent[handle].count(action)
                    for action in sorted(set(actions_by_agent[handle]))
                },
                "candidate_actions": action_rows,
            }
        )

    success_rate = sum(int(is_done_state(agent.state)) for agent in env.agents) / env.get_num_agents()
    return {
        "seed": args.seed,
        "env_time": int(env._elapsed_steps),
        "num_agents": env.get_num_agents(),
        "success_rate": success_rate,
        "normalized_reward": normalized_reward(env, reward_values),
        "failed_agents": agent_rows,
        "blocked_edges": {
            str(agent): sorted(int(other) for other in blockers)
            for agent, blockers in sorted(blocked_edges.items())
        },
        "blocked_components": connected_components(blocked_edges),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay a sampled Flatland episode and inspect final blocked clusters."
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTHONPATH", str(repo_root()))
    result = analyze_episode(args)
    print(
        "seed={seed} reward={reward:.6f} success={success:.6f} failed={failed}".format(
            seed=result["seed"],
            reward=result["normalized_reward"],
            success=result["success_rate"],
            failed=len(result["failed_agents"]),
        )
    )
    print(f"blocked_components={result['blocked_components']}")
    for agent in result["failed_agents"]:
        blockers = result["blocked_edges"].get(str(agent["agent"]), [])
        print(
            "agent={agent} state={state} pos={position} dir={direction} "
            "dist={distance:.1f} slack={slack:.1f} stationary={stationary_tail} "
            "blocked_by={blocked}".format(
                blocked=blockers,
                **agent,
            )
        )
        for action in agent["candidate_actions"]:
            blocker = action["corridor_blocker"]
            blocker_text = ""
            if blocker is not None:
                blocker_text = (
                    f" corridor_blocker=a{blocker['agent']}"
                    f"/{blocker['relation']}@{blocker['position']}"
                )
            print(
                "  {action}: target={target} occ={target_occupant} "
                "local={local_allowed} coord={coordinated_allowed}{blocker}".format(
                    blocker=blocker_text,
                    **action,
                )
            )

    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
