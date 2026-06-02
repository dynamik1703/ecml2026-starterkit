#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env_action import RailEnvActions


DEFAULT_POLICY = "submission.hybrid_policy.MyPolicy"
DEFAULT_OBS_BUILDER = "submission.my_observation_builder.MyObservationBuilder"
DEFAULT_REWARDS = "flatland.envs.rewards.ECML2026Rewards"
DEFAULT_BASE_STATE = "reinforcement-learning/sampling/level_0_scenario_1.pkl"


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


def action_id(action: Any) -> int:
    if hasattr(action, "value"):
        return int(action.value)
    return int(action)


def action_name(action: int) -> str:
    try:
        return RailEnvActions(action).name
    except ValueError:
        return str(action)


def observation_for(observations: Any, handle: int) -> Any | None:
    if isinstance(observations, dict):
        return observations.get(handle)
    if handle < len(observations):
        return observations[handle]
    return None


def parse_agents(value: str | None, handles: list[int]) -> set[int] | None:
    if value is None:
        return None
    selected = {int(item) for item in value.split(",") if item.strip()}
    unknown = sorted(selected - set(handles))
    if unknown:
        raise ValueError(f"Unknown agent handles: {unknown}")
    return selected


def state_name(state: Any) -> str:
    return getattr(state, "name", str(state))


def position_key(position: Any) -> str | None:
    return None if position is None else str(tuple(position))


def mask_values(observation: Any) -> list[float]:
    if observation is None:
        return [0.0] * 5
    return [float(value) for value in np.asarray(observation)[-5:]]


def distance_and_slack(obs_builder: Any, handle: int) -> tuple[float, float]:
    try:
        distance = float(obs_builder._current_distance_to_waypoint(handle))
        slack = float(obs_builder._deadline_slack(handle, distance))
        return distance, slack
    except Exception:
        return float("nan"), float("nan")


def done_all(dones: Any) -> bool:
    if isinstance(dones, dict):
        return bool(dones.get("__all__", False))
    return bool(dones)


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


def lock_summary(
    lock: dict[str, Any] | None,
    now: int,
    obs_builder: Any | None = None,
) -> dict[str, Any]:
    if lock is None:
        return {
            "lock_owner": None,
            "lock_age": None,
            "lock_edges": 0,
            "lock_cells": "",
            "lock_owner_position": None,
            "lock_owner_distance": None,
            "lock_owner_slack": None,
        }
    cells = sorted(lock["cells"])
    owner_position = None
    owner_distance = None
    owner_slack = None
    if obs_builder is not None:
        owner = int(lock["owner"])
        if owner < len(obs_builder.env.agents):
            owner_agent = obs_builder.env.agents[owner]
            owner_position = position_key(owner_agent.position)
            owner_distance, owner_slack = distance_and_slack(obs_builder, owner)
    return {
        "lock_owner": int(lock["owner"]),
        "lock_age": now - int(lock["step"]),
        "lock_edges": len(lock["edges"]),
        "lock_cells": ";".join(str(cell) for cell in cells[:8]),
        "lock_owner_position": owner_position,
        "lock_owner_distance": owner_distance,
        "lock_owner_slack": owner_slack,
    }


def apply_locks_with_trace(
    policy: Any,
    handles: list[int],
    observations: list[Any],
    actions: dict[int, RailEnvActions],
    obs_builder: Any,
) -> tuple[dict[int, RailEnvActions], list[dict[str, Any]]]:
    policy._sync_corridor_locks(obs_builder)
    observations_by_handle = dict(zip(handles, observations))
    adjusted = dict(actions)
    planned_locks = []
    events = []
    elapsed = int(obs_builder.env._elapsed_steps)

    for handle in sorted(handles, key=lambda h: policy._priority_key(obs_builder, h)):
        action = adjusted.get(handle, RailEnvActions.DO_NOTHING)
        action_int = action_id(action)
        edges = policy._corridor_edges_for_action(obs_builder, handle, action_int)
        if len(edges) < policy.TEMPORAL_CORRIDOR_MIN_EDGES:
            continue

        lock = policy._conflicting_corridor_lock(handle, edges, planned_locks)
        if lock is not None and policy._should_wait_for_corridor_lock(lock, handle):
            fallback = policy._corridor_lock_fallback(
                observations_by_handle.get(handle)
            )
            if fallback is not None:
                adjusted[handle] = RailEnvActions(fallback)
            events.append(
                {
                    "kind": "blocked",
                    "agent_id": int(handle),
                    "action_before": action_name(action_int),
                    "action_before_id": action_int,
                    "action_after": (
                        action_name(action_id(adjusted[handle]))
                        if fallback is not None
                        else action_name(action_int)
                    ),
                    "action_after_id": (
                        action_id(adjusted[handle])
                        if fallback is not None
                        else action_int
                    ),
                    **lock_summary(lock, elapsed, obs_builder),
                }
            )
            continue

        if lock is not None:
            events.append(
                {
                    "kind": "allowed_after_max_wait",
                    "agent_id": int(handle),
                    "action_before": action_name(action_int),
                    "action_before_id": action_int,
                    "action_after": action_name(action_int),
                    "action_after_id": action_int,
                    **lock_summary(lock, elapsed, obs_builder),
                }
            )

        new_lock = policy._make_corridor_lock(obs_builder, handle, edges)
        policy._corridor_locks[handle] = new_lock
        planned_locks.append(new_lock)
        events.append(
            {
                "kind": "created",
                "agent_id": int(handle),
                "action_before": action_name(action_int),
                "action_before_id": action_int,
                "action_after": action_name(action_int),
                "action_after_id": action_int,
                **lock_summary(new_lock, elapsed, obs_builder),
            }
        )

    return adjusted, events


def make_env(args: argparse.Namespace, obs_builder: Any, rewards: Any) -> Any:
    state_pkl = args.base_state_pkl if args.sampled else args.state_pkl
    if state_pkl is None:
        raise ValueError("Pass --state-pkl or use --sampled with --base-state-pkl.")
    env, _ = RailEnvPersister.load_new(
        state_pkl,
        obs_builder=obs_builder,
        rewards=rewards,
    )
    if args.sampled:
        if args.num_agents is not None:
            env.number_of_agents = args.num_agents
        env = load_sampling_env_generator()(
            env,
            line_length=args.line_length,
            scene=args.scene,
        )
        env.reset(random_seed=args.seed)
    return env


def policy_actions_with_trace(
    policy: Any,
    handles: list[int],
    observations: list[Any],
    obs_builder: Any,
) -> tuple[dict[int, int], dict[int, int], dict[int, int], list[dict[str, Any]]]:
    env = obs_builder.env
    rl_actions = policy.rl_policy.act_many(handles, observations)
    rl_action_ids = {handle: action_id(action) for handle, action in rl_actions.items()}
    if (
        env is None
        or env.get_num_agents() <= 2
        or env.get_num_agents() < 6
        or max(env.height, env.width) < 100
    ):
        return rl_action_ids, rl_action_ids, rl_action_ids, []

    detour_actions = policy._detour_around_long_opposing(
        rl_actions,
        obs_builder,
    )
    final_actions, events = apply_locks_with_trace(
        policy,
        handles,
        observations,
        detour_actions,
        obs_builder,
    )
    return (
        rl_action_ids,
        {handle: action_id(action) for handle, action in detour_actions.items()},
        {handle: action_id(action) for handle, action in final_actions.items()},
        events,
    )


def event_rows(
    env: Any,
    obs_builder: Any,
    observations: Any,
    events: list[dict[str, Any]],
    rl_actions: dict[int, int],
    detour_actions: dict[int, int],
    final_actions: dict[int, int],
) -> list[dict[str, Any]]:
    rows = []
    elapsed = int(env._elapsed_steps)
    for event in events:
        handle = event["agent_id"]
        agent = env.agents[handle]
        observation = observation_for(observations, handle)
        distance, slack = distance_and_slack(obs_builder, handle)
        mask = mask_values(observation)
        rows.append(
            {
                "env_time": elapsed,
                "agent_id": handle,
                "kind": event["kind"],
                "state": state_name(agent.state),
                "position": position_key(agent.position),
                "direction": agent.direction,
                "distance": distance,
                "slack": slack,
                "mask_N": mask[0],
                "mask_L": mask[1],
                "mask_F": mask[2],
                "mask_R": mask[3],
                "mask_S": mask[4],
                "rl_action": action_name(rl_actions[handle]),
                "detour_action": action_name(detour_actions[handle]),
                "final_action": action_name(final_actions[handle]),
                **event,
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trace temporal corridor locks in the hybrid policy."
    )
    parser.add_argument("--state-pkl", type=Path)
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--sampled", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--agents")
    parser.add_argument("--from-step", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-print", type=int, default=80)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    obs_builder = load_symbol(args.obs_builder)()
    rewards = load_symbol(args.rewards)()
    policy = load_symbol(args.policy)()
    env = make_env(args, obs_builder, rewards)
    handles = list(env.get_agent_handles())
    selected = parse_agents(args.agents, handles)
    max_steps = args.max_steps or env._max_episode_steps
    rewards_seen: list[float] = []
    rows: list[dict[str, Any]] = []

    for _ in range(max_steps):
        observations = env._get_observations()
        obs_list = [observation_for(observations, handle) for handle in handles]
        rl_actions, detour_actions, final_actions, events = policy_actions_with_trace(
            policy,
            handles,
            obs_list,
            obs_builder,
        )

        elapsed = int(env._elapsed_steps)
        if elapsed >= args.from_step:
            step_rows = event_rows(
                env,
                obs_builder,
                observations,
                events,
                rl_actions,
                detour_actions,
                final_actions,
            )
            if selected is not None:
                step_rows = [
                    row for row in step_rows if row["agent_id"] in selected
                ]
            rows.extend(step_rows)

        observations, rewards_by_agent, dones, _ = env.step(final_actions)
        rewards_seen.extend(
            float(rewards_by_agent.get(handle, 0.0)) for handle in handles
        )
        if done_all(dones):
            break

    summary = {
        "seed": args.seed,
        "env_time": int(env._elapsed_steps),
        "success_rate": (
            sum(int(state_name(agent.state) in {"DONE", "DONE_REMOVED"} or agent.state == 6)
                for agent in env.agents)
            / env.get_num_agents()
        ),
        "normalized_reward": normalized_reward(env, rewards_seen),
        "events": len(rows),
        "blocked_events": sum(1 for row in rows if row["kind"] == "blocked"),
        "created_events": sum(1 for row in rows if row["kind"] == "created"),
        "allowed_after_max_wait_events": sum(
            1 for row in rows if row["kind"] == "allowed_after_max_wait"
        ),
    }

    print(
        "seed={seed} reward={normalized_reward:.6f} success={success_rate:.6f} "
        "events={events} blocked={blocked_events} created={created_events} "
        "allowed_after_max_wait={allowed_after_max_wait_events}".format(
            **summary
        )
    )
    for row in rows[: args.max_print]:
        print(
            "t={env_time} a{agent_id} {kind} pos={position} "
            "rl={rl_action} detour={detour_action} final={final_action} "
            "lock_owner={lock_owner} owner_slack={lock_owner_slack} "
            "age={lock_age} edges={lock_edges} "
            "dist={distance:.1f} slack={slack:.1f}".format(**row)
        )

    result = {"summary": summary, "events": rows}
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.output_csv is not None:
        if rows:
            with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
        else:
            args.output_csv.write_text("", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
