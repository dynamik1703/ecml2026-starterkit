#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import sys
from pathlib import Path
from typing import Any

import numpy as np
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rail_env_action import RailEnvActions


DEFAULT_POLICY = "submission.my_policy.MyPolicy"
DEFAULT_OBS_BUILDER = "submission.my_observation_builder.MyObservationBuilder"
DEFAULT_REWARDS = "flatland.envs.rewards.ECML2026Rewards"

ACTION_COLUMNS = ("N", "L", "F", "R", "S")


def load_symbol(path: str) -> Any:
    module_name, symbol_name = path.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def parse_agents(value: str | None, handles: list[int]) -> list[int]:
    if value is None:
        return handles
    selected = [int(item) for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(handles))
    if unknown:
        raise ValueError(f"Unknown agent handles: {unknown}")
    return selected


def observation_for(observations: Any, handle: int) -> Any | None:
    if isinstance(observations, dict):
        return observations.get(handle)
    if handle < len(observations):
        return observations[handle]
    return None


def action_name(action: int) -> str:
    try:
        return RailEnvActions(action).name
    except ValueError:
        return str(action)


def state_name(state: Any) -> str:
    return getattr(state, "name", str(state))


def mask_values(observation: Any) -> list[float]:
    if observation is None:
        return [0.0] * len(ACTION_COLUMNS)
    mask = np.asarray(observation)[-len(ACTION_COLUMNS) :]
    return [float(value) for value in mask]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a serialized Flatland episode and emit per-step policy, "
            "mask, distance and slack traces."
        )
    )
    parser.add_argument("--state-pkl", type=Path, required=True)
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument(
        "--agents",
        help="Comma-separated agent handles to print. All agents are simulated.",
    )
    parser.add_argument("--from-step", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    obs_builder = load_symbol(args.obs_builder)()
    rewards = load_symbol(args.rewards)()
    policy = load_symbol(args.policy)()
    env, _ = RailEnvPersister.load_new(
        args.state_pkl,
        obs_builder=obs_builder,
        rewards=rewards,
    )

    handles = list(env.get_agent_handles())
    selected_handles = parse_agents(args.agents, handles)
    max_steps = args.max_steps or env._max_episode_steps

    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "env_time",
            "agent_id",
            "state",
            "position",
            "direction",
            "speed",
            "distance",
            "slack",
            *[f"mask_{name}" for name in ACTION_COLUMNS],
            "action",
            "action_id",
        ]
    )

    for _ in range(max_steps):
        observations = env._get_observations()
        actions = {}
        for handle in handles:
            observation = observation_for(observations, handle)
            actions[handle] = int(policy.act(observation)) if observation is not None else 0

        elapsed = int(env._elapsed_steps)
        if elapsed >= args.from_step:
            for handle in selected_handles:
                agent = env.agents[handle]
                observation = observation_for(observations, handle)
                distance, slack = distance_and_slack(obs_builder, handle)
                writer.writerow(
                    [
                        elapsed,
                        handle,
                        state_name(agent.state),
                        agent.position,
                        agent.direction,
                        float(agent.speed_counter.speed),
                        distance,
                        slack,
                        *mask_values(observation),
                        action_name(actions[handle]),
                        actions[handle],
                    ]
                )

        _, _, dones, _ = env.step(actions)
        if done_all(dones):
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
