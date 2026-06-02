#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
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

ACTION_COLUMNS = ("N", "L", "F", "R", "S")


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


def action_id(action: Any) -> int:
    if hasattr(action, "value"):
        return int(action.value)
    return int(action)


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
            "Replay a serialized or sampled Flatland episode and emit per-step policy, "
            "mask, distance and slack traces."
        )
    )
    parser.add_argument("--state-pkl", type=Path)
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
        help="Base environment used with --sampled.",
    )
    parser.add_argument(
        "--sampled",
        action="store_true",
        help="Apply the starterkit sampling generator before tracing.",
    )
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
    parser.add_argument(
        "--agents",
        help="Comma-separated agent handles to print. All agents are simulated.",
    )
    parser.add_argument("--from-step", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    obs_builder = load_symbol(args.obs_builder)()
    rewards = load_symbol(args.rewards)()
    policy = load_symbol(args.policy)()
    env = make_env(args, obs_builder, rewards)

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
        observation_list = [observation_for(observations, handle) for handle in handles]
        actions = {
            handle: action_id(action)
            for handle, action in policy.act_many(handles, observation_list).items()
        }

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
