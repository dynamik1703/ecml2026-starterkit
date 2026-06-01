#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from flatland.envs.persistence import RailEnvPersister


DEFAULT_POLICY = "submission.hybrid_policy.MyPolicy"
DEFAULT_OBS_BUILDER = "submission.my_observation_builder.MyObservationBuilder"
DEFAULT_REWARDS = "flatland.envs.rewards.ECML2026Rewards"
DEFAULT_BASE_STATE = (
    "reinforcement-learning/sampling/level_0_scenario_1.pkl"
)


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


def run_episode(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    obs_builder = load_symbol(args.obs_builder)()
    rewards = load_symbol(args.rewards)()
    policy = load_symbol(args.policy)()
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
    done = False
    env_time = 0
    while not done and env_time < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        actions = {
            handle: action_id(action)
            for handle, action in policy.act_many(
                handles,
                observation_list(observations, handles),
            ).items()
        }
        observations, rewards_by_agent, dones, _ = env.step(actions)
        reward_values.extend(float(rewards_by_agent.get(handle, 0.0)) for handle in handles)
        done = bool(dones.get("__all__", False))
        env_time = int(env._elapsed_steps)

    success_rate = sum(int(agent.state == 6) for agent in env.agents) / env.get_num_agents()
    return {
        "seed": seed,
        "env_time": env_time,
        "success_rate": success_rate,
        "normalized_reward": normalized_reward(env, reward_values),
        "num_agents": env.get_num_agents(),
        "max_episode_steps": env._max_episode_steps,
        "line_length": args.line_length,
        "scene": args.scene or "scene_5",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a policy on sampled ECML-style scenarios."
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--policy", default=DEFAULT_POLICY)
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
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTHONPATH", str(repo_root()))

    rows = [run_episode(args, args.seed + index) for index in range(args.episodes)]
    total_reward = sum(row["normalized_reward"] for row in rows)
    total_success = sum(row["success_rate"] for row in rows)
    summary = {
        "policy": args.policy,
        "base_state_pkl": str(args.base_state_pkl),
        "episodes": args.episodes,
        "reward_mean": total_reward / len(rows) if rows else 0.0,
        "success_rate_mean": total_success / len(rows) if rows else 0.0,
        "rows": rows,
    }

    print("seed,env_time,success_rate,normalized_reward,num_agents,max_episode_steps")
    for row in rows:
        print(
            f"{row['seed']},{row['env_time']},{row['success_rate']:.6g},"
            f"{row['normalized_reward']:.6g},{row['num_agents']},"
            f"{row['max_episode_steps']}"
        )
    print(
        "\nTotals: "
        f"reward_mean={summary['reward_mean']:.6g}, "
        f"success_rate_mean={summary['success_rate_mean']:.6g}, "
        f"episodes={args.episodes}"
    )

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")

    if args.output_csv is not None:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.output_csv.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
