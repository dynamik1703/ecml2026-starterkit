#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flatland.envs.persistence import RailEnvPersister

from submission import runtime_context
from submission.dispatch_ranker import (
    DISPATCH_PAIR_FEATURE_NAMES,
    dispatch_training_rows,
)


DEFAULT_POLICY = "submission.scene_aware_competition_policy.MyPolicy"
DEFAULT_OBS_BUILDER = "submission.my_observation_builder.MyObservationBuilder"
DEFAULT_REWARDS = "flatland.envs.rewards.ECML2026Rewards"
DEFAULT_BASE_STATE = "reinforcement-learning/sampling/level_0_scenario_1.pkl"


def repo_root() -> Path:
    return ROOT


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


def instantiate_policy(args: argparse.Namespace) -> Any:
    policy_cls = load_symbol(args.policy)
    if args.policy_checkpoint is None:
        return policy_cls()
    return policy_cls(checkpoint_path=str(args.policy_checkpoint))


def mapf_policy(policy: Any) -> Any | None:
    nested = getattr(policy, "mapf_policy", None)
    if nested is not None:
        return nested
    if hasattr(policy, "_last_mapf_candidates"):
        return policy
    return None


def run_episode(args: argparse.Namespace, seed: int) -> list[dict[str, Any]]:
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
    runtime_context.set_seed(seed)
    runtime_context.set_scene(args.scene)

    rows: list[dict[str, Any]] = []
    done = False
    while not done and int(env._elapsed_steps) < int(env._max_episode_steps):
        handles = list(env.get_agent_handles())
        actions = {
            handle: action_id(action)
            for handle, action in policy.act_many(
                handles,
                observation_list(observations, handles),
            ).items()
        }
        planner = mapf_policy(policy)
        candidates = list(getattr(planner, "_last_mapf_candidates", []) or [])
        if candidates:
            rows.extend(
                dispatch_training_rows(
                    env,
                    candidates,
                    seed=seed,
                    step=int(env._elapsed_steps),
                )
            )

        observations, _rewards_by_agent, dones, _ = env.step(actions)
        done = bool(dones.get("__all__", False))
        if args.max_steps is not None and int(env._elapsed_steps) >= args.max_steps:
            break

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect pairwise conflict rows for the high-level dispatch ranker."
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
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7410)
    parser.add_argument("--num-agents", type=int, default=80)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
        default="scene_3",
    )
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    os.environ.setdefault("PYTHONPATH", str(repo_root()))
    os.environ.setdefault("ECML_DISPATCH_RANKER_ENABLED", "0")

    rows: list[dict[str, Any]] = []
    for index in range(args.episodes):
        rows.extend(run_episode(args, args.seed + index))

    fieldnames = [
        "seed",
        "step",
        "scene",
        "left_handle",
        "right_handle",
        "teacher_probability",
        "label",
        "target_same_raw",
        "reverse_immediate_raw",
        "shared_edges_raw",
        "opposing_edges_raw",
        *DISPATCH_PAIR_FEATURE_NAMES,
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {len(rows)} dispatch-ranker rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
