#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from flatland.envs.persistence import RailEnvPersister

from submission.my_policy import ActorCritic


DEFAULT_BASE_STATE = "reinforcement-learning/sampling/level_0_scenario_1.pkl"
DEFAULT_TEACHER_POLICY = "submission.rerank_policy.MyPolicy"
DEFAULT_OBS_BUILDER = "submission.my_observation_builder.MyObservationBuilder"
DEFAULT_ROUTE_CONFLICT_OBS_BUILDER = (
    "submission.my_observation_builder.MyRouteConflictObservationBuilder"
)
DEFAULT_TRAJECTORY_CONFLICT_OBS_BUILDER = (
    "submission.my_observation_builder.MyTrajectoryConflictObservationBuilder"
)
DEFAULT_REWARDS = "flatland.envs.rewards.ECML2026Rewards"
BASE_OBS_SIZE = 36
ROUTE_CONFLICT_OBS_SIZE = 52
TRAJECTORY_CONFLICT_OBS_SIZE = 64


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


def make_env(args: argparse.Namespace, seed: int) -> tuple[Any, dict[int, Any]]:
    obs_builder = load_symbol(args.obs_builder)()
    rewards = load_symbol(args.rewards)()
    env, _ = RailEnvPersister.load_new(
        args.base_state_pkl,
        obs_builder=obs_builder,
        rewards=rewards,
    )
    env.number_of_agents = args.num_agents
    env = load_sampling_env_generator()(env, line_length=args.line_length, scene=args.scene)
    observations, _ = env.reset(random_seed=seed)
    return env, observations


def observation_list(observations: dict[int, Any], handles: list[int]) -> list[Any]:
    return [observations[handle] for handle in handles]


def collect_dataset(args: argparse.Namespace) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    teacher = load_symbol(args.teacher_policy)()
    checkpoint_path = args.init_checkpoint if args.init_checkpoint.exists() else None
    reference = ActorCritic(
        obs_size=args.obs_size,
        n_actions=args.n_actions,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
    )
    reference.eval()
    observations_out: list[np.ndarray] = []
    actions_out: list[int] = []
    invalid_teacher_actions = 0
    teacher_reference_disagreements = 0
    valid_teacher_samples = 0
    action_counts: Counter[int] = Counter()
    disagreement_action_counts: Counter[int] = Counter()
    success_rates = []
    normalized_rewards = []

    for episode in range(args.episodes):
        seed = args.seed + episode
        env, observations = make_env(args, seed)
        reward_values: list[float] = []
        done = False
        while not done and env._elapsed_steps < env._max_episode_steps:
            handles = list(env.get_agent_handles())
            obs_list = observation_list(observations, handles)
            teacher_actions = {
                handle: action_id(action)
                for handle, action in teacher.act_many(handles, obs_list).items()
            }
            with torch.no_grad():
                reference_logits = reference.masked_logits(
                    np.asarray(obs_list, dtype=np.float32)
                )
                reference_actions = reference_logits.argmax(dim=-1).cpu().numpy()

            for reference_action, handle, observation in zip(
                reference_actions,
                handles,
                obs_list,
            ):
                action = teacher_actions[handle]
                obs_array = np.asarray(observation, dtype=np.float32)
                mask = obs_array[args.obs_size : args.obs_size + args.n_actions]
                if (
                    0 <= action < args.n_actions
                    and len(mask) == args.n_actions
                    and mask[action] >= 0.5
                ):
                    valid_teacher_samples += 1
                    disagrees = action != int(reference_action)
                    if disagrees:
                        teacher_reference_disagreements += 1
                        disagreement_action_counts[action] += 1
                    if args.disagreement_only and not disagrees:
                        continue
                    observations_out.append(obs_array)
                    actions_out.append(action)
                    action_counts[action] += 1
                else:
                    invalid_teacher_actions += 1

            observations, rewards_by_agent, dones, _ = env.step(teacher_actions)
            reward_values.extend(
                float(rewards_by_agent.get(handle, 0.0))
                for handle in handles
            )
            done = bool(dones.get("__all__", False))

        success_rate = sum(int(agent.state == 6) for agent in env.agents) / env.get_num_agents()
        success_rates.append(success_rate)
        normalized_rewards.append(
            float(
                env.rewards.normalize(
                    *reward_values,
                    max_episode_steps=env._max_episode_steps,
                    num_agents=env.get_num_agents(),
                )
            )
            if reward_values
            else 0.0
        )

    if not observations_out:
        raise RuntimeError("Teacher collection produced no valid training samples.")

    stats = {
        "samples": len(observations_out),
        "valid_teacher_samples": valid_teacher_samples,
        "invalid_teacher_actions": invalid_teacher_actions,
        "teacher_reference_disagreements": teacher_reference_disagreements,
        "action_counts": dict(sorted(action_counts.items())),
        "disagreement_action_counts": dict(sorted(disagreement_action_counts.items())),
        "teacher_reward_mean": float(np.mean(normalized_rewards)),
        "teacher_success_rate_mean": float(np.mean(success_rates)),
    }
    return (
        torch.as_tensor(np.asarray(observations_out, dtype=np.float32)),
        torch.as_tensor(actions_out, dtype=torch.long),
        stats,
    )


def train_behavior_clone(
    args: argparse.Namespace,
    observations: torch.Tensor,
    actions: torch.Tensor,
) -> tuple[ActorCritic, dict[str, float]]:
    checkpoint_path = args.init_checkpoint if args.init_checkpoint.exists() else None
    policy = ActorCritic(
        obs_size=args.obs_size,
        n_actions=args.n_actions,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
    )
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    num_samples = observations.shape[0]
    class_weights = None
    if args.class_balanced_loss:
        counts = torch.bincount(actions, minlength=args.n_actions).float()
        safe_counts = torch.clamp(counts, min=1.0)
        class_weights = counts.sum() / (args.n_actions * safe_counts)
        class_weights = torch.clamp(class_weights, max=args.max_class_weight)
        class_weights = class_weights / class_weights.mean()
        print(
            "class_weights="
            f"{[round(float(weight), 6) for weight in class_weights.tolist()]}",
            flush=True,
        )
    final_loss = 0.0
    final_accuracy = 0.0

    for epoch in range(args.epochs):
        permutation = torch.randperm(num_samples)
        losses = []
        correct = 0
        seen = 0
        for start in range(0, num_samples, args.batch_size):
            batch_idx = permutation[start : start + args.batch_size]
            obs_batch = observations[batch_idx]
            action_batch = actions[batch_idx]
            logits = policy.masked_logits(obs_batch)
            loss = F.cross_entropy(logits, action_batch, weight=class_weights)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            losses.append(float(loss.item()))
            predictions = logits.argmax(dim=-1)
            correct += int((predictions == action_batch).sum().item())
            seen += int(action_batch.numel())

        final_loss = float(np.mean(losses))
        final_accuracy = correct / max(1, seen)
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"loss={final_loss:.6g} accuracy={final_accuracy:.6g}",
            flush=True,
        )

    return policy, {
        "loss": final_loss,
        "accuracy": final_accuracy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Behavior-clone the ActorCritic policy from a stronger teacher policy."
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--teacher-policy", default=DEFAULT_TEACHER_POLICY)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument(
        "--use-route-conflict-obs",
        action="store_true",
        help="Use MyRouteConflictObservationBuilder and obs_size=52.",
    )
    parser.add_argument(
        "--use-trajectory-conflict-obs",
        action="store_true",
        help=(
            "Use MyTrajectoryConflictObservationBuilder and obs_size=64. "
            "Includes route-conflict plus trajectory/priority features."
        ),
    )
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--init-checkpoint", type=Path, default=repo_root() / "submission/checkpoint.pt")
    parser.add_argument("--output-checkpoint", type=Path, default=Path("/private/tmp/ecml_bc.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--obs-size", type=int)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument(
        "--class-balanced-loss",
        action="store_true",
        help="Use inverse-frequency action weights for behavior cloning.",
    )
    parser.add_argument(
        "--disagreement-only",
        action="store_true",
        help="Train only on valid states where the teacher disagrees with the init policy.",
    )
    parser.add_argument("--max-class-weight", type=float, default=10.0)
    return parser.parse_args()


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.use_trajectory_conflict_obs:
        if args.obs_builder == DEFAULT_OBS_BUILDER:
            args.obs_builder = DEFAULT_TRAJECTORY_CONFLICT_OBS_BUILDER
        if args.obs_size is None:
            args.obs_size = TRAJECTORY_CONFLICT_OBS_SIZE
    elif args.use_route_conflict_obs:
        if args.obs_builder == DEFAULT_OBS_BUILDER:
            args.obs_builder = DEFAULT_ROUTE_CONFLICT_OBS_BUILDER
        if args.obs_size is None:
            args.obs_size = ROUTE_CONFLICT_OBS_SIZE
    elif args.obs_size is None:
        args.obs_size = BASE_OBS_SIZE

    if (
        args.use_route_conflict_obs
        and not args.use_trajectory_conflict_obs
        and args.obs_size != ROUTE_CONFLICT_OBS_SIZE
    ):
        raise ValueError(
            "--use-route-conflict-obs expects --obs-size "
            f"{ROUTE_CONFLICT_OBS_SIZE}, got {args.obs_size}."
        )
    if args.use_trajectory_conflict_obs and args.obs_size != TRAJECTORY_CONFLICT_OBS_SIZE:
        raise ValueError(
            "--use-trajectory-conflict-obs expects --obs-size "
            f"{TRAJECTORY_CONFLICT_OBS_SIZE}, got {args.obs_size}."
        )
    return args


def main() -> int:
    args = finalize_args(parse_args())
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(
        "bc_config "
        f"obs_builder={args.obs_builder} "
        f"obs_size={args.obs_size} "
        f"teacher_policy={args.teacher_policy} "
        f"episodes={args.episodes} "
        f"num_agents={args.num_agents} "
        f"line_length={args.line_length}",
        flush=True,
    )

    observations, actions, collection_stats = collect_dataset(args)
    print(
        "collection "
        f"samples={collection_stats['samples']} "
        f"valid_teacher_samples={collection_stats['valid_teacher_samples']} "
        f"invalid_teacher_actions={collection_stats['invalid_teacher_actions']} "
        f"teacher_reference_disagreements={collection_stats['teacher_reference_disagreements']} "
        f"teacher_reward_mean={collection_stats['teacher_reward_mean']:.6g} "
        f"teacher_success_rate_mean={collection_stats['teacher_success_rate_mean']:.6g} "
        f"action_counts={collection_stats['action_counts']} "
        f"disagreement_action_counts={collection_stats['disagreement_action_counts']}",
        flush=True,
    )
    policy, train_stats = train_behavior_clone(args, observations, actions)

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "config": vars(args),
            "collection_stats": collection_stats,
            "train_stats": train_stats,
        },
        args.output_checkpoint,
    )
    print(f"saved_checkpoint={args.output_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
