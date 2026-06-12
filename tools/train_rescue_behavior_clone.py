#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from flatland.envs.persistence import RailEnvPersister

from submission.my_policy import ActorCritic
from tools.analyze_policy_action_diffs import policy_actions
from tools.evaluate_sampled import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_REWARDS,
    load_sampling_env_generator,
    repo_root,
)


DEFAULT_TRAJECTORY_CONFLICT_OBS_BUILDER = (
    "submission.my_observation_builder.MyTrajectoryConflictObservationBuilder"
)
DEFAULT_ACTION_CONFLICT_OBS_BUILDER = (
    "submission.my_observation_builder.MyActionConflictObservationBuilder"
)
DEFAULT_BASELINE_POLICY = "submission.sequence_success_policy.MyPolicy"
BASE_OBS_SIZE = 36
TRAJECTORY_CONFLICT_OBS_SIZE = 64
ACTION_CONFLICT_OBS_SIZE = 79
ACTION_COUNT = 5


def load_symbol(path: str) -> Any:
    module_name, symbol_name = path.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def instantiate_policy(policy_path: str, checkpoint: Path | None) -> Any:
    policy_cls = load_symbol(policy_path)
    if checkpoint is not None:
        try:
            return policy_cls(checkpoint_path=str(checkpoint))
        except TypeError:
            pass
    return policy_cls()


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


def safe_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def utility(row: dict[str, str], args: argparse.Namespace) -> float:
    reward_delta = safe_float(row.get("reward_delta"))
    success_delta = safe_float(row.get("success_delta"))
    failed_delta = safe_float(row.get("failed_agents_delta"))
    reward_delta = reward_delta if np.isfinite(reward_delta) else 0.0
    success_delta = success_delta if np.isfinite(success_delta) else 0.0
    failed_delta = failed_delta if np.isfinite(failed_delta) else 0.0
    return float(
        reward_delta
        + args.success_weight * success_delta
        - args.failed_weight * max(0.0, failed_delta)
    )


def is_positive_rescue(row: dict[str, str], args: argparse.Namespace) -> bool:
    if (
        str(row.get("event_kind", "")).lower() == "positive_rescue"
        and safe_float(row.get("candidate_source_policy_diff_positive")) > 0.0
    ):
        return True
    reward_delta = safe_float(row.get("reward_delta"))
    success_delta = safe_float(row.get("success_delta"))
    failed_delta = safe_float(row.get("failed_agents_delta"))
    reward_delta = reward_delta if np.isfinite(reward_delta) else 0.0
    success_delta = success_delta if np.isfinite(success_delta) else 0.0
    failed_delta = failed_delta if np.isfinite(failed_delta) else 0.0
    if success_delta < -1e-9 or failed_delta > 0:
        return False
    return success_delta > 1e-9 or reward_delta > args.reward_epsilon


def is_avoidance_event(row: dict[str, str], args: argparse.Namespace) -> bool:
    if not args.include_avoidance_events:
        return False
    return str(row.get("event_kind", "")).lower() == "negative_baseline"


def read_rescue_events(
    paths: list[Path],
    args: argparse.Namespace,
) -> dict[int, dict[tuple[int, int], dict[str, Any]]]:
    best: dict[int, dict[tuple[int, int], dict[str, Any]]] = defaultdict(dict)
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if str(row.get("forced_applied", "True")).lower() == "false":
                    continue
                is_positive = is_positive_rescue(row, args)
                is_avoidance = is_avoidance_event(row, args)
                if not is_positive and not is_avoidance:
                    continue
                seed = int(float(row["seed"]))
                env_time = int(float(row["env_time"]))
                agent_id = int(float(row["agent_id"]))
                forced_action = int(float(row["forced_action"]))
                if forced_action < 0 or forced_action >= ACTION_COUNT:
                    continue
                score = utility(row, args)
                key = (env_time, agent_id)
                previous = best[seed].get(key)
                if previous is None or score > float(previous["utility"]):
                    best[seed][key] = {
                        "seed": seed,
                        "env_time": env_time,
                        "agent_id": agent_id,
                        "event_kind": (
                            "negative_baseline" if is_avoidance else "positive_rescue"
                        ),
                        "forced_action": forced_action,
                        "baseline_action": int(float(row.get("baseline_action", -1))),
                        "utility": score,
                        "reward_delta": safe_float(row.get("reward_delta")),
                        "success_delta": safe_float(row.get("success_delta")),
                        "failed_agents_delta": safe_float(row.get("failed_agents_delta")),
                    }
    return best


def parse_seed_list(value: str | None) -> list[int]:
    if not value:
        return []
    seeds: list[int] = []
    for token in value.split(","):
        item = token.strip()
        if item:
            seeds.append(int(item))
    return list(dict.fromkeys(seeds))


def mask_for_obs(observation: Any, obs_size: int, n_actions: int) -> np.ndarray:
    obs = np.asarray(observation, dtype=np.float32)
    if obs.shape[0] >= obs_size + n_actions:
        return obs[-n_actions:]
    return np.ones(n_actions, dtype=np.float32)


def valid_action(observation: Any, action: int, obs_size: int, n_actions: int) -> bool:
    mask = mask_for_obs(observation, obs_size, n_actions)
    return 0 <= action < n_actions and action < len(mask) and mask[action] >= 0.5


def sample_weight_for_rescue(event: dict[str, Any], args: argparse.Namespace) -> float:
    if event.get("event_kind") == "negative_baseline":
        return args.avoidance_weight
    success_delta = float(event.get("success_delta", 0.0) or 0.0)
    if success_delta > 1e-9:
        return args.success_rescue_weight
    return args.reward_rescue_weight


def collect_training_samples(
    args: argparse.Namespace,
    rescue_events: dict[int, dict[tuple[int, int], dict[str, Any]]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
    policy = instantiate_policy(args.baseline_policy, args.baseline_checkpoint)
    observations_out: list[np.ndarray] = []
    actions_out: list[int] = []
    weights_out: list[float] = []
    rescue_hits = 0
    rescue_misses = 0
    rescue_invalid = 0
    rescue_baseline_mismatches = 0
    anchor_samples = 0
    anchor_action_counts: Counter[int] = Counter()
    rescue_action_counts: Counter[int] = Counter()
    avoidance_hits = 0
    rescue_seed_hits: Counter[int] = Counter()

    seeds = sorted(set(rescue_events) | set(args.anchor_seed_list))
    if not seeds:
        raise ValueError("No seeds selected from rescue CSVs or --anchor-seeds")

    for seed in seeds:
        env, observations = make_env(args, seed)
        events_for_seed = rescue_events.get(seed, {})
        seen_events: set[tuple[int, int]] = set()
        while int(env._elapsed_steps) < env._max_episode_steps:
            handles = list(env.get_agent_handles())
            obs_list = observation_list(observations, handles)
            obs_by_handle = dict(zip(handles, obs_list))
            actions = policy_actions(policy, handles, obs_list)
            step = int(env._elapsed_steps)

            for handle in handles:
                observation = obs_by_handle[handle]
                action = int(actions.get(handle, 0))
                event = events_for_seed.get((step, handle))
                if event is not None:
                    forced_action = int(event["forced_action"])
                    seen_events.add((step, handle))
                    if valid_action(observation, forced_action, args.obs_size, args.n_actions):
                        if int(event["baseline_action"]) != action:
                            rescue_baseline_mismatches += 1
                        observations_out.append(np.asarray(observation, dtype=np.float32))
                        actions_out.append(forced_action)
                        weights_out.append(sample_weight_for_rescue(event, args))
                        rescue_hits += 1
                        if event.get("event_kind") == "negative_baseline":
                            avoidance_hits += 1
                        rescue_seed_hits[seed] += 1
                        rescue_action_counts[forced_action] += 1
                    else:
                        rescue_invalid += 1
                    continue

                if args.anchor_weight <= 0.0:
                    continue
                if args.anchor_stride > 1 and step % args.anchor_stride != 0:
                    continue
                if not valid_action(observation, action, args.obs_size, args.n_actions):
                    continue
                observations_out.append(np.asarray(observation, dtype=np.float32))
                actions_out.append(action)
                weights_out.append(args.anchor_weight)
                anchor_samples += 1
                anchor_action_counts[action] += 1

            observations, _, dones, _ = env.step(actions)
            if bool(dones.get("__all__", False)):
                break

        rescue_misses += len(set(events_for_seed) - seen_events)

    if not observations_out:
        raise RuntimeError("No valid training samples collected")

    stats = {
        "samples": len(observations_out),
        "rescue_hits": rescue_hits,
        "avoidance_hits": avoidance_hits,
        "rescue_misses": rescue_misses,
        "rescue_invalid": rescue_invalid,
        "rescue_baseline_mismatches": rescue_baseline_mismatches,
        "anchor_samples": anchor_samples,
        "rescue_action_counts": dict(sorted(rescue_action_counts.items())),
        "anchor_action_counts": dict(sorted(anchor_action_counts.items())),
        "rescue_seed_hits": dict(sorted(rescue_seed_hits.items())),
        "seeds": seeds,
    }
    return (
        torch.as_tensor(np.asarray(observations_out, dtype=np.float32)),
        torch.as_tensor(actions_out, dtype=torch.long),
        torch.as_tensor(weights_out, dtype=torch.float32),
        stats,
    )


def train_policy(
    args: argparse.Namespace,
    observations: torch.Tensor,
    actions: torch.Tensor,
    weights: torch.Tensor,
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
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    num_samples = observations.shape[0]
    final_loss = 0.0
    final_accuracy = 0.0
    final_weighted_accuracy = 0.0

    weights = weights / weights.mean().clamp_min(1e-6)
    for epoch in range(args.epochs):
        permutation = torch.randperm(num_samples)
        losses = []
        correct = 0
        weighted_correct = 0.0
        weight_seen = 0.0
        seen = 0
        for start in range(0, num_samples, args.batch_size):
            batch_idx = permutation[start : start + args.batch_size]
            obs_batch = observations[batch_idx]
            action_batch = actions[batch_idx]
            weight_batch = weights[batch_idx]
            logits = policy.masked_logits(obs_batch)
            loss_values = F.cross_entropy(logits, action_batch, reduction="none")
            loss = (loss_values * weight_batch).sum() / weight_batch.sum().clamp_min(1e-6)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            losses.append(float(loss.item()))
            predictions = logits.argmax(dim=-1)
            is_correct = predictions == action_batch
            correct += int(is_correct.sum().item())
            weighted_correct += float((is_correct.float() * weight_batch).sum().item())
            weight_seen += float(weight_batch.sum().item())
            seen += int(action_batch.numel())

        final_loss = float(np.mean(losses))
        final_accuracy = correct / max(1, seen)
        final_weighted_accuracy = weighted_correct / max(1e-6, weight_seen)
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"loss={final_loss:.6g} "
            f"accuracy={final_accuracy:.6g} "
            f"weighted_accuracy={final_weighted_accuracy:.6g}",
            flush=True,
        )

    return policy, {
        "loss": final_loss,
        "accuracy": final_accuracy,
        "weighted_accuracy": final_weighted_accuracy,
    }


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.use_action_conflict_obs:
        if args.obs_builder == DEFAULT_OBS_BUILDER:
            args.obs_builder = DEFAULT_ACTION_CONFLICT_OBS_BUILDER
        if args.obs_size is None:
            args.obs_size = ACTION_CONFLICT_OBS_SIZE
    elif args.use_trajectory_conflict_obs:
        if args.obs_builder == DEFAULT_OBS_BUILDER:
            args.obs_builder = DEFAULT_TRAJECTORY_CONFLICT_OBS_BUILDER
        if args.obs_size is None:
            args.obs_size = TRAJECTORY_CONFLICT_OBS_SIZE
    elif args.obs_size is None:
        args.obs_size = BASE_OBS_SIZE
    if (
        args.use_trajectory_conflict_obs
        and not args.use_action_conflict_obs
        and args.obs_size != TRAJECTORY_CONFLICT_OBS_SIZE
    ):
        raise ValueError(
            "--use-trajectory-conflict-obs expects --obs-size "
            f"{TRAJECTORY_CONFLICT_OBS_SIZE}, got {args.obs_size}."
        )
    if args.use_action_conflict_obs and args.obs_size != ACTION_CONFLICT_OBS_SIZE:
        raise ValueError(
            "--use-action-conflict-obs expects --obs-size "
            f"{ACTION_CONFLICT_OBS_SIZE}, got {args.obs_size}."
        )
    args.anchor_seed_list = parse_seed_list(args.anchor_seeds)
    return args


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train a checkpoint-compatible ActorCritic candidate from positive "
            "counterfactual rescue actions plus low-weight baseline anchors."
        )
    )
    parser.add_argument("csv", nargs="+", type=Path)
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--baseline-policy", default=DEFAULT_BASELINE_POLICY)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--init-checkpoint", type=Path, default=repo_root() / "submission/checkpoint.pt")
    parser.add_argument("--output-checkpoint", type=Path, default=Path("/private/tmp/ecml_rescue_bc.pt"))
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--obs-size", type=int)
    parser.add_argument("--n-actions", type=int, default=ACTION_COUNT)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--use-trajectory-conflict-obs", action="store_true")
    parser.add_argument(
        "--use-action-conflict-obs",
        action="store_true",
        help=(
            "Use MyActionConflictObservationBuilder and obs_size=79. "
            "Includes trajectory-conflict plus per-action conflict features."
        ),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--anchor-seeds")
    parser.add_argument("--anchor-stride", type=int, default=4)
    parser.add_argument("--anchor-weight", type=float, default=0.2)
    parser.add_argument("--reward-rescue-weight", type=float, default=6.0)
    parser.add_argument("--success-rescue-weight", type=float, default=12.0)
    parser.add_argument(
        "--include-avoidance-events",
        action="store_true",
        help=(
            "Train rows with event_kind=negative_baseline as baseline-action "
            "avoidance labels."
        ),
    )
    parser.add_argument("--avoidance-weight", type=float, default=6.0)
    parser.add_argument("--reward-epsilon", type=float, default=1e-6)
    parser.add_argument("--success-weight", type=float, default=2.0)
    parser.add_argument("--failed-weight", type=float, default=0.75)
    return parser.parse_args()


def main() -> int:
    args = finalize_args(parse_args())
    torch.manual_seed(17)
    np.random.seed(17)

    rescue_events = read_rescue_events(args.csv, args)
    rescue_count = sum(len(events) for events in rescue_events.values())
    print(
        "rescue_bc_config "
        f"obs_builder={args.obs_builder} obs_size={args.obs_size} "
        f"rescue_seeds={len(rescue_events)} rescue_events={rescue_count} "
        f"anchor_seeds={','.join(str(seed) for seed in args.anchor_seed_list)}",
        flush=True,
    )
    observations, actions, weights, collection_stats = collect_training_samples(
        args,
        rescue_events,
    )
    print(
        "collection "
        f"samples={collection_stats['samples']} "
        f"rescue_hits={collection_stats['rescue_hits']} "
        f"avoidance_hits={collection_stats['avoidance_hits']} "
        f"rescue_misses={collection_stats['rescue_misses']} "
        f"rescue_invalid={collection_stats['rescue_invalid']} "
        f"baseline_mismatches={collection_stats['rescue_baseline_mismatches']} "
        f"anchor_samples={collection_stats['anchor_samples']} "
        f"rescue_action_counts={collection_stats['rescue_action_counts']} "
        f"anchor_action_counts={collection_stats['anchor_action_counts']}",
        flush=True,
    )
    policy, train_stats = train_policy(args, observations, actions, weights)

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
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(
                {
                    "config": vars(args),
                    "collection_stats": collection_stats,
                    "train_stats": train_stats,
                },
                handle,
                indent=2,
                default=json_default,
            )
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
