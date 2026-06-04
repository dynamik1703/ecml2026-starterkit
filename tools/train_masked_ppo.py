#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rewards import ECML2026Rewards
from torch.distributions import Categorical

from submission.my_policy import ActorCritic


DEFAULT_BASE_STATE = "reinforcement-learning/sampling/level_0_scenario_1.pkl"
DEFAULT_OBS_BUILDER = "submission.my_observation_builder.MyObservationBuilder"
DEFAULT_ROUTE_CONFLICT_OBS_BUILDER = (
    "submission.my_observation_builder.MyRouteConflictObservationBuilder"
)
DEFAULT_TRAJECTORY_CONFLICT_OBS_BUILDER = (
    "submission.my_observation_builder.MyTrajectoryConflictObservationBuilder"
)
DEFAULT_TEACHER_POLICY = "submission.rerank_policy.MyPolicy"
BASE_OBS_SIZE = 36
ROUTE_CONFLICT_OBS_SIZE = 52
TRAJECTORY_CONFLICT_OBS_SIZE = 64
DISTANCE_FEATURE_INDEX = 30
MOVE_FORWARD_ACTION = 2
ROUTE_CONFLICT_DISTANCE_INDEX = 36
ROUTE_CONFLICT_OPPOSING_INDEX = 37
ROUTE_CONFLICT_OTHER_TIGHTER_INDEX = 40
FUTURE_CONFLICT_ETA_RISK_INDEX = 46
FUTURE_CONFLICT_HEAD_ON_INDEX = 48
FUTURE_CONFLICT_CROSSING_INDEX = 49
FUTURE_CONFLICT_OTHER_TIGHTER_INDEX = 50


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_sampling_env_generator() -> Any:
    path = repo_root() / "reinforcement-learning" / "sampling" / "sampling_env_generator.py"
    spec = importlib.util.spec_from_file_location("ecml_sampling_env_generator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sampling generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sampling_env_generator


def load_symbol(path: str) -> Any:
    module_name, symbol_name = path.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, symbol_name)


def make_env(args: argparse.Namespace, seed: int) -> tuple[Any, dict[int, Any], Any]:
    obs_builder = load_symbol(args.obs_builder)()
    rewards = ECML2026Rewards()
    env, _ = RailEnvPersister.load_new(
        args.base_state_pkl,
        obs_builder=obs_builder,
        rewards=rewards,
    )
    env.number_of_agents = args.num_agents
    env = load_sampling_env_generator()(env, line_length=args.line_length, scene=args.scene)
    observations, _ = env.reset(random_seed=seed)
    return env, observations, obs_builder


def observation_batch(observations: dict[int, Any], handles: list[int]) -> np.ndarray:
    return np.asarray([observations[handle] for handle in handles], dtype=np.float32)


def observation_list(observations: dict[int, Any], handles: list[int]) -> list[Any]:
    return [observations[handle] for handle in handles]


def action_id(action: Any) -> int:
    if hasattr(action, "value"):
        return int(action.value)
    return int(action)


def slack_values(obs_builder: Any, handles: list[int]) -> np.ndarray:
    slacks = []
    for handle in handles:
        try:
            distance = obs_builder._current_distance_to_waypoint(handle)
            slack = obs_builder._deadline_slack(handle, distance)
        except Exception:
            slack = np.inf
        slacks.append(float(slack) if np.isfinite(slack) else np.inf)
    return np.asarray(slacks, dtype=np.float32)


def finite_delta(
    before: np.ndarray,
    after: np.ndarray,
    normalizer: float,
) -> np.ndarray:
    valid = np.isfinite(before) & np.isfinite(after)
    delta = np.zeros_like(before, dtype=np.float32)
    delta[valid] = (after[valid] - before[valid]) / max(1.0, normalizer)
    return np.clip(delta, -1.0, 1.0)


def negative_slack(slacks: np.ndarray) -> np.ndarray:
    finite = np.where(np.isfinite(slacks), slacks, 0.0)
    return np.maximum(-finite, 0.0)


def conflict_priority_penalty(
    args: argparse.Namespace,
    observations: np.ndarray,
    actions: np.ndarray,
) -> np.ndarray:
    if (
        args.conflict_priority_penalty_coef == 0.0
        or args.obs_size <= FUTURE_CONFLICT_OTHER_TIGHTER_INDEX
    ):
        return np.zeros(observations.shape[0], dtype=np.float32)

    threshold = max(1e-6, args.conflict_priority_distance_threshold)
    route_distance = observations[:, ROUTE_CONFLICT_DISTANCE_INDEX]
    route_closeness = np.clip((threshold - route_distance) / threshold, 0.0, 1.0)
    route_risk = (
        route_closeness
        * observations[:, ROUTE_CONFLICT_OPPOSING_INDEX]
        * observations[:, ROUTE_CONFLICT_OTHER_TIGHTER_INDEX]
    )

    future_conflict_type = np.maximum(
        observations[:, FUTURE_CONFLICT_HEAD_ON_INDEX],
        observations[:, FUTURE_CONFLICT_CROSSING_INDEX],
    )
    future_risk = (
        observations[:, FUTURE_CONFLICT_ETA_RISK_INDEX]
        * future_conflict_type
        * observations[:, FUTURE_CONFLICT_OTHER_TIGHTER_INDEX]
    )

    risky_forward = (actions == MOVE_FORWARD_ACTION).astype(np.float32)
    risk = np.maximum(route_risk, future_risk)
    return -args.conflict_priority_penalty_coef * risky_forward * risk


def shaped_rewards(
    args: argparse.Namespace,
    observations: np.ndarray,
    next_observations: np.ndarray,
    rewards: list[float],
    actions: np.ndarray,
    current_slacks: np.ndarray,
    next_slacks: np.ndarray,
    max_episode_steps: int,
) -> list[float]:
    shaped = np.asarray(rewards, dtype=np.float32).copy()
    if args.progress_reward_coef != 0.0:
        progress = observations[:, DISTANCE_FEATURE_INDEX] - next_observations[:, DISTANCE_FEATURE_INDEX]
        shaped += args.progress_reward_coef * progress
    if args.slack_progress_reward_coef != 0.0:
        slack_progress = finite_delta(
            current_slacks,
            next_slacks,
            max_episode_steps,
        )
        shaped += args.slack_progress_reward_coef * slack_progress
    if args.global_slack_reward_coef != 0.0:
        before_lateness = float(np.sum(negative_slack(current_slacks)))
        after_lateness = float(np.sum(negative_slack(next_slacks)))
        team_delta = np.clip(
            (before_lateness - after_lateness)
            / max(1.0, max_episode_steps * len(current_slacks)),
            -1.0,
            1.0,
        )
        shaped += args.global_slack_reward_coef * team_delta
    shaped += conflict_priority_penalty(args, observations, actions)
    if args.reward_scale != 1.0:
        shaped *= args.reward_scale
    if args.reward_clip > 0.0:
        shaped = np.clip(shaped, -args.reward_clip, args.reward_clip)
    return shaped.astype(np.float32).tolist()


def agent_succeeded(agent: Any) -> bool:
    return bool(agent.state == 6 or getattr(agent.state, "name", "") == "DONE")


def terminal_outcome_rewards(
    args: argparse.Namespace,
    env: Any,
    handles: list[int],
    episode_done: bool,
) -> list[float]:
    bonuses = np.zeros(len(handles), dtype=np.float32)
    if not episode_done:
        return bonuses.tolist()

    successes = np.asarray(
        [float(agent_succeeded(env.agents[handle])) for handle in handles],
        dtype=np.float32,
    )
    failures = 1.0 - successes
    if args.terminal_success_bonus != 0.0:
        bonuses += args.terminal_success_bonus * successes
    if args.terminal_failure_penalty != 0.0:
        bonuses -= args.terminal_failure_penalty * failures
    if args.terminal_team_success_bonus != 0.0:
        bonuses += args.terminal_team_success_bonus * float(successes.mean())
    if args.terminal_team_failure_penalty != 0.0:
        bonuses -= args.terminal_team_failure_penalty * float(failures.mean())
    return bonuses.tolist()


def bootstrap_values(policy: ActorCritic, observations: dict[int, Any], handles: list[int]) -> torch.Tensor:
    obs = torch.as_tensor(observation_batch(observations, handles), dtype=torch.float32)
    with torch.no_grad():
        _, values = policy.masked_forward(obs)
    return values


def action_distribution(
    policy: ActorCritic,
    observations: torch.Tensor,
    temperature: float,
) -> tuple[Categorical, torch.Tensor]:
    logits, values = policy.masked_forward(observations)
    return Categorical(logits=logits / max(1e-6, temperature)), values


def anchor_policy_kl(
    policy: ActorCritic,
    anchor_policy: ActorCritic,
    observations: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    current_logits, _ = policy.masked_forward(observations)
    with torch.no_grad():
        anchor_logits, _ = anchor_policy.masked_forward(observations)

    current_logits = current_logits / max(1e-6, temperature)
    anchor_logits = anchor_logits / max(1e-6, temperature)
    valid_actions = torch.isfinite(current_logits) & torch.isfinite(anchor_logits)
    current_logits = current_logits.masked_fill(~valid_actions, -1e9)
    anchor_logits = anchor_logits.masked_fill(~valid_actions, -1e9)
    current_log_probs = torch.log_softmax(current_logits, dim=-1)
    anchor_probs = torch.softmax(anchor_logits, dim=-1)
    return F.kl_div(current_log_probs, anchor_probs, reduction="batchmean")


def collect_rollout(
    args: argparse.Namespace,
    policy: ActorCritic,
    start_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    env, observations, obs_builder = make_env(args, start_seed)
    handles = list(env.get_agent_handles())
    num_agents = len(handles)
    teacher_policy = (
        load_symbol(args.teacher_policy)()
        if args.teacher_ce_coef != 0.0
        else None
    )

    obs_steps: list[np.ndarray] = []
    action_steps: list[torch.Tensor] = []
    teacher_action_steps: list[torch.Tensor] = []
    log_prob_steps: list[torch.Tensor] = []
    value_steps: list[torch.Tensor] = []
    reward_steps: list[list[float]] = []
    done_steps: list[list[float]] = []
    completed_episodes = 0
    completed_success = 0.0
    episode_rewards: list[float] = []
    episode_reward_values: list[float] = []

    step = 0
    while True:
        if args.episodes_per_update > 0:
            if completed_episodes >= args.episodes_per_update and step > 0:
                break
        elif step >= args.steps_per_update:
            break

        obs_np = observation_batch(observations, handles)
        current_slacks = slack_values(obs_builder, handles)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
        with torch.no_grad():
            distribution, values = action_distribution(
                policy,
                obs_t,
                args.rollout_temperature,
            )
            actions = distribution.sample()
            log_probs = distribution.log_prob(actions)

        actions_np = actions.cpu().numpy()
        action_dict = {
            handle: int(action)
            for handle, action in zip(handles, actions_np)
        }
        if teacher_policy is not None:
            teacher_actions = teacher_policy.act_many(
                handles,
                observation_list(observations, handles),
            )
            teacher_action_steps.append(
                torch.as_tensor(
                    [
                        action_id(teacher_actions.get(handle, action_dict[handle]))
                        for handle in handles
                    ],
                    dtype=torch.long,
                )
            )
        next_observations, rewards_by_agent, dones, _ = env.step(action_dict)
        next_obs_np = observation_batch(next_observations, handles)
        next_slacks = slack_values(obs_builder, handles)
        rewards = [float(rewards_by_agent.get(handle, 0.0)) for handle in handles]
        rewards = shaped_rewards(
            args,
            obs_np,
            next_obs_np,
            rewards,
            actions_np,
            current_slacks,
            next_slacks,
            env._max_episode_steps,
        )
        episode_done = (
            bool(dones.get("__all__", False))
            or env._elapsed_steps >= env._max_episode_steps
        )
        terminal_rewards = terminal_outcome_rewards(
            args,
            env,
            handles,
            episode_done,
        )
        rewards = [
            reward + terminal_reward
            for reward, terminal_reward in zip(rewards, terminal_rewards)
        ]
        if args.reward_clip > 0.0:
            rewards = np.clip(
                np.asarray(rewards, dtype=np.float32),
                -args.reward_clip,
                args.reward_clip,
            ).astype(np.float32).tolist()
        done_flags = [
            float(bool(dones.get(handle, False) or dones.get("__all__", False)))
            for handle in handles
        ]

        obs_steps.append(obs_np)
        action_steps.append(actions.cpu())
        log_prob_steps.append(log_probs.cpu())
        value_steps.append(values.cpu())
        reward_steps.append(rewards)
        done_steps.append(done_flags)
        episode_reward_values.extend(rewards)

        if episode_done:
            completed_episodes += 1
            completed_success += (
                sum(int(agent_succeeded(agent)) for agent in env.agents)
                / env.get_num_agents()
            )
            episode_rewards.append(sum(episode_reward_values) / max(1, num_agents))
            episode_reward_values = []
            env, observations, obs_builder = make_env(
                args,
                start_seed + completed_episodes,
            )
            handles = list(env.get_agent_handles())
        else:
            observations = next_observations

        if step == args.steps_per_update - 1:
            next_values = bootstrap_values(policy, observations, handles).cpu()
        step += 1

    if args.episodes_per_update > 0:
        next_values = bootstrap_values(policy, observations, handles).cpu()

    rewards_t = torch.as_tensor(np.asarray(reward_steps, dtype=np.float32))
    dones_t = torch.as_tensor(np.asarray(done_steps, dtype=np.float32))
    values_t = torch.stack(value_steps)
    next_values_t = torch.zeros_like(values_t)
    next_values_t[:-1] = values_t[1:]
    next_values_t[-1] = next_values

    advantages = torch.zeros_like(rewards_t)
    last_advantage = torch.zeros(num_agents)
    for index in reversed(range(len(reward_steps))):
        not_done = 1.0 - dones_t[index]
        delta = rewards_t[index] + args.gamma * next_values_t[index] * not_done - values_t[index]
        last_advantage = delta + args.gamma * args.gae_lambda * not_done * last_advantage
        advantages[index] = last_advantage

    returns = advantages + values_t
    flat_advantages = advantages.reshape(-1)
    if args.normalize_advantages and flat_advantages.numel() > 1:
        flat_advantages = (
            flat_advantages - flat_advantages.mean()
        ) / (flat_advantages.std(unbiased=False) + 1e-8)

    obs_array = np.asarray(obs_steps, dtype=np.float32)
    rollout = {
        "observations": torch.as_tensor(obs_array).reshape(-1, obs_array.shape[-1]),
        "actions": torch.stack(action_steps).reshape(-1),
        "old_log_probs": torch.stack(log_prob_steps).reshape(-1),
        "advantages": flat_advantages,
        "returns": returns.reshape(-1),
    }
    if teacher_action_steps:
        rollout["teacher_actions"] = torch.stack(teacher_action_steps).reshape(-1)
    stats = {
        "completed_episodes": float(completed_episodes),
        "success_rate": (
            completed_success / completed_episodes if completed_episodes else float("nan")
        ),
        "reward_mean": float(rewards_t.mean().item()),
        "episode_reward_mean": (
            float(np.mean(episode_rewards)) if episode_rewards else float("nan")
        ),
        "collected_steps": float(len(reward_steps)),
    }
    return rollout, stats


def ppo_update(
    args: argparse.Namespace,
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor],
    anchor_policy: ActorCritic | None,
) -> dict[str, float]:
    observations = rollout["observations"]
    actions = rollout["actions"]
    old_log_probs = rollout["old_log_probs"]
    advantages = rollout["advantages"]
    returns = rollout["returns"]
    teacher_actions = rollout.get("teacher_actions")
    batch_size = observations.shape[0]
    indices = torch.randperm(batch_size)

    policy_losses = []
    value_losses = []
    entropies = []
    teacher_losses = []
    teacher_valid_fractions = []
    anchor_kls = []
    for _ in range(args.ppo_epochs):
        for start in range(0, batch_size, args.minibatch_size):
            batch_idx = indices[start : start + args.minibatch_size]
            distribution, values = action_distribution(
                policy,
                observations[batch_idx],
                args.rollout_temperature,
            )
            log_probs = distribution.log_prob(actions[batch_idx])
            entropy = distribution.entropy()
            ratio = torch.exp(log_probs - old_log_probs[batch_idx])
            unclipped = ratio * advantages[batch_idx]
            clipped = torch.clamp(
                ratio,
                1.0 - args.clip_coef,
                1.0 + args.clip_coef,
            ) * advantages[batch_idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = F.mse_loss(values, returns[batch_idx])
            entropy_loss = entropy.mean()
            loss = policy_loss + args.vf_coef * value_loss - args.ent_coef * entropy_loss
            if teacher_actions is not None and args.teacher_ce_coef != 0.0:
                teacher_logits, _ = policy.masked_forward(observations[batch_idx])
                teacher_action_batch = teacher_actions[batch_idx]
                teacher_action_logits = teacher_logits.gather(
                    1,
                    teacher_action_batch.unsqueeze(1),
                ).squeeze(1)
                valid_teacher = torch.isfinite(teacher_action_logits)
                teacher_valid_fractions.append(float(valid_teacher.float().mean().item()))
                if not valid_teacher.any():
                    teacher_loss = teacher_logits.new_tensor(0.0)
                else:
                    teacher_loss = F.cross_entropy(
                        teacher_logits[valid_teacher],
                        teacher_action_batch[valid_teacher],
                    )
                loss = loss + args.teacher_ce_coef * teacher_loss
                teacher_losses.append(float(teacher_loss.item()))

            if anchor_policy is not None and args.anchor_kl_coef != 0.0:
                anchor_kl = anchor_policy_kl(
                    policy,
                    anchor_policy,
                    observations[batch_idx],
                    args.rollout_temperature,
                )
                loss = loss + args.anchor_kl_coef * anchor_kl
                anchor_kls.append(float(anchor_kl.item()))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm)
            optimizer.step()

            policy_losses.append(float(policy_loss.item()))
            value_losses.append(float(value_loss.item()))
            entropies.append(float(entropy_loss.item()))

    return {
        "policy_loss": float(np.mean(policy_losses)),
        "value_loss": float(np.mean(value_losses)),
        "entropy": float(np.mean(entropies)),
        "teacher_loss": (
            float(np.mean(teacher_losses)) if teacher_losses else float("nan")
        ),
        "teacher_valid_fraction": (
            float(np.mean(teacher_valid_fractions))
            if teacher_valid_fractions
            else float("nan")
        ),
        "anchor_kl": (
            float(np.mean(anchor_kls)) if anchor_kls else float("nan")
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the masked ActorCritic policy with PPO.")
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--init-checkpoint", type=Path, default=repo_root() / "submission/checkpoint.pt")
    parser.add_argument("--output-checkpoint", type=Path, default=Path("/private/tmp/ecml_masked_ppo.pt"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--updates", type=int, default=10)
    parser.add_argument("--steps-per-update", type=int, default=128)
    parser.add_argument(
        "--episodes-per-update",
        type=int,
        default=0,
        help="Collect complete episodes per PPO update. Disabled when 0.",
    )
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument(
        "--use-route-conflict-obs",
        action="store_true",
        help=(
            "Use MyRouteConflictObservationBuilder and obs_size=52. "
            "Required for the route/future conflict reward terms."
        ),
    )
    parser.add_argument(
        "--use-trajectory-conflict-obs",
        action="store_true",
        help=(
            "Use MyTrajectoryConflictObservationBuilder and obs_size=64. "
            "Includes route-conflict plus trajectory/priority features."
        ),
    )
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--obs-size", type=int)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--progress-reward-coef",
        type=float,
        default=0.0,
        help="Optional dense reward for reducing observation[30], the normalized waypoint distance.",
    )
    parser.add_argument(
        "--slack-progress-reward-coef",
        type=float,
        default=0.0,
        help="Reward preserving/improving true deadline slack per agent.",
    )
    parser.add_argument(
        "--global-slack-reward-coef",
        type=float,
        default=0.0,
        help="Team reward for reducing total negative deadline slack.",
    )
    parser.add_argument(
        "--conflict-priority-penalty-coef",
        type=float,
        default=0.0,
        help=(
            "Penalty for moving forward into route conflicts where another train "
            "has tighter slack. Requires the 52-feature route observation."
        ),
    )
    parser.add_argument(
        "--conflict-priority-distance-threshold",
        type=float,
        default=0.35,
        help="Normalized route distance threshold for the conflict-priority penalty.",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=1.0,
        help="Multiply environment and progress rewards before PPO return estimation.",
    )
    parser.add_argument(
        "--reward-clip",
        type=float,
        default=0.0,
        help="Clip scaled rewards to [-reward_clip, reward_clip]. Disabled when 0.",
    )
    parser.add_argument(
        "--terminal-success-bonus",
        type=float,
        default=0.0,
        help="Per-agent bonus added at episode end for agents that reached DONE.",
    )
    parser.add_argument(
        "--terminal-failure-penalty",
        type=float,
        default=0.0,
        help="Per-agent penalty added at episode end for agents that did not reach DONE.",
    )
    parser.add_argument(
        "--terminal-team-success-bonus",
        type=float,
        default=0.0,
        help="Team bonus added to every agent at episode end, scaled by success rate.",
    )
    parser.add_argument(
        "--terminal-team-failure-penalty",
        type=float,
        default=0.0,
        help="Team penalty added to every agent at episode end, scaled by failure rate.",
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=1.0,
        help="Temperature applied to masked policy logits during PPO collection and updates.",
    )
    parser.add_argument("--teacher-policy", default=DEFAULT_TEACHER_POLICY)
    parser.add_argument(
        "--teacher-ce-coef",
        type=float,
        default=0.0,
        help="Cross-entropy weight for keeping PPO close to a teacher policy.",
    )
    parser.add_argument(
        "--anchor-kl-coef",
        type=float,
        default=0.0,
        help=(
            "KL penalty against the initial actor checkpoint. Useful when adding "
            "new observation features but keeping PPO close to the stable actor."
        ),
    )
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
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
        args.conflict_priority_penalty_coef != 0.0
        and args.obs_size <= FUTURE_CONFLICT_OTHER_TIGHTER_INDEX
    ):
        raise ValueError(
            "--conflict-priority-penalty-coef requires route-conflict features. "
            "Pass --use-route-conflict-obs or --use-trajectory-conflict-obs, "
            "or set --obs-builder "
            f"{DEFAULT_ROUTE_CONFLICT_OBS_BUILDER} --obs-size {ROUTE_CONFLICT_OBS_SIZE}."
        )

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
    if args.rollout_temperature <= 0:
        raise ValueError("--rollout-temperature must be positive")

    print(
        "training_config "
        f"obs_builder={args.obs_builder} "
        f"obs_size={args.obs_size} "
        f"num_agents={args.num_agents} "
        f"line_length={args.line_length} "
        f"updates={args.updates} "
        f"steps_per_update={args.steps_per_update} "
        f"episodes_per_update={args.episodes_per_update}",
        flush=True,
    )

    checkpoint_path = args.init_checkpoint if args.init_checkpoint.exists() else None
    policy = ActorCritic(
        obs_size=args.obs_size,
        n_actions=args.n_actions,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
    )
    policy.train()
    anchor_policy = None
    if args.anchor_kl_coef != 0.0:
        anchor_policy = ActorCritic(
            obs_size=args.obs_size,
            n_actions=args.n_actions,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            checkpoint_path=str(checkpoint_path) if checkpoint_path is not None else None,
        )
        anchor_policy.eval()
        for parameter in anchor_policy.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)

    for update in range(args.updates):
        rollout, rollout_stats = collect_rollout(args, policy, args.seed + update * 1000)
        loss_stats = ppo_update(args, policy, optimizer, rollout, anchor_policy)
        teacher_loss_text = (
            f" teacher_loss={loss_stats['teacher_loss']:.6g}"
            f" teacher_valid={loss_stats['teacher_valid_fraction']:.3g}"
            if not np.isnan(loss_stats["teacher_loss"])
            else ""
        )
        anchor_kl_text = (
            f" anchor_kl={loss_stats['anchor_kl']:.6g}"
            if not np.isnan(loss_stats["anchor_kl"])
            else ""
        )
        print(
            f"update={update + 1}/{args.updates} "
            f"reward_mean={rollout_stats['reward_mean']:.6g} "
            f"episode_reward_mean={rollout_stats['episode_reward_mean']:.6g} "
            f"success_rate={rollout_stats['success_rate']:.6g} "
            f"collected_steps={rollout_stats['collected_steps']:.0f} "
            f"policy_loss={loss_stats['policy_loss']:.6g} "
            f"value_loss={loss_stats['value_loss']:.6g} "
            f"entropy={loss_stats['entropy']:.6g}"
            f"{teacher_loss_text}"
            f"{anchor_kl_text}",
            flush=True,
        )

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "config": vars(args),
        },
        args.output_checkpoint,
    )
    print(f"saved_checkpoint={args.output_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
