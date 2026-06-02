#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rewards import ECML2026Rewards

from submission.my_observation_builder import MyObservationBuilder
from submission.my_policy import ActorCritic


DEFAULT_BASE_STATE = "reinforcement-learning/sampling/level_0_scenario_1.pkl"


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


def make_env(args: argparse.Namespace, seed: int) -> tuple[Any, dict[int, Any]]:
    obs_builder = MyObservationBuilder()
    rewards = ECML2026Rewards()
    env, _ = RailEnvPersister.load_new(
        args.base_state_pkl,
        obs_builder=obs_builder,
        rewards=rewards,
    )
    env.number_of_agents = args.num_agents
    env = load_sampling_env_generator()(env, line_length=args.line_length, scene=args.scene)
    observations, _ = env.reset(random_seed=seed)
    return env, observations


def observation_batch(observations: dict[int, Any], handles: list[int]) -> np.ndarray:
    return np.asarray([observations[handle] for handle in handles], dtype=np.float32)


def bootstrap_values(policy: ActorCritic, observations: dict[int, Any], handles: list[int]) -> torch.Tensor:
    obs = torch.as_tensor(observation_batch(observations, handles), dtype=torch.float32)
    with torch.no_grad():
        _, values = policy.masked_forward(obs)
    return values


def collect_rollout(
    args: argparse.Namespace,
    policy: ActorCritic,
    start_seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    env, observations = make_env(args, start_seed)
    handles = list(env.get_agent_handles())
    num_agents = len(handles)

    obs_steps: list[np.ndarray] = []
    action_steps: list[torch.Tensor] = []
    log_prob_steps: list[torch.Tensor] = []
    value_steps: list[torch.Tensor] = []
    reward_steps: list[list[float]] = []
    done_steps: list[list[float]] = []
    completed_episodes = 0
    completed_success = 0.0
    episode_rewards: list[float] = []
    episode_reward_values: list[float] = []

    for step in range(args.steps_per_update):
        obs_np = observation_batch(observations, handles)
        obs_t = torch.as_tensor(obs_np, dtype=torch.float32)
        with torch.no_grad():
            actions, log_probs, _, values = policy.sample_actions(obs_t)

        action_dict = {
            handle: int(action)
            for handle, action in zip(handles, actions.cpu().numpy())
        }
        next_observations, rewards_by_agent, dones, _ = env.step(action_dict)
        rewards = [float(rewards_by_agent.get(handle, 0.0)) for handle in handles]
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

        if bool(dones.get("__all__", False)) or env._elapsed_steps >= env._max_episode_steps:
            completed_episodes += 1
            completed_success += (
                sum(int(agent.state == 6) for agent in env.agents) / env.get_num_agents()
            )
            episode_rewards.append(sum(episode_reward_values) / max(1, num_agents))
            episode_reward_values = []
            env, observations = make_env(args, start_seed + completed_episodes)
            handles = list(env.get_agent_handles())
        else:
            observations = next_observations

        if step == args.steps_per_update - 1:
            next_values = bootstrap_values(policy, observations, handles).cpu()

    rewards_t = torch.as_tensor(np.asarray(reward_steps, dtype=np.float32))
    dones_t = torch.as_tensor(np.asarray(done_steps, dtype=np.float32))
    values_t = torch.stack(value_steps)
    next_values_t = torch.zeros_like(values_t)
    next_values_t[:-1] = values_t[1:]
    next_values_t[-1] = next_values

    advantages = torch.zeros_like(rewards_t)
    last_advantage = torch.zeros(num_agents)
    for index in reversed(range(args.steps_per_update)):
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
    stats = {
        "completed_episodes": float(completed_episodes),
        "success_rate": (
            completed_success / completed_episodes if completed_episodes else float("nan")
        ),
        "reward_mean": float(rewards_t.mean().item()),
        "episode_reward_mean": (
            float(np.mean(episode_rewards)) if episode_rewards else float("nan")
        ),
    }
    return rollout, stats


def ppo_update(
    args: argparse.Namespace,
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout: dict[str, torch.Tensor],
) -> dict[str, float]:
    observations = rollout["observations"]
    actions = rollout["actions"]
    old_log_probs = rollout["old_log_probs"]
    advantages = rollout["advantages"]
    returns = rollout["returns"]
    batch_size = observations.shape[0]
    indices = torch.randperm(batch_size)

    policy_losses = []
    value_losses = []
    entropies = []
    for _ in range(args.ppo_epochs):
        for start in range(0, batch_size, args.minibatch_size):
            batch_idx = indices[start : start + args.minibatch_size]
            log_probs, entropy, values = policy.evaluate_actions(
                observations[batch_idx],
                actions[batch_idx],
            )
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
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--obs-size", type=int, default=36)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--minibatch-size", type=int, default=256)
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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

    for update in range(args.updates):
        rollout, rollout_stats = collect_rollout(args, policy, args.seed + update * 1000)
        loss_stats = ppo_update(args, policy, optimizer, rollout)
        print(
            f"update={update + 1}/{args.updates} "
            f"reward_mean={rollout_stats['reward_mean']:.6g} "
            f"episode_reward_mean={rollout_stats['episode_reward_mean']:.6g} "
            f"success_rate={rollout_stats['success_rate']:.6g} "
            f"policy_loss={loss_stats['policy_loss']:.6g} "
            f"value_loss={loss_stats['value_loss']:.6g} "
            f"entropy={loss_stats['entropy']:.6g}",
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
