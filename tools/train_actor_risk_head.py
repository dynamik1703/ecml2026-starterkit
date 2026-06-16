#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from submission.my_policy import ActorCritic
from tools.evaluate_rollout_mc_critic import average_precision, roc_auc, top_fraction_metrics
from tools.train_masked_ppo import load_aux_risk_dataset


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def selected_risk_scores(
    policy: ActorCritic,
    observations: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    scores = []
    with torch.no_grad():
        for start in range(0, observations.shape[0], batch_size):
            batch = observations[start : start + batch_size]
            action_batch = actions[start : start + batch_size]
            logits = policy.risk_logits(batch).gather(
                1,
                action_batch.unsqueeze(1),
            ).squeeze(1)
            scores.append(torch.sigmoid(logits).cpu())
    return torch.cat(scores).numpy()


def evaluate_risk_head(
    policy: ActorCritic,
    dataset: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, Any]:
    labels = dataset["labels"].cpu().numpy()
    scores = selected_risk_scores(
        policy,
        dataset["observations"],
        dataset["actions"],
        batch_size,
    )
    return {
        "rows": int(labels.shape[0]),
        "positive": int(labels.sum()),
        "auc": roc_auc(labels, scores),
        "average_precision": average_precision(labels, scores),
        "score_mean": float(scores.mean()),
        "top_fraction_metrics": top_fraction_metrics(labels, scores, [0.01, 0.05, 0.1, 0.2]),
    }


def train_risk_head(
    policy: ActorCritic,
    dataset: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.risk_head.parameters():
        parameter.requires_grad_(True)
    policy.train()

    optimizer = torch.optim.Adam(
        policy.risk_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    observations = dataset["observations"]
    actions = dataset["actions"]
    labels = dataset["labels"]
    weights = dataset["weights"]
    history = []
    for epoch in range(args.epochs):
        indices = torch.randperm(actions.shape[0])
        losses = []
        positive_fractions = []
        for start in range(0, actions.shape[0], args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            logits = policy.risk_logits(observations[batch_idx]).gather(
                1,
                actions[batch_idx].unsqueeze(1),
            ).squeeze(1)
            loss_values = F.binary_cross_entropy_with_logits(
                logits,
                labels[batch_idx],
                reduction="none",
            )
            weight_batch = weights[batch_idx]
            loss = (loss_values * weight_batch).sum() / weight_batch.sum().clamp_min(1e-6)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
            positive_fractions.append(float(labels[batch_idx].mean().item()))
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": float(np.mean(losses)),
                "positive_fraction": float(np.mean(positive_fractions)),
            }
        )
        print(
            f"epoch={epoch + 1}/{args.epochs} "
            f"loss={history[-1]['loss']:.6g} "
            f"positive_fraction={history[-1]['positive_fraction']:.6g}",
            flush=True,
        )
    policy.eval()
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train only ActorCritic.risk_head from rollout-MC labels."
    )
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--train-csv", nargs="+", required=True, type=Path)
    parser.add_argument("--validation-csv", nargs="+", required=True, type=Path)
    parser.add_argument("--obs-size", type=int, default=79)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--target", choices=["agent_failure", "team_failure"], default="agent_failure")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--positive-weight", type=float, default=4.0)
    parser.add_argument("--negative-weight", type=float, default=1.0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def risk_loader_args(
    args: argparse.Namespace,
    csv_paths: list[Path],
    max_samples: int,
) -> argparse.Namespace:
    return argparse.Namespace(
        aux_risk_coef=1.0,
        aux_risk_cache=None,
        aux_risk_refresh_cache=False,
        aux_risk_csv=csv_paths,
        aux_risk_target=args.target,
        aux_risk_positive_weight=args.positive_weight,
        aux_risk_negative_weight=args.negative_weight,
        aux_risk_max_samples=max_samples,
        seed=args.seed,
        n_actions=args.n_actions,
        obs_size=args.obs_size,
    )


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data, train_stats = load_aux_risk_dataset(
        risk_loader_args(args, args.train_csv, args.max_train_samples)
    )
    validation_data, validation_stats = load_aux_risk_dataset(
        risk_loader_args(args, args.validation_csv, 0)
    )
    assert train_data is not None and validation_data is not None

    policy = ActorCritic(
        obs_size=args.obs_size,
        n_actions=args.n_actions,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        checkpoint_path=str(args.init_checkpoint),
    )
    before = evaluate_risk_head(policy, validation_data, args.batch_size)
    print(
        "before "
        f"auc={before['auc']:.6g} "
        f"ap={before['average_precision']:.6g} "
        f"score_mean={before['score_mean']:.6g}",
        flush=True,
    )
    history = train_risk_head(policy, train_data, args)
    after = evaluate_risk_head(policy, validation_data, args.batch_size)
    print(
        "after "
        f"auc={after['auc']:.6g} "
        f"ap={after['average_precision']:.6g} "
        f"score_mean={after['score_mean']:.6g}",
        flush=True,
    )

    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": policy.state_dict(),
            "config": {
                "obs_size": args.obs_size,
                "n_actions": args.n_actions,
                "hidden_size": args.hidden_size,
                "num_hidden_layers": args.num_hidden_layers,
                "risk_head_trained": True,
                "risk_target": args.target,
            },
            "risk_train_stats": train_stats,
            "risk_validation_stats": validation_stats,
            "risk_validation_before": before,
            "risk_validation_after": after,
            "risk_training_history": history,
        },
        args.output_checkpoint,
    )
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(
                {
                    "config": vars(args),
                    "train_stats": train_stats,
                    "validation_stats": validation_stats,
                    "before": before,
                    "after": after,
                    "history": history,
                },
                handle,
                indent=2,
                default=json_default,
            )
            handle.write("\n")
    print(f"saved_checkpoint={args.output_checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
