#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from submission.my_policy import ActorCritic


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def safe_float(value: Any) -> float:
    if value in (None, ""):
        return float("nan")
    try:
        result = float(value)
    except Exception:
        return float("nan")
    return result if np.isfinite(result) else float("nan")


def finite_or_zero(value: Any) -> float:
    result = safe_float(value)
    return result if np.isfinite(result) else 0.0


def obs_columns(rows: list[dict[str, Any]]) -> list[str]:
    def key(name: str) -> int:
        try:
            return int(name.split("_", maxsplit=1)[1])
        except Exception:
            return 10**9

    return sorted(
        {column for row in rows for column in row if column.startswith("obs_")},
        key=key,
    )


def read_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def load_dataset(
    paths: list[Path],
    *,
    n_actions: int,
    max_samples: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    rows = read_rows(paths)
    columns = obs_columns(rows)
    if not columns:
        raise ValueError("value-head training requires obs_* columns")

    observations = []
    actions = []
    targets = []
    skipped_invalid_action = 0
    skipped_invalid_mask = 0
    skipped_missing_target = 0
    for row in rows:
        action = safe_float(row.get("action"))
        if not np.isfinite(action) or int(action) != action or not 0 <= int(action) < n_actions:
            skipped_invalid_action += 1
            continue
        action_valid = safe_float(row.get("action_valid"))
        if np.isfinite(action_valid) and action_valid < 0.5:
            skipped_invalid_mask += 1
            continue
        target = safe_float(row.get("episode_normalized_reward"))
        if not np.isfinite(target):
            skipped_missing_target += 1
            continue
        observations.append([finite_or_zero(row.get(column)) for column in columns])
        actions.append(int(action))
        targets.append(float(np.clip(target, 0.0, 1.0)))

    if not observations:
        raise ValueError("value-head training produced no valid samples")

    observations_array = np.asarray(observations, dtype=np.float32)
    actions_array = np.asarray(actions, dtype=np.int64)
    targets_array = np.asarray(targets, dtype=np.float32)
    if max_samples > 0 and len(actions_array) > max_samples:
        rng = np.random.default_rng(seed + 2901)
        keep = rng.choice(len(actions_array), size=max_samples, replace=False)
        observations_array = observations_array[keep]
        actions_array = actions_array[keep]
        targets_array = targets_array[keep]

    dataset = {
        "observations": torch.as_tensor(observations_array, dtype=torch.float32),
        "actions": torch.as_tensor(actions_array, dtype=torch.long),
        "targets": torch.as_tensor(targets_array, dtype=torch.float32),
    }
    stats = {
        "samples": int(len(actions_array)),
        "source_rows": int(len(rows)),
        "obs_columns": int(len(columns)),
        "target_mean": float(targets_array.mean()),
        "target_std": float(targets_array.std()),
        "skipped_invalid_action": skipped_invalid_action,
        "skipped_invalid_mask": skipped_invalid_mask,
        "skipped_missing_target": skipped_missing_target,
    }
    return dataset, stats


def selected_values(
    policy: ActorCritic,
    observations: torch.Tensor,
    actions: torch.Tensor,
    batch_size: int,
) -> np.ndarray:
    values = []
    with torch.no_grad():
        for start in range(0, observations.shape[0], batch_size):
            batch = observations[start : start + batch_size]
            action_batch = actions[start : start + batch_size]
            predicted = policy.action_value_scores(batch).gather(
                1,
                action_batch.unsqueeze(1),
            ).squeeze(1)
            values.append(predicted.cpu())
    return torch.cat(values).numpy()


def evaluate_value_head(
    policy: ActorCritic,
    dataset: dict[str, torch.Tensor],
    batch_size: int,
) -> dict[str, Any]:
    targets = dataset["targets"].cpu().numpy()
    predictions = selected_values(
        policy,
        dataset["observations"],
        dataset["actions"],
        batch_size,
    )
    errors = predictions - targets
    corr = float(np.corrcoef(predictions, targets)[0, 1]) if len(targets) > 1 else float("nan")
    return {
        "rows": int(targets.shape[0]),
        "target_mean": float(targets.mean()),
        "prediction_mean": float(predictions.mean()),
        "mae": float(np.abs(errors).mean()),
        "rmse": float(np.sqrt(np.mean(errors * errors))),
        "corr": corr,
    }


def train_value_head(
    policy: ActorCritic,
    dataset: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> list[dict[str, float]]:
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    for parameter in policy.action_value_head.parameters():
        parameter.requires_grad_(True)
    policy.train()

    optimizer = torch.optim.Adam(
        policy.action_value_head.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    observations = dataset["observations"]
    actions = dataset["actions"]
    targets = dataset["targets"]
    history = []
    for epoch in range(args.epochs):
        indices = torch.randperm(actions.shape[0])
        losses = []
        for start in range(0, actions.shape[0], args.batch_size):
            batch_idx = indices[start : start + args.batch_size]
            predictions = policy.action_value_scores(observations[batch_idx]).gather(
                1,
                actions[batch_idx].unsqueeze(1),
            ).squeeze(1)
            loss = F.mse_loss(predictions, targets[batch_idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": float(epoch + 1), "loss": float(np.mean(losses))})
        print(
            f"epoch={epoch + 1}/{args.epochs} loss={history[-1]['loss']:.6g}",
            flush=True,
        )
    policy.eval()
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train only ActorCritic.action_value_head from rollout MC returns."
    )
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--train-csv", nargs="+", required=True, type=Path)
    parser.add_argument("--validation-csv", nargs="+", required=True, type=Path)
    parser.add_argument("--obs-size", type=int, default=79)
    parser.add_argument("--n-actions", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-hidden-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1901)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_data, train_stats = load_dataset(
        args.train_csv,
        n_actions=args.n_actions,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    validation_data, validation_stats = load_dataset(
        args.validation_csv,
        n_actions=args.n_actions,
        max_samples=0,
        seed=args.seed,
    )
    policy = ActorCritic(
        obs_size=args.obs_size,
        n_actions=args.n_actions,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        checkpoint_path=str(args.init_checkpoint),
    )
    before = evaluate_value_head(policy, validation_data, args.batch_size)
    print(
        f"before mae={before['mae']:.6g} rmse={before['rmse']:.6g} "
        f"corr={before['corr']:.6g}",
        flush=True,
    )
    history = train_value_head(policy, train_data, args)
    after = evaluate_value_head(policy, validation_data, args.batch_size)
    print(
        f"after mae={after['mae']:.6g} rmse={after['rmse']:.6g} "
        f"corr={after['corr']:.6g}",
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
                "action_value_head_trained": True,
            },
            "value_train_stats": train_stats,
            "value_validation_stats": validation_stats,
            "value_validation_before": before,
            "value_validation_after": after,
            "value_training_history": history,
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
