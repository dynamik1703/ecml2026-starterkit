#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.dispatch_ranker import (
    DISPATCH_PAIR_FEATURE_DIM,
    DISPATCH_PAIR_FEATURE_NAMES,
    DispatchPairwiseMLP,
)


def load_rows(path: Path) -> tuple[list[list[float]], list[float], list[int]]:
    features: list[list[float]] = []
    labels: list[float] = []
    seeds: list[int] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in DISPATCH_PAIR_FEATURE_NAMES if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        for row in reader:
            vector = [float(row[name]) for name in DISPATCH_PAIR_FEATURE_NAMES]
            if len(vector) != DISPATCH_PAIR_FEATURE_DIM:
                raise ValueError("feature dimension mismatch")
            features.append(vector)
            labels.append(float(row["label"]))
            seeds.append(int(row.get("seed") or 0))
    if not features:
        raise ValueError(f"no rows found in {path}")
    return features, labels, seeds


def seed_split(seeds: list[int], validation_fraction: float, seed: int) -> set[int]:
    unique_seeds = sorted(set(seeds))
    rng = random.Random(seed)
    rng.shuffle(unique_seeds)
    validation_count = max(1, int(round(len(unique_seeds) * validation_fraction)))
    return set(unique_seeds[:validation_count])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the high-level pairwise dispatch ranker."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    features, labels, seeds = load_rows(args.input_csv)
    validation_seeds = seed_split(seeds, args.validation_fraction, args.seed)
    train_indices = [index for index, seed in enumerate(seeds) if seed not in validation_seeds]
    validation_indices = [index for index, seed in enumerate(seeds) if seed in validation_seeds]
    if not train_indices:
        train_indices = validation_indices

    x = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32).view(-1, 1)
    train_x = x[train_indices]
    train_y = y[train_indices]
    validation_x = x[validation_indices]
    validation_y = y[validation_indices]

    positives = float(train_y.sum().item())
    negatives = float(train_y.numel() - positives)
    pos_weight = torch.tensor([negatives / max(1.0, positives)], dtype=torch.float32)

    model = DispatchPairwiseMLP.build(
        input_dim=DISPATCH_PAIR_FEATURE_DIM,
        hidden_size=args.hidden_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(1, args.epochs + 1):
        permutation = torch.randperm(train_x.shape[0])
        model.train()
        total_loss = 0.0
        for start in range(0, train_x.shape[0], args.batch_size):
            batch_indices = permutation[start : start + args.batch_size]
            batch_x = train_x[batch_indices]
            batch_y = train_y[batch_indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * batch_x.shape[0]

        model.eval()
        with torch.no_grad():
            train_loss = total_loss / max(1, train_x.shape[0])
            if validation_x.numel() > 0:
                validation_logits = model(validation_x)
                validation_loss = float(criterion(validation_logits, validation_y).item())
                validation_accuracy = float(
                    ((torch.sigmoid(validation_logits) >= 0.5) == validation_y.bool())
                    .float()
                    .mean()
                    .item()
                )
            else:
                validation_loss = train_loss
                validation_accuracy = 0.0
        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_accuracy={validation_accuracy:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": DISPATCH_PAIR_FEATURE_DIM,
            "hidden_size": args.hidden_size,
            "feature_names": list(DISPATCH_PAIR_FEATURE_NAMES),
            "validation_seeds": sorted(validation_seeds),
        },
        args.output,
    )
    print(f"saved dispatch ranker checkpoint to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
