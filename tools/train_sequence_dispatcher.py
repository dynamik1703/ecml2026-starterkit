#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from submission.sequence_dispatcher import (
    SEQUENCE_DISPATCH_FEATURE_DIM,
    SEQUENCE_DISPATCH_FEATURE_NAMES,
    SequenceDispatchTransformer,
)


def load_examples(path: Path, max_candidates: int) -> list[dict[str, Any]]:
    examples = []
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            example = json.loads(line)
            features = example.get("features", [])
            scores = example.get("teacher_scores", [])
            if len(features) < 2 or len(features) != len(scores):
                continue
            if len(features) > max_candidates:
                features = features[:max_candidates]
                scores = scores[:max_candidates]
                example["features"] = features
                example["teacher_scores"] = scores
            for vector in features:
                if len(vector) != SEQUENCE_DISPATCH_FEATURE_DIM:
                    raise ValueError(
                        f"{path}:{line_number} feature dim {len(vector)} != "
                        f"{SEQUENCE_DISPATCH_FEATURE_DIM}"
                    )
            examples.append(example)
    if not examples:
        raise ValueError(f"no usable examples found in {path}")
    return examples


def split_by_seed(
    examples: list[dict[str, Any]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    seeds = sorted({int(example.get("seed") or 0) for example in examples})
    rng = random.Random(seed)
    rng.shuffle(seeds)
    validation_count = max(1, int(round(len(seeds) * validation_fraction)))
    validation_seeds = set(seeds[:validation_count])
    train = [
        example
        for example in examples
        if int(example.get("seed") or 0) not in validation_seeds
    ]
    validation = [
        example
        for example in examples
        if int(example.get("seed") or 0) in validation_seeds
    ]
    if not train:
        train = validation
    return train, validation


def batches(
    examples: list[dict[str, Any]],
    batch_size: int,
    rng: random.Random,
) -> list[list[dict[str, Any]]]:
    shuffled = list(examples)
    rng.shuffle(shuffled)
    return [
        shuffled[start : start + batch_size]
        for start in range(0, len(shuffled), batch_size)
    ]


def collate(batch: list[dict[str, Any]], device: Any) -> tuple[Any, Any, Any, list[int]]:
    import torch

    max_len = max(len(example["features"]) for example in batch)
    x = torch.zeros(
        (len(batch), max_len, SEQUENCE_DISPATCH_FEATURE_DIM),
        dtype=torch.float32,
        device=device,
    )
    teacher_scores = torch.full(
        (len(batch), max_len),
        -1.0e9,
        dtype=torch.float32,
        device=device,
    )
    padding_mask = torch.ones((len(batch), max_len), dtype=torch.bool, device=device)
    top_indices: list[int] = []
    for row, example in enumerate(batch):
        features = torch.tensor(example["features"], dtype=torch.float32, device=device)
        scores = torch.tensor(example["teacher_scores"], dtype=torch.float32, device=device)
        length = features.shape[0]
        x[row, :length] = features
        teacher_scores[row, :length] = scores
        padding_mask[row, :length] = False
        top_indices.append(int(scores.argmax().item()))
    return x, teacher_scores, padding_mask, top_indices


def listwise_loss(logits: Any, teacher_scores: Any, padding_mask: Any, temperature: float) -> Any:
    import torch

    masked_logits = logits.masked_fill(padding_mask, -1.0e9)
    target_distribution = torch.softmax(
        teacher_scores / max(1.0e-6, temperature),
        dim=1,
    )
    log_distribution = torch.log_softmax(masked_logits, dim=1)
    return -(target_distribution * log_distribution).sum(dim=1).mean()


def evaluate(model: Any, examples: list[dict[str, Any]], args: argparse.Namespace, device: Any) -> dict[str, float]:
    import torch

    if not examples:
        return {"loss": 0.0, "top1": 0.0}
    model.eval()
    total_loss = 0.0
    total_examples = 0
    top1 = 0
    with torch.no_grad():
        for batch in batches(examples, args.batch_size, random.Random(args.seed)):
            x, teacher_scores, padding_mask, top_indices = collate(batch, device)
            logits = model(x, padding_mask=padding_mask)
            loss = listwise_loss(logits, teacher_scores, padding_mask, args.temperature)
            total_loss += float(loss.item()) * len(batch)
            predictions = logits.masked_fill(padding_mask, -1.0e9).argmax(dim=1).cpu().tolist()
            top1 += sum(int(prediction == target) for prediction, target in zip(predictions, top_indices))
            total_examples += len(batch)
    return {
        "loss": total_loss / float(max(1, total_examples)),
        "top1": top1 / float(max(1, total_examples)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a transformer BC model for high-level dispatch sequences."
    )
    parser.add_argument("--input-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=96)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--temperature", type=float, default=0.20)
    parser.add_argument("--max-candidates", type=int, default=96)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    examples = load_examples(args.input_jsonl, args.max_candidates)
    train_examples, validation_examples = split_by_seed(
        examples,
        args.validation_fraction,
        args.seed,
    )

    model = SequenceDispatchTransformer.build(
        input_dim=SEQUENCE_DISPATCH_FEATURE_DIM,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    rng = random.Random(args.seed)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_examples = 0
        for batch in batches(train_examples, args.batch_size, rng):
            x, teacher_scores, padding_mask, _top_indices = collate(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x, padding_mask=padding_mask)
            loss = listwise_loss(logits, teacher_scores, padding_mask, args.temperature)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(batch)
            total_examples += len(batch)

        validation = evaluate(model, validation_examples, args, device)
        print(
            f"epoch={epoch} "
            f"train_loss={total_loss / float(max(1, total_examples)):.6f} "
            f"validation_loss={validation['loss']:.6f} "
            f"validation_top1={validation['top1']:.4f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "input_dim": SEQUENCE_DISPATCH_FEATURE_DIM,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "feature_names": list(SEQUENCE_DISPATCH_FEATURE_NAMES),
            "max_candidates": args.max_candidates,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        args.output,
    )
    print(f"saved sequence dispatch checkpoint to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
