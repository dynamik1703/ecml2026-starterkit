#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.evaluate_success_rescue_classifier import format_float, json_default


CONFIG_COLUMNS = (
    "ranker_score_threshold",
    "veto_max_risk_probability",
    "consensus_min_splits",
)

METRIC_COLUMNS = (
    "accepted_unique_candidates",
    "accepted_good",
    "accepted_neutral",
    "accepted_bad",
    "accepted_success_positive",
    "accepted_success_negative",
    "accepted_reward_positive",
    "accepted_reward_negative",
    "accepted_reward_delta_sum",
    "accepted_success_delta_sum",
    "accepted_failed_delta_sum",
)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def config_key(row: dict[str, Any]) -> tuple[float, float, int]:
    return (
        safe_float(row.get("ranker_score_threshold")),
        safe_float(row.get("veto_max_risk_probability")),
        safe_int(row.get("consensus_min_splits")),
    )


def read_metrics(path: Path, metrics_key: str) -> list[dict[str, Any]]:
    with path.open() as handle:
        payload = json.load(handle)
    rows = payload.get(metrics_key) if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} does not contain a {metrics_key} list")
    return [row for row in rows if isinstance(row, dict)]


def named_path(value: str) -> tuple[str, Path]:
    if "=" in value:
        name, path = value.split("=", 1)
        return name, Path(path)
    path = Path(value)
    return path.stem, path


def aggregate_common_configs(
    blocks: list[tuple[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    by_block = []
    for name, rows in blocks:
        by_block.append((name, {config_key(row): row for row in rows}))
    common_keys = set(by_block[0][1])
    for _name, rows_by_key in by_block[1:]:
        common_keys &= set(rows_by_key)

    aggregate_rows = []
    for key in common_keys:
        block_rows = {name: rows_by_key[key] for name, rows_by_key in by_block}
        aggregate: dict[str, Any] = dict(zip(CONFIG_COLUMNS, key, strict=True))
        aggregate["blocks"] = len(block_rows)
        aggregate["block_rows"] = block_rows
        for column in METRIC_COLUMNS:
            aggregate[column] = sum(safe_float(row.get(column)) for row in block_rows.values())
        aggregate["is_safe"] = (
            safe_int(aggregate["accepted_bad"]) == 0
            and safe_int(aggregate["accepted_success_negative"]) == 0
        )
        aggregate_rows.append(aggregate)

    aggregate_rows.sort(
        key=lambda row: (
            not row["is_safe"],
            safe_int(row["accepted_bad"]),
            safe_int(row["accepted_success_negative"]),
            -safe_float(row["accepted_success_delta_sum"]),
            -safe_float(row["accepted_reward_delta_sum"]),
            -safe_int(row["accepted_success_positive"]),
            -safe_int(row["accepted_good"]),
        )
    )
    return aggregate_rows


def print_rows(rows: list[dict[str, Any]], top_k: int) -> None:
    columns = (
        *CONFIG_COLUMNS,
        "is_safe",
        "accepted_unique_candidates",
        "accepted_good",
        "accepted_neutral",
        "accepted_bad",
        "accepted_success_positive",
        "accepted_success_negative",
        "accepted_reward_delta_sum",
        "accepted_success_delta_sum",
        "accepted_failed_delta_sum",
    )
    print(",".join(columns))
    for row in rows[:top_k]:
        values = []
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                values.append(format_float(value))
            else:
                values.append(str(value))
        print(",".join(values))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ranker deployment summaries across OOD blocks and rank "
            "shared configs by aggregate safety and lift."
        )
    )
    parser.add_argument(
        "json",
        nargs="+",
        help="Deployment JSON paths. Use name=/path/file.json to set a block name.",
    )
    parser.add_argument(
        "--metrics-key",
        default="consensus_metrics",
        help="Metrics list to compare from each JSON.",
    )
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blocks = [
        (name, read_metrics(path, args.metrics_key))
        for name, path in map(named_path, args.json)
    ]
    rows = aggregate_common_configs(blocks)
    safe_rows = [row for row in rows if row["is_safe"]]
    print(
        f"Data: blocks={len(blocks)} common_configs={len(rows)} "
        f"safe_common_configs={len(safe_rows)}"
    )
    print_rows(rows, args.top_k)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(
                {
                    "summary": {
                        "blocks": [name for name, _rows in blocks],
                        "common_configs": len(rows),
                        "safe_common_configs": len(safe_rows),
                        "metrics_key": args.metrics_key,
                    },
                    "metrics": rows,
                },
                handle,
                indent=2,
                default=json_default,
            )
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
