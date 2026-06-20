#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


OUTPUT_FIELDS = [
    "seed",
    "agent_id",
    "env_time",
    "has_diff",
    "accepted",
    "selector_source",
    "reject_reason",
    "baseline_action",
    "baseline_action_name",
    "candidate_action",
    "candidate_action_name",
    "risk_head_candidate",
    "risk_head_baseline_minus_candidate",
    "listwise_margin",
    "candidate_raw_candidate_minus_baseline_logit",
]


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def keep_row(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.baseline_action and row.get("baseline_action_name") != args.baseline_action:
        return False
    if args.candidate_action and row.get("candidate_action_name") != args.candidate_action:
        return False
    if args.selector_source and row.get("selector_source", "") != args.selector_source:
        return False
    if args.accepted is not None and bool(row.get("accepted")) != args.accepted:
        return False
    if args.no_reject_reason and row.get("reject_reason"):
        return False

    checks = (
        ("risk_head_baseline_minus_candidate", args.min_risk_delta, None),
        ("risk_head_candidate", None, args.max_candidate_risk),
        ("listwise_margin", args.min_listwise_margin, None),
        (
            "candidate_raw_candidate_minus_baseline_logit",
            args.min_raw_margin,
            None,
        ),
    )
    for key, min_value, max_value in checks:
        value = as_float(row.get(key), float("nan"))
        if min_value is not None and not value >= min_value:
            return False
        if max_value is not None and not value <= max_value:
            return False
    return True


def row_sort_key(row: dict[str, Any], sort_by: str) -> tuple[float, float, float]:
    if sort_by == "risk_delta":
        primary = as_float(row.get("risk_head_baseline_minus_candidate"), -1e9)
    elif sort_by == "listwise_margin":
        primary = as_float(row.get("listwise_margin"), -1e9)
    elif sort_by == "raw_margin":
        primary = as_float(row.get("candidate_raw_candidate_minus_baseline_logit"), -1e9)
    else:
        primary = as_float(row.get("env_time"), -1e9)
    return (
        primary,
        as_float(row.get("risk_head_baseline_minus_candidate"), -1e9),
        as_float(row.get("listwise_margin"), -1e9),
    )


def read_rows(paths: list[Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, str]] = set()
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not keep_row(row, args):
                    continue
                key = (
                    int(row["seed"]),
                    int(row["agent_id"]),
                    int(row["env_time"]),
                    str(row.get("candidate_action_name", "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    rows.sort(key=lambda row: row_sort_key(row, args.sort_by), reverse=args.descending)
    if args.top_k is not None:
        rows = rows[: args.top_k]
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    extra_fields = sorted(
        {
            key
            for row in rows
            for key in row.keys()
            if key not in OUTPUT_FIELDS and key not in {"baseline_action", "candidate_action"}
        }
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*OUTPUT_FIELDS, *extra_fields])
        writer.writeheader()
        for row in rows:
            output = {key: row.get(key, "") for key in [*OUTPUT_FIELDS, *extra_fields]}
            output["has_diff"] = True
            writer.writerow(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert SequenceSuccessPolicy JSONL traces to counterfactual focus CSV."
    )
    parser.add_argument("trace_jsonl", nargs="+", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--baseline-action", default="")
    parser.add_argument("--candidate-action", default="")
    parser.add_argument("--selector-source", default="")
    parser.add_argument("--accepted", action=argparse.BooleanOptionalAction)
    parser.add_argument("--no-reject-reason", action="store_true")
    parser.add_argument("--min-risk-delta", type=float)
    parser.add_argument("--max-candidate-risk", type=float)
    parser.add_argument("--min-listwise-margin", type=float)
    parser.add_argument("--min-raw-margin", type=float)
    parser.add_argument(
        "--sort-by",
        choices=["risk_delta", "listwise_margin", "raw_margin", "env_time"],
        default="risk_delta",
    )
    parser.add_argument("--ascending", dest="descending", action="store_false")
    parser.set_defaults(descending=True)
    parser.add_argument("--top-k", type=int)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = read_rows(args.trace_jsonl, args)
    write_csv(args.output_csv, rows)
    print(f"wrote {len(rows)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
