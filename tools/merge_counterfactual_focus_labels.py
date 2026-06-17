#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


KEY_FIELDS = ("seed", "agent_id", "env_time")
LABEL_FIELDS = (
    "baseline_reward",
    "forced_reward",
    "reward_delta",
    "baseline_success",
    "forced_success",
    "success_delta",
    "baseline_env_time",
    "forced_env_time",
    "env_time_delta",
    "baseline_failed_agents",
    "forced_failed_agents",
    "failed_agents_delta",
    "forced_applied",
)


def key_for(row: dict[str, str]) -> tuple[str, str, str]:
    return tuple(str(int(float(row[field]))) for field in KEY_FIELDS)


def read_counterfactual_rows(paths: list[Path]) -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                rows[key_for(row)] = row
    return rows


def merged_rows(
    focus_paths: list[Path],
    counterfactual_by_key: dict[tuple[str, str, str], dict[str, str]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in focus_paths:
        with path.open(newline="") as handle:
            for focus_row in csv.DictReader(handle):
                key = key_for(focus_row)
                if key in seen:
                    continue
                outcome_row = counterfactual_by_key.get(key)
                if outcome_row is None:
                    continue
                seen.add(key)
                merged = dict(focus_row)
                for field in LABEL_FIELDS:
                    merged[field] = outcome_row.get(field, "")
                rows.append(merged)
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge sequence trace focus rows with counterfactual outcome labels."
        )
    )
    parser.add_argument("--focus-csv", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--counterfactual-csv",
        nargs="+",
        required=True,
        type=Path,
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    outcomes = read_counterfactual_rows(args.counterfactual_csv)
    rows = merged_rows(args.focus_csv, outcomes)
    write_csv(args.output_csv, rows)
    print(f"wrote {len(rows)} merged rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
