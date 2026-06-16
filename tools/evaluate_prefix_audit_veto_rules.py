#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tools.evaluate_success_rescue_classifier import format_float, json_default


RuleFn = Callable[[dict[str, Any]], bool]


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


def candidate_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("seed", "")),
        str(row.get("prefix_len", "")),
        str(row.get("forced_applied", "")),
        str(row.get("events", "")),
        round(safe_float(row.get("reward_delta")), 9),
        round(safe_float(row.get("success_delta")), 9),
    )


def read_validation_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open() as handle:
            payload = json.load(handle)
        block_rows = payload.get("rows") if isinstance(payload, dict) else payload
        if not isinstance(block_rows, list):
            raise ValueError(f"{path} does not contain a row list")
        rows.extend(row for row in block_rows if isinstance(row, dict))
    return rows


def read_audit_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def joined_accepted_rows(
    audit_rows: list[dict[str, str]],
    validation_rows: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    missing = 0
    mismatched = 0
    for audit_row in audit_rows:
        if safe_int(audit_row.get("accepted")) != 1:
            continue
        if safe_int(audit_row.get("selected_best_for_seed")) != 1:
            continue
        if safe_int(audit_row.get("is_baseline_candidate")) == 1:
            continue
        if abs(safe_float(audit_row.get("score_threshold")) - threshold) > 1e-9:
            continue

        row_index = safe_int(audit_row.get("row_index"), default=-1)
        if row_index < 0 or row_index >= len(validation_rows):
            missing += 1
            continue

        validation_row = validation_rows[row_index]
        if (
            str(validation_row.get("seed", "")) != str(audit_row.get("seed", ""))
            or str(validation_row.get("prefix_len", ""))
            != str(audit_row.get("prefix_len", ""))
            or str(validation_row.get("events", "")) != str(audit_row.get("events", ""))
        ):
            mismatched += 1
            continue

        joined = dict(validation_row)
        joined.update(audit_row)
        joined["row_index"] = row_index
        joined["rank_score_lcb"] = safe_float(audit_row.get("rank_score_lcb"))
        rows.append(joined)
    return rows, missing, mismatched


def metric(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "accepted": len(rows),
        "accepted_good": sum(row.get("outcome_category") == "good" for row in rows),
        "accepted_neutral": sum(row.get("outcome_category") == "neutral" for row in rows),
        "accepted_bad": sum(row.get("outcome_category") == "bad" for row in rows),
        "accepted_success_positive": sum(
            safe_float(row.get("success_delta")) > 1e-9 for row in rows
        ),
        "accepted_success_negative": sum(
            safe_float(row.get("success_delta")) < -1e-9 for row in rows
        ),
        "accepted_reward_positive": sum(
            safe_float(row.get("reward_delta")) > 1e-6 for row in rows
        ),
        "accepted_reward_negative": sum(
            safe_float(row.get("reward_delta")) < -1e-6 for row in rows
        ),
        "accepted_reward_delta_sum": sum(safe_float(row.get("reward_delta")) for row in rows),
        "accepted_success_delta_sum": sum(
            safe_float(row.get("success_delta")) for row in rows
        ),
        "accepted_failed_delta_sum": sum(
            safe_float(row.get("failed_agents_delta")) for row in rows
        ),
    }


def dedupe_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = candidate_key(row)
        current = by_key.get(key)
        if current is None or safe_float(row.get("rank_score_lcb")) > safe_float(
            current.get("rank_score_lcb")
        ):
            by_key[key] = row
    return list(by_key.values())


def no_veto(_row: dict[str, Any]) -> bool:
    return False


def wide_multi_agent_prefix(row: dict[str, Any]) -> bool:
    return safe_int(row.get("event_unique_agents")) >= 3 and safe_int(
        row.get("prefix_len")
    ) >= 4


def long_multi_agent_prefix(row: dict[str, Any]) -> bool:
    return safe_float(row.get("event_time_span")) >= 100.0 and safe_int(
        row.get("event_unique_agents")
    ) >= 4


def repeated_right_to_forward_span(row: dict[str, Any]) -> bool:
    return (
        safe_int(row.get("event_transition_MOVE_RIGHT__MOVE_FORWARD")) >= 3
        and safe_float(row.get("event_time_span")) >= 35.0
        and safe_int(row.get("event_unique_agents")) >= 2
    )


def early_multi_agent_burst(row: dict[str, Any]) -> bool:
    return (
        safe_float(row.get("event_time_first")) <= 40.0
        and safe_int(row.get("event_unique_agents")) >= 3
        and safe_int(row.get("prefix_len")) >= 4
    )


def combined_risk_v1(row: dict[str, Any]) -> bool:
    return repeated_right_to_forward_span(row) or wide_multi_agent_prefix(row)


RULES: dict[str, RuleFn] = {
    "no_veto": no_veto,
    "wide_multi_agent_prefix": wide_multi_agent_prefix,
    "long_multi_agent_prefix": long_multi_agent_prefix,
    "repeated_right_to_forward_span": repeated_right_to_forward_span,
    "early_multi_agent_burst": early_multi_agent_burst,
    "combined_risk_v1": combined_risk_v1,
}


def summarize(
    audit_rows: list[dict[str, str]],
    validation_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for threshold in args.score_threshold:
        accepted_rows, missing, mismatched = joined_accepted_rows(
            audit_rows=audit_rows,
            validation_rows=validation_rows,
            threshold=threshold,
        )
        for scope, scoped_rows in (
            ("split_rows", accepted_rows),
            ("unique_candidates", dedupe_rows(accepted_rows)),
        ):
            before = metric(scoped_rows)
            for rule_name in args.rule:
                rule = RULES[rule_name]
                kept_rows = [row for row in scoped_rows if not rule(row)]
                vetoed_rows = [row for row in scoped_rows if rule(row)]
                row = {
                    "score_threshold": threshold,
                    "scope": scope,
                    "rule": rule_name,
                    "missing_join_rows": missing,
                    "mismatched_join_rows": mismatched,
                    "vetoed": len(vetoed_rows),
                    "vetoed_good": sum(
                        candidate.get("outcome_category") == "good"
                        for candidate in vetoed_rows
                    ),
                    "vetoed_neutral": sum(
                        candidate.get("outcome_category") == "neutral"
                        for candidate in vetoed_rows
                    ),
                    "vetoed_bad": sum(
                        candidate.get("outcome_category") == "bad"
                        for candidate in vetoed_rows
                    ),
                    "vetoed_success_positive": sum(
                        safe_float(candidate.get("success_delta")) > 1e-9
                        for candidate in vetoed_rows
                    ),
                    "vetoed_success_negative": sum(
                        safe_float(candidate.get("success_delta")) < -1e-9
                        for candidate in vetoed_rows
                    ),
                }
                row.update({f"before_{key}": value for key, value in before.items()})
                row.update({f"after_{key}": value for key, value in metric(kept_rows).items()})
                summaries.append(row)
    summaries.sort(
        key=lambda row: (
            row["score_threshold"],
            row["scope"],
            row["after_accepted_success_negative"],
            row["after_accepted_bad"],
            -row["after_accepted_success_delta_sum"],
            -row["after_accepted_reward_delta_sum"],
            row["rule"],
        )
    )
    return summaries


def print_table(rows: list[dict[str, Any]], top_k: int) -> None:
    columns = [
        "score_threshold",
        "scope",
        "rule",
        "before_accepted",
        "before_accepted_good",
        "before_accepted_neutral",
        "before_accepted_bad",
        "before_accepted_success_positive",
        "before_accepted_success_negative",
        "before_accepted_reward_delta_sum",
        "before_accepted_success_delta_sum",
        "vetoed",
        "vetoed_good",
        "vetoed_neutral",
        "vetoed_bad",
        "vetoed_success_positive",
        "vetoed_success_negative",
        "after_accepted",
        "after_accepted_good",
        "after_accepted_neutral",
        "after_accepted_bad",
        "after_accepted_success_positive",
        "after_accepted_success_negative",
        "after_accepted_reward_delta_sum",
        "after_accepted_success_delta_sum",
        "after_accepted_failed_delta_sum",
        "missing_join_rows",
        "mismatched_join_rows",
    ]
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
            "Join prefix ranker audit rows with mined prefix JSON features and "
            "evaluate simple veto rules before considering online deployment."
        )
    )
    parser.add_argument("--audit-csv", required=True, type=Path)
    parser.add_argument("--validation-json", nargs="+", required=True, type=Path)
    parser.add_argument(
        "--score-threshold",
        nargs="+",
        type=float,
        default=[2.5, 3.0, 4.0, 5.0],
    )
    parser.add_argument(
        "--rule",
        nargs="+",
        choices=sorted(RULES),
        default=sorted(RULES),
    )
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    audit_rows = read_audit_rows(args.audit_csv)
    validation_rows = read_validation_rows(args.validation_json)
    summaries = summarize(audit_rows, validation_rows, args)
    print(
        "Data: "
        f"audit_rows={len(audit_rows)} validation_rows={len(validation_rows)} "
        f"thresholds={','.join(format_float(value) for value in args.score_threshold)}"
    )
    print_table(summaries, args.top_k)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(
                {
                    "summary": {
                        "audit_csv": str(args.audit_csv),
                        "validation_json": [str(path) for path in args.validation_json],
                        "audit_rows": len(audit_rows),
                        "validation_rows": len(validation_rows),
                        "args": vars(args),
                    },
                    "metrics": summaries,
                },
                handle,
                indent=2,
                default=json_default,
            )
            handle.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
