#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from tools.evaluate_sequence_rescue_planner import read_json_rows
from tools.evaluate_success_rescue_classifier import json_default
from tools.mine_diff_prefix_dataset import (
    add_delta_features,
    aggregate_event_features,
    candidate_source_features,
)
from tools.train_counterfactual_gate import outcome_category, safe_float


def enrich_row(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    events = row.get("event_details", [])
    if not isinstance(events, list):
        events = []

    enriched_events = []
    for index, event in enumerate(events, start=1):
        if not isinstance(event, dict):
            continue
        detail = add_delta_features(event)
        detail["prefix_index"] = int(detail.get("prefix_index", index))
        enriched_events.append(detail)

    for key in list(enriched):
        if key.startswith("event_") or key.startswith("candidate_source_"):
            enriched.pop(key, None)

    enriched["event_details"] = enriched_events
    enriched.update(aggregate_event_features(enriched_events))

    checkpoint = ""
    if enriched_events:
        checkpoint = str(enriched_events[-1].get("candidate_checkpoint", ""))
    if checkpoint:
        enriched.update(candidate_source_features("", checkpoint))
    else:
        for event in enriched_events:
            for key, value in event.items():
                if key.startswith("candidate_source_"):
                    enriched[key] = value

    reward_delta = safe_float(enriched.get("reward_delta"))
    success_delta = safe_float(enriched.get("success_delta"))
    enriched["outcome_category"] = outcome_category(
        {
            "reward_delta": reward_delta,
            "success_delta": success_delta,
        },
        1e-9,
    )
    return enriched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add current derived event/aggregate features to prefix-row JSONs."
    )
    parser.add_argument("json", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = [enrich_row(row) for row in read_json_rows(args.json)]
    categories = Counter(str(row.get("outcome_category", "")) for row in rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w") as handle:
        json.dump(
            {
                "summary": {
                    "rows": len(rows),
                    "categories": dict(categories),
                    "inputs": [str(path) for path in args.json],
                },
                "rows": rows,
            },
            handle,
            indent=2,
            default=json_default,
        )
        handle.write("\n")
    print(
        f"wrote rows={len(rows)} categories={dict(categories)} "
        f"output={args.output_json}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
