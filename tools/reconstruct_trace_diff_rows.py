#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from submission import runtime_context
from tools.analyze_policy_action_diffs import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_REWARDS,
    diff_row,
    done_all,
    instantiate_policy,
    make_env,
    observation_list,
    policy_actions,
    raw_policy_actions,
    repo_root,
    warn_if_policy_observation_mismatch,
)


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def int_value(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def scene_value(row: dict[str, Any], fallback: str | None) -> str:
    value = str(row.get("scene") or fallback or "scene_5").strip()
    return value or "scene_5"


def keep_row(row: dict[str, Any], args: argparse.Namespace) -> bool:
    if args.accepted is not None and as_bool(row.get("accepted")) != args.accepted:
        return False
    if args.selector_source and row.get("selector_source", "") != args.selector_source:
        return False
    if args.scene and scene_value(row, args.scene) != args.scene:
        return False
    return True


def read_trace_rows(paths: list[Path], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int, int]] = set()
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if not keep_row(row, args):
                    continue
                key = (
                    scene_value(row, args.scene),
                    int_value(row, "seed"),
                    int_value(row, "env_time"),
                    int_value(row, "agent_id"),
                    int_value(row, "candidate_action"),
                )
                if key in seen:
                    continue
                seen.add(key)
                rows.append(row)
    rows.sort(
        key=lambda row: (
            scene_value(row, args.scene),
            int_value(row, "seed"),
            int_value(row, "env_time"),
            int_value(row, "agent_id"),
            int_value(row, "candidate_action"),
        )
    )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump({"summary": summary, "rows": rows}, handle, indent=2)
        handle.write("\n")


def events_by_scene_seed(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[tuple[str, int], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(scene_value(row, args.scene), int_value(row, "seed"))].append(row)
    return grouped


def reconstruct_seed_rows(
    args: argparse.Namespace,
    scene: str,
    seed: int,
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    args.scene = scene
    env, obs_builder = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    runtime_context.set_seed(seed)
    runtime_context.set_scene(scene)
    baseline_policy = instantiate_policy(args.baseline_policy, args.baseline_checkpoint)
    candidate_policy = instantiate_policy(args.candidate_policy, args.candidate_checkpoint)
    events_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        events_by_step[int_value(event, "env_time")].append(event)

    rows: list[dict[str, Any]] = []
    while int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
        if obs_list:
            warn_if_policy_observation_mismatch(
                args,
                baseline_policy,
                "baseline",
                obs_list[0],
            )
            warn_if_policy_observation_mismatch(
                args,
                candidate_policy,
                "candidate",
                obs_list[0],
            )
        baseline_raw_actions = raw_policy_actions(baseline_policy, handles, obs_list)
        candidate_raw_actions = raw_policy_actions(candidate_policy, handles, obs_list)
        baseline_actions = policy_actions(baseline_policy, handles, obs_list)
        candidate_actions = policy_actions(candidate_policy, handles, obs_list)
        observations_by_handle = dict(zip(handles, obs_list))

        for event in events_by_step.get(int(env._elapsed_steps), []):
            handle = int_value(event, "agent_id")
            if handle not in baseline_actions or handle not in observations_by_handle:
                continue
            trace_candidate_action = int_value(event, "candidate_action")
            candidate_actions_for_row = dict(candidate_actions)
            candidate_actions_for_row[handle] = trace_candidate_action
            row = diff_row(
                args,
                seed,
                env,
                obs_builder,
                handle,
                observations_by_handle[handle],
                baseline_policy,
                candidate_policy,
                baseline_raw_actions,
                candidate_raw_actions,
                baseline_actions,
                candidate_actions_for_row,
            )
            trace_baseline_action = event.get("baseline_action")
            trace_candidate_matches_policy = (
                candidate_actions.get(handle) == trace_candidate_action
            )
            trace_baseline_matches_policy = (
                trace_baseline_action is not None
                and baseline_actions.get(handle) == int_value(event, "baseline_action")
            )
            row.update(
                {
                    "scene": scene,
                    "trace_accepted": str(as_bool(event.get("accepted"))),
                    "trace_candidate_source": str(event.get("candidate_source", "")),
                    "trace_candidate_policy_rank": str(
                        event.get("candidate_policy_rank", ""),
                    ),
                    "trace_candidate_policy_logit": str(
                        event.get("candidate_policy_logit", ""),
                    ),
                    "trace_selector_source": str(event.get("selector_source", "")),
                    "trace_reject_reason": str(event.get("reject_reason", "")),
                    "trace_action_match_status": (
                        "baseline_candidate"
                        if trace_baseline_matches_policy and trace_candidate_matches_policy
                        else "candidate"
                        if trace_candidate_matches_policy
                        else "baseline"
                        if trace_baseline_matches_policy
                        else "none"
                    ),
                }
            )
            rows.append(row)

        observations, _, dones, _ = env.step(baseline_actions)
        if done_all(dones):
            break
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct full analyze_policy_action_diffs.py feature rows for "
            "events recorded in policy JSONL traces."
        )
    )
    parser.add_argument("trace_jsonl", nargs="+", type=Path)
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--baseline-policy", default="submission.sequence_success_policy.MyPolicy")
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--candidate-policy", default="submission.rerank_policy.MyPolicy")
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
        help="Scene to use when trace rows do not contain a scene field.",
    )
    parser.add_argument("--selector-source", default="")
    parser.add_argument("--accepted", action=argparse.BooleanOptionalAction)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    trace_rows = read_trace_rows(args.trace_jsonl, args)
    output_rows: list[dict[str, Any]] = []
    for (scene, seed), events in sorted(events_by_scene_seed(trace_rows, args).items()):
        seed_rows = reconstruct_seed_rows(args, scene, seed, events)
        output_rows.extend(seed_rows)
        print(
            f"scene={scene} seed={seed} events={len(events)} rows={len(seed_rows)}",
            flush=True,
        )
    summary = {
        "trace_rows": len(trace_rows),
        "rows": len(output_rows),
        "missing_rows": len(trace_rows) - len(output_rows),
    }
    print(
        "summary trace_rows={trace_rows} rows={rows} missing_rows={missing_rows}".format(
            **summary,
        ),
        flush=True,
    )
    write_csv(args.output_csv, output_rows)
    if args.output_json:
        write_json(args.output_json, output_rows, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
