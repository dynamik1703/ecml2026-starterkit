#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from tools.evaluate_sampled import (
        DEFAULT_BASE_STATE,
        DEFAULT_OBS_BUILDER,
        DEFAULT_REWARDS,
        repo_root,
        run_episode,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback.
    from evaluate_sampled import (  # type: ignore
        DEFAULT_BASE_STATE,
        DEFAULT_OBS_BUILDER,
        DEFAULT_REWARDS,
        repo_root,
        run_episode,
    )


@dataclass(frozen=True)
class Candidate:
    name: str
    policy: str
    obs_builder: str
    checkpoint: Path | None = None


def parse_seed_blocks(values: list[str]) -> list[int]:
    seeds: list[int] = []
    for value in values:
        for part in value.split(","):
            token = part.strip()
            if not token:
                continue
            if ":" in token:
                start_text, count_text = token.split(":", maxsplit=1)
                start = int(start_text)
                count = int(count_text)
                if count < 0:
                    raise argparse.ArgumentTypeError(f"Negative seed count: {token}")
                seeds.extend(range(start, start + count))
            elif "-" in token:
                start_text, end_text = token.split("-", maxsplit=1)
                start = int(start_text)
                end = int(end_text)
                if end < start:
                    raise argparse.ArgumentTypeError(
                        f"Seed range ends before start: {token}"
                    )
                seeds.extend(range(start, end + 1))
            else:
                seeds.append(int(token))
    return list(dict.fromkeys(seeds))


def parse_candidate(value: str) -> Candidate:
    if "=" in value:
        name, spec = value.split("=", maxsplit=1)
    else:
        spec = value
        name = spec.split(",", maxsplit=1)[0].rsplit(".", maxsplit=1)[-1]

    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(f"Empty candidate spec: {value}")

    policy_spec = parts[0]
    checkpoint: Path | None = None
    if "@" in policy_spec:
        policy, checkpoint_text = policy_spec.rsplit("@", maxsplit=1)
        checkpoint = Path(checkpoint_text)
    else:
        policy = policy_spec

    obs_builder = DEFAULT_OBS_BUILDER
    for part in parts[1:]:
        if "=" not in part:
            raise argparse.ArgumentTypeError(
                f"Candidate option must be key=value: {part}"
            )
        key, item_value = (text.strip() for text in part.split("=", maxsplit=1))
        if key in {"obs", "obs_builder"}:
            obs_builder = item_value
        elif key in {"ckpt", "checkpoint"}:
            checkpoint = Path(item_value)
        else:
            raise argparse.ArgumentTypeError(f"Unknown candidate option: {key}")

    return Candidate(
        name=name.strip(),
        policy=policy.strip(),
        obs_builder=obs_builder,
        checkpoint=checkpoint,
    )


def episode_args(
    args: argparse.Namespace,
    candidate: Candidate,
    seed: int,
    num_agents: int,
    line_length: int,
    scene: str | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        base_state_pkl=args.base_state_pkl,
        policy=candidate.policy,
        policy_checkpoint=candidate.checkpoint,
        obs_builder=candidate.obs_builder,
        rewards=args.rewards,
        episodes=1,
        seed=seed,
        num_agents=num_agents,
        line_length=line_length,
        scene=scene,
        output_json=None,
        output_csv=None,
        agent_details=False,
    )


def flatten_row(
    candidate: Candidate,
    config: argparse.Namespace,
    row: dict[str, Any],
) -> dict[str, Any]:
    failed_agents = row.get("failed_agents", [])
    return {
        "candidate": candidate.name,
        "policy": candidate.policy,
        "obs_builder": candidate.obs_builder,
        "policy_checkpoint": str(candidate.checkpoint) if candidate.checkpoint else "",
        "seed": int(row["seed"]),
        "num_agents_config": int(config.num_agents),
        "line_length_config": int(config.line_length),
        "scene_config": config.scene or "scene_5",
        "env_time": int(row["env_time"]),
        "success_rate": float(row["success_rate"]),
        "normalized_reward": float(row["normalized_reward"]),
        "num_agents": int(row["num_agents"]),
        "max_episode_steps": int(row["max_episode_steps"]),
        "failed_agents": len(failed_agents),
    }


def row_key(row: dict[str, Any]) -> tuple[int, int, str, int]:
    return (
        int(row["num_agents_config"]),
        int(row["line_length_config"]),
        str(row["scene_config"]),
        int(row["seed"]),
    )


def add_baseline_deltas(
    rows: list[dict[str, Any]],
    baseline_name: str,
) -> list[dict[str, Any]]:
    baseline_by_key = {
        row_key(row): row
        for row in rows
        if row["candidate"] == baseline_name
    }
    enriched = []
    for row in rows:
        output = dict(row)
        baseline = baseline_by_key.get(row_key(row))
        if baseline is None:
            output.update(
                {
                    "baseline_candidate": baseline_name,
                    "reward_delta": "",
                    "success_delta": "",
                    "env_time_delta": "",
                    "failed_agents_delta": "",
                }
            )
        else:
            output.update(
                {
                    "baseline_candidate": baseline_name,
                    "reward_delta": (
                        float(row["normalized_reward"])
                        - float(baseline["normalized_reward"])
                    ),
                    "success_delta": (
                        float(row["success_rate"]) - float(baseline["success_rate"])
                    ),
                    "env_time_delta": int(row["env_time"]) - int(baseline["env_time"]),
                    "failed_agents_delta": (
                        int(row["failed_agents"]) - int(baseline["failed_agents"])
                    ),
                }
            )
        enriched.append(output)
    return enriched


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows) if rows else 0.0


def summarize(
    rows: list[dict[str, Any]],
    baseline_name: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row["candidate"],
            row["policy"],
            row["obs_builder"],
            row["policy_checkpoint"],
            row["num_agents_config"],
            row["line_length_config"],
            row["scene_config"],
        )
        groups[key].append(row)

    summaries = []
    for (
        candidate,
        policy,
        obs_builder,
        checkpoint,
        num_agents,
        line_length,
        scene,
    ), group_rows in groups.items():
        reward_deltas = [
            float(row["reward_delta"])
            for row in group_rows
            if row["reward_delta"] != ""
        ]
        success_deltas = [
            float(row["success_delta"])
            for row in group_rows
            if row["success_delta"] != ""
        ]
        summaries.append(
            {
                "candidate": candidate,
                "baseline_candidate": baseline_name,
                "policy": policy,
                "obs_builder": obs_builder,
                "policy_checkpoint": checkpoint,
                "num_agents": num_agents,
                "line_length": line_length,
                "scene": scene,
                "episodes": len(group_rows),
                "reward_mean": _mean(group_rows, "normalized_reward"),
                "success_rate_mean": _mean(group_rows, "success_rate"),
                "reward_min": min(float(row["normalized_reward"]) for row in group_rows),
                "success_rate_min": min(float(row["success_rate"]) for row in group_rows),
                "reward_delta_mean": (
                    sum(reward_deltas) / len(reward_deltas)
                    if reward_deltas
                    else ""
                ),
                "success_delta_mean": (
                    sum(success_deltas) / len(success_deltas)
                    if success_deltas
                    else ""
                ),
                "reward_wins": sum(delta > 1e-9 for delta in reward_deltas),
                "reward_losses": sum(delta < -1e-9 for delta in reward_deltas),
                "reward_ties": sum(abs(delta) <= 1e-9 for delta in reward_deltas),
                "success_wins": sum(delta > 1e-9 for delta in success_deltas),
                "success_losses": sum(delta < -1e-9 for delta in success_deltas),
                "success_ties": sum(abs(delta) <= 1e-9 for delta in success_deltas),
                "failed_agents_total": sum(
                    int(row["failed_agents"]) for row in group_rows
                ),
            }
        )

    return sorted(
        summaries,
        key=lambda row: (
            int(row["num_agents"]),
            int(row["line_length"]),
            str(row["scene"]),
            -float(row["reward_mean"]),
            -float(row["success_rate_mean"]),
            str(row["candidate"]),
        ),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a policy scoreboard with per-candidate observation builders "
            "and deltas against a baseline candidate."
        )
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument(
        "--candidate",
        action="append",
        type=parse_candidate,
        required=True,
        help=(
            "Candidate as name=module.Class[@checkpoint],obs=module.ObsBuilder. "
            "Can be repeated. If obs is omitted, the starterkit default is used."
        ),
    )
    parser.add_argument(
        "--baseline",
        help="Candidate name used for deltas. Defaults to the first candidate.",
    )
    parser.add_argument(
        "--blocks",
        nargs="+",
        default=["100-129"],
        help="Seed blocks, e.g. 100-129, 100:30, or comma-separated lists.",
    )
    parser.add_argument("--num-agents", type=int, nargs="+", default=[6])
    parser.add_argument("--line-lengths", type=int, nargs="+", default=[2])
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
        default=["scene_5"],
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = list(args.candidate)
    baseline_name = args.baseline or candidates[0].name
    candidate_names = {candidate.name for candidate in candidates}
    if baseline_name not in candidate_names:
        raise ValueError(f"Unknown baseline candidate: {baseline_name}")
    seeds = parse_seed_blocks(args.blocks)
    rows = []

    for candidate in candidates:
        for num_agents in args.num_agents:
            for line_length in args.line_lengths:
                for scene_name in args.scenes:
                    scene = None if scene_name == "scene_5" else scene_name
                    for seed in seeds:
                        config = episode_args(
                            args,
                            candidate,
                            seed,
                            num_agents,
                            line_length,
                            scene,
                        )
                        episode_row = run_episode(config, seed)
                        flat = flatten_row(candidate, config, episode_row)
                        rows.append(flat)
                        if not args.quiet:
                            print(
                                f"{flat['candidate']} seed={seed} "
                                f"agents={num_agents} line={line_length} "
                                f"scene={scene_name} "
                                f"reward={flat['normalized_reward']:.6g} "
                                f"success={flat['success_rate']:.6g}",
                                flush=True,
                            )

    rows = add_baseline_deltas(rows, baseline_name)
    summaries = summarize(rows, baseline_name)

    print("\nSummary:")
    print(
        "candidate,baseline,agents,line,scene,episodes,"
        "reward_mean,success_mean,reward_delta,success_delta,"
        "reward_w/l/t,success_w/l/t"
    )
    for summary in summaries:
        reward_delta = summary["reward_delta_mean"]
        success_delta = summary["success_delta_mean"]
        reward_delta_text = (
            f"{float(reward_delta):.6g}" if reward_delta != "" else ""
        )
        success_delta_text = (
            f"{float(success_delta):.6g}" if success_delta != "" else ""
        )
        print(
            f"{summary['candidate']},{summary['baseline_candidate']},"
            f"{summary['num_agents']},{summary['line_length']},{summary['scene']},"
            f"{summary['episodes']},{summary['reward_mean']:.6g},"
            f"{summary['success_rate_mean']:.6g},"
            f"{reward_delta_text},{success_delta_text},"
            f"{summary['reward_wins']}/{summary['reward_losses']}/"
            f"{summary['reward_ties']},"
            f"{summary['success_wins']}/{summary['success_losses']}/"
            f"{summary['success_ties']}"
        )

    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    if args.summary_csv is not None:
        write_csv(args.summary_csv, summaries)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump({"rows": rows, "summaries": summaries}, handle, indent=2)
            handle.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
