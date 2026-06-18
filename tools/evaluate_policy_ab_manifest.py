#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/ecml_mpl_cache")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/ecml_xdg_cache")

from tools.evaluate_sampled import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_REWARDS,
    repo_root,
)


DEFAULT_POLICY = "submission.sequence_success_policy.MyPolicy"
ENV_PATH_HINTS = ("CHECKPOINT", "MODEL", "PATH", "PKL")


def parse_env_item(item: str) -> tuple[str, str]:
    if "=" not in item:
        raise argparse.ArgumentTypeError(
            f"environment override must be KEY=VALUE, got {item!r}"
        )
    key, value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("environment override key must not be empty")
    return key, value


def env_items(items: list[str] | None) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in items or []:
        key, value = parse_env_item(item)
        result[key] = value
    return result


def safe_name(value: str | None) -> str:
    if not value:
        return "scene_5"
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)


def sha256_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    stat = path.stat()
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "size": stat.st_size,
        "mtime": stat.st_mtime,
    }


def git_output(args: list[str]) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=repo_root(),
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return ""
    return result.stdout.strip()


def base_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    if not args.inherit_ecml_sequence_env:
        for key in list(env):
            if key.startswith("ECML_SEQUENCE_"):
                env.pop(key, None)
    env["PYTHONPATH"] = str(repo_root())
    env.setdefault("MPLCONFIGDIR", "/private/tmp/ecml_mpl_cache")
    env.setdefault("XDG_CACHE_HOME", "/private/tmp/ecml_xdg_cache")
    return env


def merged_env(
    args: argparse.Namespace,
    shared: dict[str, str],
    specific: dict[str, str],
) -> dict[str, str]:
    env = base_env(args)
    env.update(shared)
    env.update(specific)
    return env


def command_for(
    args: argparse.Namespace,
    policy: str,
    checkpoint: Path | None,
    scene: str,
    output_json: Path,
    output_csv: Path,
) -> list[str]:
    command = [
        str(args.python),
        str(repo_root() / "tools" / "evaluate_sampled.py"),
        "--policy",
        policy,
        "--base-state-pkl",
        str(args.base_state_pkl),
        "--obs-builder",
        args.obs_builder,
        "--rewards",
        args.rewards,
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--line-length",
        str(args.line_length),
        "--output-json",
        str(output_json),
        "--output-csv",
        str(output_csv),
    ]
    if args.num_agents is not None:
        command.extend(["--num-agents", str(args.num_agents)])
    if scene != "scene_5":
        command.extend(["--scene", scene])
    if checkpoint is not None:
        command.extend(["--policy-checkpoint", str(checkpoint)])
    return command


def run_eval(
    command: list[str],
    env: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as stdout_handle, stderr_path.open("w") as stderr_handle:
        result = subprocess.run(
            command,
            cwd=repo_root(),
            env=env,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
        )
    if result.returncode != 0:
        stdout_tail = stdout_path.read_text(errors="replace")[-4000:]
        stderr_tail = stderr_path.read_text(errors="replace")[-4000:]
        raise RuntimeError(
            "evaluation command failed with "
            f"exit code {result.returncode}: {' '.join(command)}\n"
            f"stdout tail:\n{stdout_tail}\n"
            f"stderr tail:\n{stderr_tail}"
        )


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def compare_rows(
    scene: str,
    baseline_csv: Path,
    candidate_csv: Path,
) -> list[dict[str, Any]]:
    baseline_rows = {int(row["seed"]): row for row in read_csv_rows(baseline_csv)}
    candidate_rows = {int(row["seed"]): row for row in read_csv_rows(candidate_csv)}
    rows: list[dict[str, Any]] = []
    for seed in sorted(set(baseline_rows) & set(candidate_rows)):
        baseline = baseline_rows[seed]
        candidate = candidate_rows[seed]
        baseline_reward = safe_float(baseline["normalized_reward"])
        candidate_reward = safe_float(candidate["normalized_reward"])
        baseline_success = safe_float(baseline["success_rate"])
        candidate_success = safe_float(candidate["success_rate"])
        baseline_env_time = int(safe_float(baseline["env_time"]))
        candidate_env_time = int(safe_float(candidate["env_time"]))
        rows.append(
            {
                "scene": scene,
                "seed": seed,
                "baseline_reward": baseline_reward,
                "candidate_reward": candidate_reward,
                "reward_delta": candidate_reward - baseline_reward,
                "baseline_success": baseline_success,
                "candidate_success": candidate_success,
                "success_delta": candidate_success - baseline_success,
                "baseline_env_time": baseline_env_time,
                "candidate_env_time": candidate_env_time,
                "env_time_delta": candidate_env_time - baseline_env_time,
                "baseline_num_agents": int(safe_float(baseline["num_agents"])),
                "candidate_num_agents": int(safe_float(candidate["num_agents"])),
                "baseline_max_episode_steps": int(
                    safe_float(baseline["max_episode_steps"])
                ),
                "candidate_max_episode_steps": int(
                    safe_float(candidate["max_episode_steps"])
                ),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "episodes": 0,
            "reward_delta_mean": 0.0,
            "success_delta_mean": 0.0,
            "reward_wins": 0,
            "reward_losses": 0,
            "reward_ties": 0,
            "success_wins": 0,
            "success_losses": 0,
            "success_ties": 0,
        }

    def mean(key: str) -> float:
        return sum(float(row[key]) for row in rows) / len(rows)

    return {
        "episodes": len(rows),
        "baseline_reward_mean": mean("baseline_reward"),
        "candidate_reward_mean": mean("candidate_reward"),
        "reward_delta_mean": mean("reward_delta"),
        "baseline_success_mean": mean("baseline_success"),
        "candidate_success_mean": mean("candidate_success"),
        "success_delta_mean": mean("success_delta"),
        "reward_wins": sum(int(float(row["reward_delta"]) > 1e-9) for row in rows),
        "reward_losses": sum(int(float(row["reward_delta"]) < -1e-9) for row in rows),
        "reward_ties": sum(
            int(abs(float(row["reward_delta"])) <= 1e-9) for row in rows
        ),
        "success_wins": sum(int(float(row["success_delta"]) > 1e-9) for row in rows),
        "success_losses": sum(int(float(row["success_delta"]) < -1e-9) for row in rows),
        "success_ties": sum(
            int(abs(float(row["success_delta"])) <= 1e-9) for row in rows
        ),
    }


def write_compare_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def trace_summary(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {
            "path": str(path) if path is not None else None,
            "exists": False,
            "rows": 0,
            "accepted": 0,
            "accepted_by_source": {},
        }
    accepted_by_source: Counter[str] = Counter()
    rows = 0
    accepted = 0
    with path.open() as handle:
        for line in handle:
            rows += 1
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("accepted"):
                accepted += 1
                accepted_by_source[str(payload.get("selector_source", ""))] += 1
    return {
        "path": str(path),
        "exists": True,
        "rows": rows,
        "accepted": accepted,
        "accepted_by_source": dict(sorted(accepted_by_source.items())),
    }


def env_file_candidates(env: dict[str, str]) -> list[Path]:
    paths: list[Path] = []
    for key, value in env.items():
        if not any(hint in key for hint in ENV_PATH_HINTS):
            continue
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            path = Path(item)
            if not path.is_absolute():
                path = repo_root() / path
            if path.exists() and path.is_file():
                paths.append(path)
    return paths


def collect_file_hashes(
    args: argparse.Namespace,
    baseline_env: dict[str, str],
    candidate_env: dict[str, str],
) -> dict[str, dict[str, Any]]:
    paths = [
        args.base_state_pkl,
        *(args.model_path or []),
        *env_file_candidates(baseline_env),
        *env_file_candidates(candidate_env),
    ]
    if args.baseline_checkpoint is not None:
        paths.append(args.baseline_checkpoint)
    if args.candidate_checkpoint is not None:
        paths.append(args.candidate_checkpoint)

    result: dict[str, dict[str, Any]] = {}
    for path in paths:
        resolved = path if path.is_absolute() else repo_root() / path
        if resolved.exists() and resolved.is_file():
            result[str(resolved)] = sha256_file(resolved)
    return result


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_block(
    args: argparse.Namespace,
    scene: str,
    shared_env: dict[str, str],
    baseline_env_overrides: dict[str, str],
    candidate_env_overrides: dict[str, str],
) -> dict[str, Any]:
    block_name = f"{safe_name(scene)}_seed{args.seed}_n{args.episodes}"
    baseline_json = args.output_dir / f"{block_name}_baseline.json"
    baseline_csv = args.output_dir / f"{block_name}_baseline.csv"
    candidate_json = args.output_dir / f"{block_name}_candidate.json"
    candidate_csv = args.output_dir / f"{block_name}_candidate.csv"
    compare_csv = args.output_dir / f"{block_name}_compare.csv"
    baseline_stdout = args.output_dir / f"{block_name}_baseline.stdout.txt"
    baseline_stderr = args.output_dir / f"{block_name}_baseline.stderr.txt"
    candidate_stdout = args.output_dir / f"{block_name}_candidate.stdout.txt"
    candidate_stderr = args.output_dir / f"{block_name}_candidate.stderr.txt"

    candidate_specific = dict(candidate_env_overrides)
    candidate_trace_path = None
    if args.candidate_trace:
        candidate_trace_path = args.output_dir / f"{block_name}_candidate_trace.jsonl"
        candidate_trace_path.unlink(missing_ok=True)
        candidate_specific["ECML_SEQUENCE_TRACE_PATH"] = str(candidate_trace_path)
    if args.candidate_trace_all:
        candidate_specific["ECML_SEQUENCE_TRACE_ALL"] = "1"

    baseline_env = merged_env(args, shared_env, baseline_env_overrides)
    candidate_env = merged_env(args, shared_env, candidate_specific)

    baseline_command = command_for(
        args,
        args.baseline_policy,
        args.baseline_checkpoint,
        scene,
        baseline_json,
        baseline_csv,
    )
    candidate_command = command_for(
        args,
        args.candidate_policy,
        args.candidate_checkpoint,
        scene,
        candidate_json,
        candidate_csv,
    )

    print(f"baseline scene={scene} seed={args.seed} episodes={args.episodes}", flush=True)
    run_eval(baseline_command, baseline_env, baseline_stdout, baseline_stderr)
    print(f"candidate scene={scene} seed={args.seed} episodes={args.episodes}", flush=True)
    run_eval(candidate_command, candidate_env, candidate_stdout, candidate_stderr)

    rows = compare_rows(scene, baseline_csv, candidate_csv)
    write_compare_csv(compare_csv, rows)
    summary = summarize(rows)
    trace = trace_summary(candidate_trace_path)
    print(
        "summary "
        f"scene={scene} episodes={summary['episodes']} "
        f"reward_delta_mean={summary['reward_delta_mean']:.6g} "
        f"success_delta_mean={summary['success_delta_mean']:.6g} "
        f"reward={summary['reward_wins']}/{summary['reward_losses']}/"
        f"{summary['reward_ties']} "
        f"success={summary['success_wins']}/{summary['success_losses']}/"
        f"{summary['success_ties']} "
        f"accepted={trace['accepted_by_source']}",
        flush=True,
    )

    return {
        "scene": scene,
        "seed": args.seed,
        "episodes": args.episodes,
        "summary": summary,
        "trace_summary": trace,
        "paths": {
            "baseline_json": str(baseline_json),
            "baseline_csv": str(baseline_csv),
            "candidate_json": str(candidate_json),
            "candidate_csv": str(candidate_csv),
            "compare_csv": str(compare_csv),
            "baseline_stdout": str(baseline_stdout),
            "baseline_stderr": str(baseline_stderr),
            "candidate_stdout": str(candidate_stdout),
            "candidate_stderr": str(candidate_stderr),
        },
        "commands": {
            "baseline": baseline_command,
            "candidate": candidate_command,
        },
        "env": {
            "baseline_overrides": baseline_env_overrides,
            "candidate_overrides": candidate_specific,
            "shared_overrides": shared_env,
            "inherit_ecml_sequence_env": bool(args.inherit_ecml_sequence_env),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run reproducible A/B sampled-policy evaluations by executing "
            "baseline and candidate in separate subprocesses with separate envs."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-policy", default=DEFAULT_POLICY)
    parser.add_argument("--candidate-policy", default=DEFAULT_POLICY)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--candidate-checkpoint", type=Path)
    parser.add_argument("--base-state-pkl", type=Path, default=repo_root() / DEFAULT_BASE_STATE)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-agents", type=int)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scenes",
        nargs="+",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
        default=["scene_5"],
    )
    parser.add_argument(
        "--shared-env",
        action="append",
        default=[],
        help="Environment override KEY=VALUE applied to both subprocesses.",
    )
    parser.add_argument(
        "--baseline-env",
        action="append",
        default=[],
        help="Environment override KEY=VALUE applied only to baseline.",
    )
    parser.add_argument(
        "--candidate-env",
        action="append",
        default=[],
        help="Environment override KEY=VALUE applied only to candidate.",
    )
    parser.add_argument(
        "--inherit-ecml-sequence-env",
        action="store_true",
        help="Do not scrub inherited ECML_SEQUENCE_* variables before applying overrides.",
    )
    parser.add_argument(
        "--candidate-trace",
        action="store_true",
        help="Set ECML_SEQUENCE_TRACE_PATH for candidate runs and summarize the trace.",
    )
    parser.add_argument(
        "--candidate-trace-all",
        action="store_true",
        help="Set ECML_SEQUENCE_TRACE_ALL=1 for candidate runs.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        action="append",
        help="Additional model/data file to hash in the manifest.",
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--manifest-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    shared_env = env_items(args.shared_env)
    baseline_env_overrides = env_items(args.baseline_env)
    candidate_env_overrides = env_items(args.candidate_env)

    runs = [
        run_block(args, scene, shared_env, baseline_env_overrides, candidate_env_overrides)
        for scene in args.scenes
    ]

    all_rows: list[dict[str, Any]] = []
    for run in runs:
        all_rows.extend(read_csv_rows(Path(run["paths"]["compare_csv"])))
    aggregate = summarize(all_rows)

    baseline_manifest_env = merged_env(args, shared_env, baseline_env_overrides)
    candidate_manifest_env = merged_env(args, shared_env, candidate_env_overrides)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git_output(["rev-parse", "HEAD"]),
            "status_short": git_output(["status", "--short"]),
        },
        "config": {
            "baseline_policy": args.baseline_policy,
            "candidate_policy": args.candidate_policy,
            "baseline_checkpoint": args.baseline_checkpoint,
            "candidate_checkpoint": args.candidate_checkpoint,
            "base_state_pkl": args.base_state_pkl,
            "obs_builder": args.obs_builder,
            "rewards": args.rewards,
            "seed": args.seed,
            "episodes": args.episodes,
            "num_agents": args.num_agents,
            "line_length": args.line_length,
            "scenes": args.scenes,
            "python": args.python,
        },
        "file_hashes": collect_file_hashes(
            args,
            baseline_manifest_env,
            candidate_manifest_env,
        ),
        "runs": runs,
        "aggregate_summary": aggregate,
    }

    manifest_json = args.manifest_json or args.output_dir / "manifest.json"
    with manifest_json.open("w") as handle:
        json.dump(manifest, handle, indent=2, default=json_default)
        handle.write("\n")

    aggregate_csv = args.output_dir / "aggregate_compare.csv"
    write_compare_csv(aggregate_csv, all_rows)
    print(
        "aggregate "
        f"episodes={aggregate['episodes']} "
        f"reward_delta_mean={aggregate['reward_delta_mean']:.6g} "
        f"success_delta_mean={aggregate['success_delta_mean']:.6g} "
        f"reward={aggregate['reward_wins']}/{aggregate['reward_losses']}/"
        f"{aggregate['reward_ties']} "
        f"success={aggregate['success_wins']}/{aggregate['success_losses']}/"
        f"{aggregate['success_ties']}"
    )
    print(f"manifest={manifest_json}")
    print(f"aggregate_csv={aggregate_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
