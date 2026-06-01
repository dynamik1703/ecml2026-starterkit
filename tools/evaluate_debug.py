#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path


DEBUG_ENV_URL = (
    "https://data.flatland.cloud/benchmarks/Flatland3/debug-environments.zip"
)
DEFAULT_POLICY = "submission.my_policy.MyPolicy"
DEFAULT_OBS_BUILDER = "submission.my_observation_builder.MyObservationBuilder"
DEFAULT_REWARDS = "flatland.envs.rewards.ECML2026Rewards"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_entrypoint(name: str) -> str:
    candidate = Path(sys.executable).with_name(name)
    if candidate.exists():
        return str(candidate)
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(
            f"Could not find {name}. Activate .venv or run with .venv/bin/python."
        )
    return found


def ensure_debug_metadata(metadata_csv: Path, download: bool) -> None:
    if metadata_csv.exists():
        return
    if not download:
        raise FileNotFoundError(
            f"Missing metadata CSV: {metadata_csv}. "
            "Pass --download-debug to fetch the starterkit debug scenarios."
        )

    target_dir = metadata_csv.parent.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    archive_path = target_dir / "debug-environments.zip"
    print(f"Downloading debug scenarios to {archive_path}")
    urllib.request.urlretrieve(DEBUG_ENV_URL, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir)

    if not metadata_csv.exists():
        raise FileNotFoundError(f"Downloaded archive did not create {metadata_csv}")


def run_command(command: list[str], env: dict[str, str]) -> None:
    print("+ " + " ".join(command))
    subprocess.run(command, check=True, env=env)


def load_summary(analysis_dir: Path) -> list[dict[str, str]]:
    summary_path = analysis_dir / "all_trains_arrived.csv"
    with summary_path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def print_summary(rows: list[dict[str, str]]) -> None:
    if not rows:
        print("No episodes found in analysis output.")
        return

    print("\nEpisode summary:")
    print("episode_id,env_time,success_rate,normalized_reward")
    total_reward = 0.0
    total_success = 0.0
    for row in rows:
        reward = float(row["normalized_reward"])
        success = float(row["success_rate"])
        total_reward += reward
        total_success += success
        print(
            f"{row['episode_id']},{row['env_time']},"
            f"{success:.6g},{reward:.6g}"
        )

    print(
        "\nTotals: "
        f"normalized_reward_sum={total_reward:.6g}, "
        f"success_rate_mean={total_success / len(rows):.6g}, "
        f"episodes={len(rows)}"
    )


def parse_args() -> argparse.Namespace:
    root = repo_root()
    default_metadata = root / "scenarios" / "debug-environments" / "metadata.csv"
    default_output_root = root / "outputs-meta"

    parser = argparse.ArgumentParser(
        description="Run the ECML 2026 starterkit debug evaluation locally."
    )
    parser.add_argument("--metadata-csv", type=Path, default=default_metadata)
    parser.add_argument("--output-root", type=Path, default=default_output_root)
    parser.add_argument("--run-id", default=datetime.now().strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--policy", default=DEFAULT_POLICY)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument(
        "--download-debug",
        action="store_true",
        help="Download debug-environments.zip if --metadata-csv is missing.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = repo_root()
    metadata_csv = args.metadata_csv.resolve()
    ensure_debug_metadata(metadata_csv, args.download_debug)

    run_dir = (args.output_root / args.run_id).resolve()
    trajectories_dir = run_dir / "trajectories"
    analysis_dir = run_dir / "analysis"
    mplconfig_dir = run_dir / "mplconfig"
    trajectories_dir.mkdir(parents=True, exist_ok=False)
    analysis_dir.mkdir(parents=True, exist_ok=False)
    mplconfig_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{root}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env["MPLCONFIGDIR"] = str(mplconfig_dir)

    generate = resolve_entrypoint("flatland-trajectory-generate-from-metadata")
    analyze = resolve_entrypoint("flatland-trajectory-analysis")

    run_command(
        [
            generate,
            "--metadata-csv",
            str(metadata_csv),
            "--data-dir",
            str(trajectories_dir),
            "--policy",
            args.policy,
            "--obs-builder",
            args.obs_builder,
            "--rewards",
            args.rewards,
        ],
        env,
    )
    run_command(
        [
            analyze,
            "--root-data-dir",
            str(trajectories_dir),
            "--output-dir",
            str(analysis_dir),
        ],
        env,
    )

    rows = load_summary(analysis_dir)
    print_summary(rows)

    summary = {
        "run_id": args.run_id,
        "metadata_csv": str(metadata_csv),
        "trajectories_dir": str(trajectories_dir),
        "analysis_dir": str(analysis_dir),
        "normalized_reward_sum": sum(float(row["normalized_reward"]) for row in rows),
        "success_rate_mean": (
            sum(float(row["success_rate"]) for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "episodes": rows,
    }
    with (run_dir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
