#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


STATE_NAMES = {
    "0": "WAITING",
    "1": "READY_TO_DEPART",
    "2": "MALFUNCTION_OFF_MAP",
    "3": "MOVING",
    "4": "STOPPED",
    "5": "MALFUNCTION",
    "6": "DONE",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group_by(rows: list[dict[str, str]], *keys: str) -> dict[tuple[str, ...], list[dict[str, str]]]:
    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    return grouped


def action_name(action: str) -> str:
    return action.rsplit(".", maxsplit=1)[-1] if action else ""


def final_rows(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    by_agent = group_by(rows, "episode_id", "agent_id")
    finals = {}
    for key, agent_rows in by_agent.items():
        finals[key] = max(agent_rows, key=lambda row: int(row["env_time"]))
    return finals


def last_position_stats(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, object]]:
    by_agent = group_by(rows, "episode_id", "agent_id")
    stats = {}
    for key, agent_rows in by_agent.items():
        sorted_rows = sorted(agent_rows, key=lambda row: int(row["env_time"]))
        last_position = ""
        last_change_time = 0
        previous_position = None
        for row in sorted_rows:
            position = row["position"]
            if position:
                last_position = position
            if position != previous_position:
                last_change_time = int(row["env_time"])
                previous_position = position
        final_time = int(sorted_rows[-1]["env_time"]) if sorted_rows else 0
        stats[key] = {
            "last_position": last_position,
            "last_change_time": last_change_time,
            "stationary_tail": max(0, final_time - last_change_time),
        }
    return stats


def action_stats(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, object]]:
    by_agent = group_by(rows, "episode_id", "agent_id")
    stats = {}
    for key, agent_rows in by_agent.items():
        names = [action_name(row["action"]) for row in agent_rows]
        counts = Counter(names)
        stats[key] = {
            "counts": counts,
            "tail": names[-12:],
        }
    return stats


def select_episodes(rows: list[dict[str, str]], reward_below: float, success_below: float) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        reward = float(row["normalized_reward"])
        success = float(row["success_rate"])
        if reward < reward_below or success < success_below:
            selected.append(row)
    return selected


def print_episode_report(
    episode: dict[str, str],
    agent_rows: list[dict[str, str]],
    final_info: dict[tuple[str, str], dict[str, str]],
    position_info: dict[tuple[str, str], dict[str, object]],
    actions_info: dict[tuple[str, str], dict[str, object]],
) -> None:
    episode_id = episode["episode_id"]
    print(
        f"\n== {episode_id} =="
        f"\nsummary: env_time={episode['env_time']}, "
        f"success_rate={episode['success_rate']}, "
        f"normalized_reward={episode['normalized_reward']}"
    )

    failed_agents = []
    for row in agent_rows:
        key = (episode_id, row["agent_id"])
        final = final_info.get(key, {})
        if final.get("state") != "6":
            failed_agents.append(row)

    if not failed_agents:
        print("all agents reached DONE; low reward is likely delay/deadline related")
        return

    print("failed agents:")
    for row in failed_agents:
        agent_id = row["agent_id"]
        key = (episode_id, agent_id)
        final = final_info.get(key, {})
        pos = position_info.get(key, {})
        act = actions_info.get(key, {})
        counts = act.get("counts", Counter())
        tail = " ".join(act.get("tail", []))
        state = STATE_NAMES.get(final.get("state", ""), final.get("state", "unknown"))
        latest_arrival = row.get("latest_arrival", "")
        final_time = final.get("env_time", "")
        missed_by = ""
        if latest_arrival and final_time:
            missed_by = str(max(0, int(final_time) - int(float(latest_arrival))))

        print(
            f"- agent {agent_id}: state={state}, final_time={final_time}, "
            f"latest_arrival={latest_arrival}, missed_by={missed_by}, "
            f"last_position={pos.get('last_position', '')}, "
            f"stationary_tail={pos.get('stationary_tail', 0)}, "
            f"actions={{F:{counts.get('MOVE_FORWARD', 0)}, "
            f"L:{counts.get('MOVE_LEFT', 0)}, R:{counts.get('MOVE_RIGHT', 0)}, "
            f"S:{counts.get('STOP_MOVING', 0)}, N:{counts.get('DO_NOTHING', 0)}}}, "
            f"tail=[{tail}]"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize failed or low-reward Flatland debug episodes."
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        required=True,
        help="Directory produced by flatland-trajectory-analysis.",
    )
    parser.add_argument("--reward-below", type=float, default=1.0)
    parser.add_argument("--success-below", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    analysis_dir = args.analysis_dir

    arrived = read_csv(analysis_dir / "all_trains_arrived.csv")
    agent_stats_rows = read_csv(analysis_dir / "agent_stats.csv")
    final_info = final_rows(read_csv(analysis_dir / "all_trains_rewards_dones_infos.csv"))
    position_info = last_position_stats(read_csv(analysis_dir / "all_trains_positions.csv"))
    actions_info = action_stats(read_csv(analysis_dir / "all_actions.csv"))
    agents_by_episode = group_by(agent_stats_rows, "episode_id")

    selected = select_episodes(arrived, args.reward_below, args.success_below)
    if not selected:
        print("No low-reward or incomplete episodes selected.")
        return 0

    for episode in selected:
        print_episode_report(
            episode,
            agents_by_episode.get((episode["episode_id"],), []),
            final_info,
            position_info,
            actions_info,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
