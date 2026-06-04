#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tools.analyze_policy_action_diffs import (
    action_name,
    corridor_len,
    direction_features,
    distance_and_slack,
    done_all,
    future_risk,
    instantiate_policy,
    logit_features,
    mask_values,
    observation_list,
    observation_scalar_features,
    planned_prefixes,
    policy_actions,
    prefix_conflict_features,
    raw_logits,
    route_prefix_for_action,
    state_name,
    target_metrics,
)
from tools.evaluate_sampled import (
    DEFAULT_BASE_STATE,
    DEFAULT_OBS_BUILDER,
    DEFAULT_REWARDS,
    failed_agent_details,
    load_sampling_env_generator,
    load_symbol,
    normalized_reward,
    repo_root,
)
from flatland.envs.persistence import RailEnvPersister
from tools.train_counterfactual_gate import outcome_category


MOVE_ACTIONS = (1, 2, 3)
DEFAULT_ACTIONS = (1, 2, 3, 4)


def make_env(args: argparse.Namespace) -> tuple[Any, Any]:
    obs_builder = load_symbol(args.obs_builder)()
    rewards = load_symbol(args.rewards)()
    env, _ = RailEnvPersister.load_new(
        args.base_state_pkl,
        obs_builder=obs_builder,
        rewards=rewards,
    )
    if args.num_agents is not None:
        env.number_of_agents = args.num_agents
    env = load_sampling_env_generator()(env, line_length=args.line_length, scene=args.scene)
    return env, obs_builder


def final_result(
    args: argparse.Namespace,
    seed: int,
    env: Any,
    reward_values: list[float],
    positions: dict[int, list[Any]],
    actions_by_agent: dict[int, list[int]],
) -> dict[str, Any]:
    failed_agents = failed_agent_details(env, positions, actions_by_agent)
    success_rate = sum(int(agent.state == 6) for agent in env.agents) / env.get_num_agents()
    return {
        "seed": seed,
        "env_time": int(env._elapsed_steps),
        "success_rate": success_rate,
        "normalized_reward": normalized_reward(env, reward_values),
        "failed_agents": failed_agents,
        "failed_agent_ids": ",".join(str(agent["agent_id"]) for agent in failed_agents),
        "num_agents": env.get_num_agents(),
        "max_episode_steps": env._max_episode_steps,
        "line_length": args.line_length,
        "scene": args.scene or "scene_5",
    }


def valid_actions_from_mask(
    observation: Any,
    baseline_action: int,
    include_do_nothing: bool,
    forced_actions: tuple[int, ...] | None,
) -> list[int]:
    values = np.asarray(observation, dtype=np.float32)
    mask = values[-5:] if values.shape[0] >= 5 else np.ones(5, dtype=np.float32)
    if forced_actions is not None:
        action_pool = forced_actions
    else:
        action_pool = (0, *DEFAULT_ACTIONS) if include_do_nothing else DEFAULT_ACTIONS
    return [
        action
        for action in action_pool
        if action != baseline_action and action < len(mask) and mask[action] >= 0.5
    ]


def has_opposing_ahead(observation: Any) -> bool:
    values = np.asarray(observation, dtype=np.float32)
    return bool(values.shape[0] >= 18 and np.max(values[14:18]) >= 0.5)


def has_same_direction_ahead(observation: Any) -> bool:
    values = np.asarray(observation, dtype=np.float32)
    return bool(values.shape[0] >= 22 and np.max(values[18:22]) >= 0.5)


def is_critical_decision(
    args: argparse.Namespace,
    observation: Any,
    baseline_action: int,
    baseline_risk: float,
    baseline_corridor_len: int,
    alternatives: list[int],
) -> bool:
    if not args.critical_only:
        return True
    if not alternatives:
        return False
    values = np.asarray(observation, dtype=np.float32)
    on_switch = values.shape[0] >= 10 and bool(np.max(values[7:10]) >= 0.5)
    if on_switch:
        return True
    if baseline_risk > 0.0:
        return True
    if baseline_corridor_len >= args.min_corridor_len:
        return True
    if baseline_action == 4 and any(action in MOVE_ACTIONS for action in alternatives):
        return True
    if has_opposing_ahead(observation) or has_same_direction_ahead(observation):
        return True
    return False


def forced_action_score(
    obs_builder: Any,
    handle: int,
    baseline_target: dict[str, Any],
    forced_action: int,
    baseline_risk: float,
    forced_risk: float,
) -> tuple[float, float, float, int]:
    forced_target = target_metrics(obs_builder, handle, forced_action)
    forced_distance = float(forced_target["target_distance"])
    baseline_distance = float(baseline_target["target_distance"])
    distance_delta = forced_distance - baseline_distance
    occupied_penalty = 1 if forced_target["target_occupied_by_other"] else 0
    unreachable_penalty = 1 if not np.isfinite(forced_distance) else 0
    risk_delta = forced_risk - baseline_risk
    action_order = {2: 0, 1: 1, 3: 2, 4: 3, 0: 4}.get(forced_action, 9)
    return (
        float(unreachable_penalty),
        float(occupied_penalty),
        float(risk_delta),
        float(distance_delta),
        action_order,
    )


def decision_row(
    args: argparse.Namespace,
    seed: int,
    decision_index: int,
    env: Any,
    obs_builder: Any,
    policy: Any,
    handle: int,
    observation: Any,
    baseline_actions: dict[int, int],
    forced_action: int,
    baseline_prefixes: dict[int, Any],
) -> dict[str, Any]:
    agent = env.agents[handle]
    baseline_action = baseline_actions[handle]
    distance, slack = distance_and_slack(obs_builder, handle)
    baseline_target = target_metrics(obs_builder, handle, baseline_action)
    forced_target = target_metrics(obs_builder, handle, forced_action)
    logits = raw_logits(policy, observation)
    baseline_risk = future_risk(
        policy,
        obs_builder,
        handle,
        baseline_action,
        baseline_prefixes,
    )
    forced_risk = future_risk(
        policy,
        obs_builder,
        handle,
        forced_action,
        baseline_prefixes,
    )
    baseline_prefix = route_prefix_for_action(
        policy,
        obs_builder,
        handle,
        baseline_action,
    )
    forced_prefix = route_prefix_for_action(
        policy,
        obs_builder,
        handle,
        forced_action,
    )
    baseline_prefix_features = prefix_conflict_features(
        "baseline",
        baseline_prefix,
        baseline_prefixes,
        handle,
        env,
    )
    forced_prefix_features = prefix_conflict_features(
        "forced",
        forced_prefix,
        baseline_prefixes,
        handle,
        env,
    )
    prefix_delta_features = {}
    for suffix in (
        "prefix_len",
        "prefix_conflict_agents",
        "prefix_cell_intersections",
        "prefix_first_intersection_step",
        "prefix_min_intersection_eta_gap",
        "prefix_same_direction_intersections",
        "prefix_opposing_direction_intersections",
        "prefix_crossing_direction_intersections",
        "prefix_same_edge_conflicts",
        "prefix_head_on_edge_conflicts",
        "prefix_min_head_on_eta_gap",
        "prefix_min_own_deadline_slack",
        "prefix_min_other_deadline_slack",
        "prefix_min_pair_deadline_slack",
        "prefix_own_deadline_miss_conflicts",
        "prefix_other_deadline_miss_conflicts",
        "prefix_pair_deadline_miss_conflicts",
        "prefix_own_tight_deadline_conflicts",
        "prefix_other_tight_deadline_conflicts",
        "prefix_pair_tight_deadline_conflicts",
    ):
        prefix_delta_features[f"forced_{suffix}_delta"] = (
            forced_prefix_features[f"forced_{suffix}"]
            - baseline_prefix_features[f"baseline_{suffix}"]
        )
    return {
        "seed": seed,
        "decision_index": decision_index,
        "env_time": int(env._elapsed_steps),
        "agent_id": int(handle),
        "state": state_name(agent.state),
        "position": str(agent.position),
        "direction": agent.direction,
        "speed": float(agent.speed_counter.speed),
        "distance": distance,
        "slack": slack,
        **observation_scalar_features(observation),
        **mask_values(observation),
        "baseline_action": baseline_action,
        "baseline_action_name": action_name(baseline_action),
        "forced_action": forced_action,
        "forced_action_name": action_name(forced_action),
        "baseline_target": baseline_target["target"],
        "baseline_target_direction": baseline_target["target_direction"],
        "baseline_target_distance": baseline_target["target_distance"],
        "baseline_target_occupied_by_other": baseline_target["target_occupied_by_other"],
        **direction_features(
            observation,
            "baseline_direction",
            baseline_target["target_direction"],
        ),
        "forced_target": forced_target["target"],
        "forced_target_direction": forced_target["target_direction"],
        "forced_target_distance": forced_target["target_distance"],
        "forced_target_occupied_by_other": forced_target["target_occupied_by_other"],
        **direction_features(
            observation,
            "forced_direction",
            forced_target["target_direction"],
        ),
        "forced_distance_delta": (
            forced_target["target_distance"] - baseline_target["target_distance"]
        ),
        **logit_features("policy", logits, baseline_action, forced_action),
        "baseline_corridor_len": corridor_len(obs_builder, handle, baseline_action),
        "forced_corridor_len": corridor_len(obs_builder, handle, forced_action),
        "baseline_future_head_on_risk": baseline_risk,
        "forced_future_head_on_risk": forced_risk,
        "future_head_on_risk_delta": forced_risk - baseline_risk,
        **baseline_prefix_features,
        **forced_prefix_features,
        **prefix_delta_features,
    }


def collect_decisions(
    args: argparse.Namespace,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env, obs_builder = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    policy = instantiate_policy(args.policy, args.policy_checkpoint)
    reward_values: list[float] = []
    positions: dict[int, list[Any]] = defaultdict(list)
    actions_by_agent: dict[int, list[int]] = defaultdict(list)
    rows: list[dict[str, Any]] = []
    decision_count = 0
    focus_handles = focused_handles_for_seed(args, seed)

    while int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
        observations_by_handle = dict(zip(handles, obs_list))
        actions = policy_actions(policy, handles, obs_list)
        prefixes = planned_prefixes(policy, obs_builder, actions)

        for handle in handles:
            if len(rows) >= args.max_decisions_per_seed * args.max_alternatives_per_decision:
                break
            if focus_handles is not None and handle not in focus_handles:
                continue
            focus_window = focused_window_for_seed_handle(args, seed, handle)
            if focus_window is not None:
                focus_start, focus_end = focus_window
                if not focus_start <= int(env._elapsed_steps) <= focus_end:
                    continue
            if handle not in actions:
                continue
            observation = observations_by_handle[handle]
            baseline_action = int(actions[handle])
            alternatives = valid_actions_from_mask(
                observation,
                baseline_action,
                args.include_do_nothing,
                args.forced_action_ids,
            )
            baseline_risk = future_risk(
                policy,
                obs_builder,
                handle,
                baseline_action,
                prefixes,
            )
            baseline_corridor_len = corridor_len(obs_builder, handle, baseline_action)
            if not is_critical_decision(
                args,
                observation,
                baseline_action,
                baseline_risk,
                baseline_corridor_len,
                alternatives,
            ):
                continue

            baseline_target = target_metrics(obs_builder, handle, baseline_action)
            alternatives = sorted(
                alternatives,
                key=lambda action: forced_action_score(
                    obs_builder,
                    handle,
                    baseline_target,
                    action,
                    baseline_risk,
                    future_risk(policy, obs_builder, handle, action, prefixes),
                ),
            )
            decision_count += 1
            for forced_action in alternatives[: args.max_alternatives_per_decision]:
                rows.append(
                    decision_row(
                        args,
                        seed,
                        decision_count,
                        env,
                        obs_builder,
                        policy,
                        handle,
                        observation,
                        actions,
                        forced_action,
                        prefixes,
                    )
                )

        observations, rewards_by_agent, dones, _ = env.step(actions)
        for handle, action in actions.items():
            actions_by_agent[handle].append(action)
        for handle in handles:
            positions[handle].append(env.agents[handle].position)
        reward_values.extend(float(rewards_by_agent.get(handle, 0.0)) for handle in handles)
        if done_all(dones):
            break

    result = final_result(args, seed, env, reward_values, positions, actions_by_agent)
    return result, rows


def run_forced_episode(
    args: argparse.Namespace,
    seed: int,
    event: dict[str, Any],
) -> dict[str, Any]:
    env, _ = make_env(args)
    observations, _ = env.reset(random_seed=seed)
    policy = instantiate_policy(args.policy, args.policy_checkpoint)
    reward_values: list[float] = []
    positions: dict[int, list[Any]] = defaultdict(list)
    actions_by_agent: dict[int, list[int]] = defaultdict(list)
    force_step = int(event["env_time"])
    force_handle = int(event["agent_id"])
    force_action = int(event["forced_action"])
    forced_applied = False

    while int(env._elapsed_steps) < env._max_episode_steps:
        handles = list(env.get_agent_handles())
        obs_list = observation_list(observations, handles)
        actions = policy_actions(policy, handles, obs_list)
        if int(env._elapsed_steps) == force_step and force_handle in actions:
            actions[force_handle] = force_action
            forced_applied = True

        observations, rewards_by_agent, dones, _ = env.step(actions)
        for handle, action in actions.items():
            actions_by_agent[handle].append(action)
        for handle in handles:
            positions[handle].append(env.agents[handle].position)
        reward_values.extend(float(rewards_by_agent.get(handle, 0.0)) for handle in handles)
        if done_all(dones):
            break

    result = final_result(args, seed, env, reward_values, positions, actions_by_agent)
    result["forced_applied"] = forced_applied
    return result


def annotate_outcome(
    row: dict[str, Any],
    baseline: dict[str, Any],
    forced: dict[str, Any],
) -> dict[str, Any]:
    baseline_failed = len(baseline["failed_agents"])
    forced_failed = len(forced["failed_agents"])
    return {
        **row,
        "baseline_reward": baseline["normalized_reward"],
        "forced_reward": forced["normalized_reward"],
        "reward_delta": forced["normalized_reward"] - baseline["normalized_reward"],
        "baseline_success": baseline["success_rate"],
        "forced_success": forced["success_rate"],
        "success_delta": forced["success_rate"] - baseline["success_rate"],
        "baseline_env_time": baseline["env_time"],
        "forced_env_time": forced["env_time"],
        "env_time_delta": forced["env_time"] - baseline["env_time"],
        "baseline_failed_agents": baseline_failed,
        "forced_failed_agents": forced_failed,
        "failed_agents_delta": forced_failed - baseline_failed,
        "baseline_failed_agent_ids": baseline["failed_agent_ids"],
        "forced_failed_agent_ids": forced["failed_agent_ids"],
        "forced_applied": forced["forced_applied"],
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "reward_wins": sum(int(row["reward_delta"] > 1e-9) for row in rows),
        "reward_losses": sum(int(row["reward_delta"] < -1e-9) for row in rows),
        "reward_ties": sum(int(abs(row["reward_delta"]) <= 1e-9) for row in rows),
        "success_wins": sum(int(row["success_delta"] > 1e-9) for row in rows),
        "success_losses": sum(int(row["success_delta"] < -1e-9) for row in rows),
        "success_ties": sum(int(abs(row["success_delta"]) <= 1e-9) for row in rows),
        "forced_not_applied": sum(int(not row["forced_applied"]) for row in rows),
    }


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one-step counterfactual actions at critical policy decisions."
        )
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--policy", default="submission.rerank_policy.MyPolicy")
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--obs-builder", default=DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=DEFAULT_REWARDS)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument(
        "--seeds",
        help="Comma-separated exact seed list. Overrides --seed/--episodes when set.",
    )
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--max-decisions-per-seed", type=int, default=8)
    parser.add_argument("--max-alternatives-per-decision", type=int, default=2)
    parser.add_argument("--min-corridor-len", type=int, default=4)
    parser.add_argument(
        "--focus-agent-ids",
        help="Comma-separated agent ids to sample decisions from for every seed.",
    )
    parser.add_argument(
        "--focus-failures-json",
        type=Path,
        help=(
            "JSON from tools/mine_failure_seeds.py. When present, sample only "
            "the failed agents for each seed that appears in the file."
        ),
    )
    parser.add_argument(
        "--focus-counterfactual-csv",
        nargs="+",
        type=Path,
        help=(
            "CSV rows from previous counterfactual runs. Sample only the "
            "seed/agent windows around rows whose outcome category matches "
            "--focus-counterfactual-categories."
        ),
    )
    parser.add_argument(
        "--focus-counterfactual-categories",
        default="good,bad",
        help="Comma-separated outcome categories to focus from the CSV: good,neutral,bad.",
    )
    parser.add_argument(
        "--focus-counterfactual-window-before",
        type=int,
        default=20,
        help="Steps before each selected counterfactual event to include.",
    )
    parser.add_argument(
        "--focus-counterfactual-window-after",
        type=int,
        default=20,
        help="Steps after each selected counterfactual event to include.",
    )
    parser.add_argument(
        "--focus-window-before-stationary",
        type=int,
        help=(
            "With --focus-failures-json, sample only decisions within this many "
            "steps before each failed agent's last observed movement."
        ),
    )
    parser.add_argument(
        "--focus-window-before-deadline",
        type=int,
        help=(
            "With --focus-failures-json, sample only decisions within this many "
            "steps before each failed agent's latest_arrival deadline. This "
            "takes precedence over --focus-window-before-stationary."
        ),
    )
    parser.add_argument(
        "--focus-window-after-deadline",
        type=int,
        default=30,
        help=(
            "With --focus-window-before-deadline, also keep this many steps "
            "after each failed agent's latest_arrival deadline."
        ),
    )
    parser.add_argument(
        "--focus-window-after-stationary",
        type=int,
        default=30,
        help=(
            "With --focus-window-before-stationary, also keep this many steps "
            "after each failed agent's last observed movement."
        ),
    )
    parser.add_argument("--critical-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--include-do-nothing", action="store_true")
    parser.add_argument(
        "--forced-actions",
        help=(
            "Comma-separated action IDs or names to consider as alternatives, "
            "for example 1,2,3 or LEFT,FORWARD,RIGHT."
        ),
    )
    parser.add_argument("--output-csv", type=Path)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    args.forced_action_ids = parse_forced_actions(args.forced_actions)
    args.focus_agent_ids = parse_focus_agent_ids(args.focus_agent_ids)
    args.focus_handles_by_seed, args.focus_windows_by_seed = load_focus_failures(
        args.focus_failures_json,
        args.focus_window_before_stationary,
        args.focus_window_after_stationary,
        args.focus_window_before_deadline,
        args.focus_window_after_deadline,
    )
    merge_focus(
        args.focus_handles_by_seed,
        args.focus_windows_by_seed,
        *load_focus_counterfactuals(
            args.focus_counterfactual_csv,
            args.focus_counterfactual_categories,
            args.focus_counterfactual_window_before,
            args.focus_counterfactual_window_after,
        ),
    )
    return args


def parse_focus_agent_ids(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item) for item in value.split(",") if item.strip()}


def load_focus_failures(
    path: Path | None,
    window_before_stationary: int | None,
    window_after_stationary: int,
    window_before_deadline: int | None,
    window_after_deadline: int,
) -> tuple[dict[int, set[int]], dict[tuple[int, int], tuple[int, int]]]:
    if path is None:
        return {}, {}
    with path.open() as handle:
        payload = json.load(handle)
    rows = payload.get("details", payload.get("rows", []))
    focus: dict[int, set[int]] = {}
    windows: dict[tuple[int, int], tuple[int, int]] = {}
    for row in rows:
        seed = int(row["seed"])
        env_time = int(row.get("env_time", 0))
        failed_agents = row.get("failed_agents", [])
        handles = {
            int(agent["agent_id"])
            for agent in failed_agents
            if "agent_id" in agent
        }
        if handles:
            focus[seed] = handles
        if window_before_deadline is None and window_before_stationary is None:
            continue
        for agent in failed_agents:
            if "agent_id" not in agent:
                continue
            handle = int(agent["agent_id"])
            missed_by = agent.get("missed_by")
            if window_before_deadline is not None and missed_by is not None:
                deadline_step = max(0, env_time - int(missed_by))
                windows[(seed, handle)] = (
                    max(0, deadline_step - window_before_deadline),
                    max(0, deadline_step + window_after_deadline),
                )
                continue
            if window_before_stationary is None:
                continue
            stationary_tail = int(agent.get("stationary_tail") or 0)
            last_progress_step = max(0, env_time - stationary_tail)
            windows[(seed, handle)] = (
                max(0, last_progress_step - window_before_stationary),
                max(0, last_progress_step + window_after_stationary),
            )
    return focus, windows


def merge_focus(
    focus: dict[int, set[int]],
    windows: dict[tuple[int, int], tuple[int, int]],
    new_focus: dict[int, set[int]],
    new_windows: dict[tuple[int, int], tuple[int, int]],
) -> None:
    for seed, handles in new_focus.items():
        focus.setdefault(seed, set()).update(handles)
    for key, window in new_windows.items():
        if key not in windows:
            windows[key] = window
            continue
        current_start, current_end = windows[key]
        new_start, new_end = window
        windows[key] = (min(current_start, new_start), max(current_end, new_end))


def load_focus_counterfactuals(
    paths: list[Path] | None,
    categories_value: str,
    window_before: int,
    window_after: int,
) -> tuple[dict[int, set[int]], dict[tuple[int, int], tuple[int, int]]]:
    if not paths:
        return {}, {}
    categories = {
        item.strip().lower()
        for item in categories_value.split(",")
        if item.strip()
    }
    valid_categories = {"good", "neutral", "bad"}
    unknown = categories - valid_categories
    if unknown:
        raise ValueError(
            "--focus-counterfactual-categories contains unknown values: "
            f"{','.join(sorted(unknown))}"
        )

    focus: dict[int, set[int]] = {}
    windows: dict[tuple[int, int], tuple[int, int]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                category = outcome_category(row, reward_epsilon=1e-6)
                if category not in categories:
                    continue
                seed = int(float(row["seed"]))
                handle_id = int(float(row["agent_id"]))
                env_time = int(float(row["env_time"]))
                focus.setdefault(seed, set()).add(handle_id)
                key = (seed, handle_id)
                window = (
                    max(0, env_time - window_before),
                    max(0, env_time + window_after),
                )
                if key not in windows:
                    windows[key] = window
                else:
                    current_start, current_end = windows[key]
                    windows[key] = (
                        min(current_start, window[0]),
                        max(current_end, window[1]),
                    )
    return focus, windows


def focused_handles_for_seed(args: argparse.Namespace, seed: int) -> set[int] | None:
    if args.focus_agent_ids is not None:
        return args.focus_agent_ids
    return args.focus_handles_by_seed.get(seed)


def focused_window_for_seed_handle(
    args: argparse.Namespace,
    seed: int,
    handle: int,
) -> tuple[int, int] | None:
    return args.focus_windows_by_seed.get((seed, handle))


def parse_forced_actions(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    mapping = {
        "N": 0,
        "DO_NOTHING": 0,
        "LEFT": 1,
        "MOVE_LEFT": 1,
        "L": 1,
        "FORWARD": 2,
        "MOVE_FORWARD": 2,
        "F": 2,
        "RIGHT": 3,
        "MOVE_RIGHT": 3,
        "R": 3,
        "STOP": 4,
        "STOP_MOVING": 4,
        "S": 4,
    }
    actions = []
    for item in value.split(","):
        token = item.strip().upper()
        if not token:
            continue
        if token.isdigit():
            action = int(token)
        else:
            if token not in mapping:
                raise argparse.ArgumentTypeError(f"Unknown action token: {item}")
            action = mapping[token]
        if action < 0 or action > 4:
            raise argparse.ArgumentTypeError(f"Action out of range 0..4: {item}")
        actions.append(action)
    return tuple(dict.fromkeys(actions))


def main() -> int:
    args = parse_args()
    if args.seeds:
        seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    elif args.focus_counterfactual_csv and args.focus_handles_by_seed:
        seeds = sorted(args.focus_handles_by_seed)
    else:
        seeds = [args.seed + index for index in range(args.episodes)]
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        baseline, events = collect_decisions(args, seed)
        print(
            f"seed={seed} baseline={baseline['normalized_reward']:.6g}/"
            f"{baseline['success_rate']:.6g} events={len(events)}",
            flush=True,
        )
        for event in events:
            forced = run_forced_episode(args, seed, event)
            row = annotate_outcome(event, baseline, forced)
            rows.append(row)
            print(
                "  step={env_time} agent={agent_id} "
                "{baseline_action_name}->{forced_action_name} "
                "reward_delta={reward_delta:.6g} success_delta={success_delta:.6g} "
                "failed_delta={failed_agents_delta}".format(**row),
                flush=True,
            )

    summary = summarize(rows)
    print(
        "\nSummary: "
        f"rows={summary['rows']} "
        f"reward={summary['reward_wins']}/{summary['reward_losses']}/"
        f"{summary['reward_ties']} "
        f"success={summary['success_wins']}/{summary['success_losses']}/"
        f"{summary['success_ties']} "
        f"forced_not_applied={summary['forced_not_applied']}"
    )
    if args.output_csv is not None:
        write_csv(args.output_csv, rows)
    if args.output_json is not None:
        write_json(args.output_json, rows, summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
