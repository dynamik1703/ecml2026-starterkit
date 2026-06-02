#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_count_nonzero
from flatland.envs.persistence import RailEnvPersister
from flatland.envs.rewards import ECML2026Rewards

from submission.my_observation_builder import MyObservationBuilder


DEFAULT_BASE_STATE = "reinforcement-learning/sampling/level_0_scenario_1.pkl"


@dataclass(frozen=True)
class RouteVariant:
    agent: int
    name: str
    nodes: tuple[tuple[tuple[int, int], int], ...]

    @property
    def length(self) -> int:
        return max(0, len(self.nodes) - 1)

    @property
    def reached_target(self) -> bool:
        return bool(self.nodes)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_sampling_env_generator() -> Any:
    path = repo_root() / "reinforcement-learning" / "sampling" / "sampling_env_generator.py"
    spec = importlib.util.spec_from_file_location("ecml_sampling_env_generator", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load sampling generator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sampling_env_generator


def make_env(args: argparse.Namespace) -> tuple[Any, MyObservationBuilder]:
    obs_builder = MyObservationBuilder()
    env, _ = RailEnvPersister.load_new(
        args.base_state_pkl,
        obs_builder=obs_builder,
        rewards=ECML2026Rewards(),
    )
    if args.num_agents is not None:
        env.number_of_agents = args.num_agents
    env = load_sampling_env_generator()(
        env,
        line_length=args.line_length,
        scene=args.scene,
    )
    env.reset(random_seed=args.seed)
    return env, obs_builder


def route_target(obs_builder: MyObservationBuilder, handle: int) -> tuple[int, int]:
    return obs_builder._get_next_waypoint_target(handle)


def follow_route(
    obs_builder: MyObservationBuilder,
    handle: int,
    start_position: tuple[int, int],
    start_direction: int,
    max_steps: int,
    forced_first_direction: int | None = None,
) -> list[tuple[tuple[int, int], int]]:
    env = obs_builder.env
    distance_map = obs_builder._get_distance_map(handle)
    target = route_target(obs_builder, handle)
    current_position = start_position
    current_direction = start_direction
    nodes = [(current_position, current_direction)]
    visited = {(current_position, current_direction)}

    for step in range(1, max_steps + 1):
        if current_position == target:
            return nodes

        transitions = env.rail.get_transitions((current_position, current_direction))
        if forced_first_direction is not None and step == 1:
            next_direction = forced_first_direction
            if not transitions[next_direction]:
                return []
        else:
            next_direction = obs_builder._best_progress_direction(
                transitions,
                current_position,
                distance_map,
            )
            if next_direction is None:
                return []

        next_position = get_new_position(current_position, next_direction)
        if not obs_builder._is_in_bounds(next_position):
            return []
        state = (next_position, next_direction)
        if state in visited:
            return []
        visited.add(state)
        nodes.append(state)
        current_position = next_position
        current_direction = next_direction

    return nodes if current_position == target else []


def dedupe_variants(variants: list[RouteVariant], limit: int) -> list[RouteVariant]:
    seen = set()
    unique = []
    for variant in sorted(variants, key=lambda item: item.length):
        key = variant.nodes
        if key in seen:
            continue
        seen.add(key)
        unique.append(variant)
        if len(unique) >= limit:
            break
    return unique


def route_variants(
    obs_builder: MyObservationBuilder,
    handle: int,
    max_variants: int,
    max_steps: int,
) -> list[RouteVariant]:
    agent = obs_builder.env.agents[handle]
    start_position = agent.initial_position
    start_direction = agent.initial_direction
    if start_position is None or start_direction is None:
        return []

    baseline_nodes = follow_route(
        obs_builder,
        handle,
        start_position,
        start_direction,
        max_steps,
    )
    if not baseline_nodes:
        return []

    variants = [
        RouteVariant(
            agent=handle,
            name="greedy",
            nodes=tuple(baseline_nodes),
        )
    ]
    distance_map = obs_builder._get_distance_map(handle)
    for index, (position, direction) in enumerate(baseline_nodes[:-1]):
        transitions = obs_builder.env.rail.get_transitions((position, direction))
        if fast_count_nonzero(transitions) <= 1:
            continue
        greedy_next_direction = baseline_nodes[index + 1][1]
        for alt_direction, is_open in enumerate(transitions):
            if not is_open or alt_direction == greedy_next_direction:
                continue
            alt_position = get_new_position(position, alt_direction)
            if not obs_builder._is_in_bounds(alt_position):
                continue
            alt_distance = distance_map[alt_position[0], alt_position[1], alt_direction]
            if not np.isfinite(alt_distance):
                continue
            continuation = follow_route(
                obs_builder,
                handle,
                position,
                direction,
                max_steps - index,
                forced_first_direction=alt_direction,
            )
            if not continuation:
                continue
            nodes = baseline_nodes[:index] + continuation
            variants.append(
                RouteVariant(
                    agent=handle,
                    name=f"switch{index}:dir{alt_direction}",
                    nodes=tuple(nodes),
                )
            )

    return dedupe_variants(variants, max_variants)


def combo_metrics(
    env: Any,
    combo: tuple[RouteVariant, ...],
    start_delays: dict[int, int] | None = None,
) -> dict[str, Any]:
    start_delays = start_delays or {}
    position_occupancy: dict[tuple[int, tuple[int, int]], list[int]] = {}
    edge_occupancy: dict[
        tuple[int, tuple[int, int] | None, tuple[int, int]],
        list[int],
    ] = {}
    arrivals = {}
    lateness = {}
    total_length = 0

    for variant in combo:
        agent = env.agents[variant.agent]
        departure = int(agent.earliest_departure or 0) + int(
            start_delays.get(variant.agent, 0)
        )
        total_length += variant.length
        arrivals[variant.agent] = departure + variant.length
        lateness[variant.agent] = max(
            0,
            arrivals[variant.agent] - int(agent.latest_arrival or arrivals[variant.agent]),
        )
        for step, (position, _) in enumerate(variant.nodes):
            time = departure + step
            position_occupancy.setdefault((time, position), []).append(variant.agent)
            if step > 0:
                previous_position = variant.nodes[step - 1][0]
                edge_occupancy.setdefault((time, previous_position, position), []).append(
                    variant.agent
                )

    position_conflicts = [
        {"time": time, "position": position, "agents": agents}
        for (time, position), agents in position_occupancy.items()
        if len(agents) > 1
    ]
    edge_conflicts = []
    seen_edges = set()
    for (time, source, target), agents in edge_occupancy.items():
        reverse_key = (time, target, source)
        if reverse_key in seen_edges:
            continue
        reverse_agents = edge_occupancy.get(reverse_key, [])
        if reverse_agents:
            edge_conflicts.append(
                {
                    "time": time,
                    "edge": [source, target],
                    "agents": agents + reverse_agents,
                }
            )
        seen_edges.add((time, source, target))

    late_agents = {agent: value for agent, value in lateness.items() if value > 0}
    score = (
        100000 * len(position_conflicts)
        + 100000 * len(edge_conflicts)
        + 1000 * len(late_agents)
        + 10 * sum(late_agents.values())
        + total_length
    )
    return {
        "score": score,
        "position_conflicts": position_conflicts,
        "edge_conflicts": edge_conflicts,
        "late_agents": late_agents,
        "arrivals": arrivals,
        "start_delays": dict(sorted(start_delays.items())),
        "total_length": total_length,
    }


def first_conflict(metrics: dict[str, Any]) -> dict[str, Any] | None:
    conflicts = []
    for conflict in metrics["position_conflicts"]:
        conflicts.append(
            {
                "type": "position",
                "time": conflict["time"],
                "agents": conflict["agents"],
            }
        )
    for conflict in metrics["edge_conflicts"]:
        conflicts.append(
            {
                "type": "edge",
                "time": conflict["time"],
                "agents": conflict["agents"],
            }
        )
    if not conflicts:
        return None
    return min(conflicts, key=lambda item: item["time"])


def conflict_count(metrics: dict[str, Any]) -> int:
    return len(metrics["position_conflicts"]) + len(metrics["edge_conflicts"])


def schedule_search_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        conflict_count(metrics),
        len(metrics["late_agents"]),
        sum(metrics["late_agents"].values()),
        sum(metrics["start_delays"].values()),
    )


def best_delay_move(
    env: Any,
    combo: tuple[RouteVariant, ...],
    agents: list[int],
    start_delays: dict[int, int],
    delay_window: int,
) -> tuple[dict[int, int], dict[str, Any]]:
    best_delays = None
    best_metrics = None
    current_metrics = combo_metrics(env, combo, start_delays)
    current_key = schedule_search_key(current_metrics)
    for handle in agents:
        for delta in range(1, delay_window + 1):
            candidate_delays = dict(start_delays)
            candidate_delays[handle] = candidate_delays.get(handle, 0) + delta
            metrics = combo_metrics(env, combo, candidate_delays)
            if (
                best_metrics is None
                or schedule_search_key(metrics) < schedule_search_key(best_metrics)
            ):
                best_delays = candidate_delays
                best_metrics = metrics

    if best_metrics is None or schedule_search_key(best_metrics) >= current_key:
        return start_delays, current_metrics

    assert best_delays is not None
    return best_delays, best_metrics


def schedule_combo(
    env: Any,
    combo: tuple[RouteVariant, ...],
    max_iterations: int,
    delay_window: int,
) -> dict[str, Any]:
    start_delays = {variant.agent: 0 for variant in combo}
    metrics = combo_metrics(env, combo, start_delays)
    iterations = 0
    while iterations < max_iterations:
        conflict = first_conflict(metrics)
        if conflict is None:
            break
        previous_key = schedule_search_key(metrics)
        start_delays, metrics = best_delay_move(
            env,
            combo,
            list(dict.fromkeys(conflict["agents"])),
            start_delays,
            delay_window,
        )
        iterations += 1
        if schedule_search_key(metrics) >= previous_key:
            break

    metrics["schedule_iterations"] = iterations
    metrics["schedule_resolved"] = first_conflict(metrics) is None
    return metrics


def best_combo(
    env: Any,
    variants_by_agent: list[list[RouteVariant]],
    max_combos: int,
    schedule_iterations: int,
    schedule_delay_window: int,
    schedule_combos: bool,
    schedule_top_combos: int,
) -> tuple[tuple[RouteVariant, ...], dict[str, Any], int, int]:
    best = None
    best_metrics = None
    checked = 0
    raw_candidates = []
    for combo in itertools.product(*variants_by_agent):
        checked += 1
        metrics = combo_metrics(env, combo)
        if best_metrics is None or metrics["score"] < best_metrics["score"]:
            best = combo
            best_metrics = metrics
        if schedule_combos:
            raw_candidates.append((metrics["score"], combo))
        if checked >= max_combos:
            break

    scheduled_checked = 0
    if schedule_combos:
        best = None
        best_metrics = None
        for _, combo in sorted(raw_candidates, key=lambda item: item[0])[
            : max(1, schedule_top_combos)
        ]:
            scheduled_checked += 1
            metrics = schedule_combo(
                env,
                combo,
                schedule_iterations,
                schedule_delay_window,
            )
            if best_metrics is None or metrics["score"] < best_metrics["score"]:
                best = combo
                best_metrics = metrics

    assert best is not None and best_metrics is not None
    return best, best_metrics, checked, scheduled_checked


def serializable_combo(combo: tuple[RouteVariant, ...], metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "metrics": metrics,
        "routes": [
            {
                "agent": variant.agent,
                "name": variant.name,
                "length": variant.length,
            }
            for variant in combo
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze route alternatives and time-window conflicts for sampled Flatland scenarios."
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=repo_root() / DEFAULT_BASE_STATE,
    )
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument(
        "--scene",
        choices=["scene_1", "scene_2", "scene_3", "scene_4", "scene_5"],
    )
    parser.add_argument("--max-variants", type=int, default=5)
    parser.add_argument("--max-route-steps", type=int, default=700)
    parser.add_argument("--max-combos", type=int, default=20000)
    parser.add_argument("--schedule-iterations", type=int, default=300)
    parser.add_argument("--schedule-delay-window", type=int, default=30)
    parser.add_argument("--schedule-top-combos", type=int, default=24)
    parser.add_argument(
        "--schedule-combos",
        action="store_true",
        help="Schedule the best raw route combinations after static conflict scoring.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    env, obs_builder = make_env(args)

    variants_by_agent = [
        route_variants(
            obs_builder,
            handle,
            args.max_variants,
            args.max_route_steps,
        )
        for handle in env.get_agent_handles()
    ]
    missing = [index for index, variants in enumerate(variants_by_agent) if not variants]
    if missing:
        raise RuntimeError(f"No route variants generated for agents: {missing}")

    greedy_combo = tuple(variants[0] for variants in variants_by_agent)
    greedy_metrics = combo_metrics(env, greedy_combo)
    greedy_scheduled_metrics = schedule_combo(
        env,
        greedy_combo,
        args.schedule_iterations,
        args.schedule_delay_window,
    )
    best, best_metrics, checked, scheduled_checked = best_combo(
        env,
        variants_by_agent,
        args.max_combos,
        args.schedule_iterations,
        args.schedule_delay_window,
        args.schedule_combos,
        args.schedule_top_combos,
    )

    print("agent,earliest,latest,variants,best_length,variant_lengths")
    for handle, variants in enumerate(variants_by_agent):
        agent = env.agents[handle]
        lengths = [variant.length for variant in variants]
        print(
            f"{handle},{agent.earliest_departure},{agent.latest_arrival},"
            f"{len(variants)},{min(lengths)},{'|'.join(str(length) for length in lengths)}"
        )

    print("\nGreedy:")
    print(json.dumps(serializable_combo(greedy_combo, greedy_metrics), indent=2, default=str))
    print("\nGreedy scheduled:")
    print(
        json.dumps(
            serializable_combo(greedy_combo, greedy_scheduled_metrics),
            indent=2,
            default=str,
        )
    )
    print(f"\nBest checked_combos={checked} scheduled_checked={scheduled_checked}:")
    print(json.dumps(serializable_combo(best, best_metrics), indent=2, default=str))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(
                {
                    "seed": args.seed,
                    "num_agents": args.num_agents,
                    "line_length": args.line_length,
                    "scene": args.scene or "scene_5",
                    "checked_combos": checked,
                    "scheduled_checked": scheduled_checked,
                    "variants": [
                        [
                            {
                                "agent": variant.agent,
                                "name": variant.name,
                                "length": variant.length,
                            }
                            for variant in variants
                        ]
                        for variants in variants_by_agent
                    ],
                    "greedy": serializable_combo(greedy_combo, greedy_metrics),
                    "greedy_scheduled": serializable_combo(
                        greedy_combo,
                        greedy_scheduled_metrics,
                    ),
                    "best": serializable_combo(best, best_metrics),
                },
                handle,
                indent=2,
                default=str,
            )
            handle.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
