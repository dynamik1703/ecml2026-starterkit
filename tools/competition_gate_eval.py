#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SCENES = ("scene_1", "scene_2", "scene_3", "scene_4", "scene_5")
CRITICAL_SCENES = ("scene_1", "scene_2", "scene_5")
MAPF_OPPORTUNITY_SCENES = ("scene_4",)


def load_evaluate_sampled() -> Any:
    path = ROOT / "tools" / "evaluate_sampled.py"
    spec = importlib.util.spec_from_file_location("ecml_evaluate_sampled", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluator from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_scene_list(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    invalid = [value for value in values if value not in SCENES]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid scenes: {invalid}")
    return values


def parse_mode_list(raw: str) -> list[str]:
    values = [value.strip() for value in raw.split(",") if value.strip()]
    valid = {"known", "unknown", "level"}
    invalid = [value for value in values if value not in valid]
    if invalid:
        raise argparse.ArgumentTypeError(f"invalid modes: {invalid}")
    return values


def runtime_scene_for(mode: str, scene: str) -> str:
    if mode == "known":
        return scene
    if mode == "unknown":
        return "__none__"
    if mode == "level":
        return scene.replace("scene_", "level_")
    raise ValueError(f"unknown mode {mode}")


def evaluator_args(args: argparse.Namespace, scene: str, runtime_scene: str) -> SimpleNamespace:
    evaluator = load_evaluate_sampled()
    return SimpleNamespace(
        base_state_pkl=args.base_state_pkl,
        policy=args.policy,
        policy_checkpoint=args.policy_checkpoint,
        obs_builder=args.obs_builder,
        rewards=args.rewards,
        episodes=args.episodes,
        seed=args.seed,
        num_agents=args.num_agents,
        line_length=args.line_length,
        scene=scene,
        runtime_scene=runtime_scene,
        output_json=None,
        output_csv=None,
        agent_details=False,
        _evaluator=evaluator,
    )


def run_block(args: argparse.Namespace, scene: str, mode: str) -> dict[str, Any]:
    runtime_scene = runtime_scene_for(mode, scene)
    eval_args = evaluator_args(args, scene, runtime_scene)
    rows = []
    for index in range(args.episodes):
        seed = args.seed + index
        if not args.quiet:
            print(
                f"RUN mode={mode} scene={scene} runtime_scene="
                f"{None if runtime_scene == '__none__' else runtime_scene} "
                f"seed={seed}",
                flush=True,
            )
        row = eval_args._evaluator.run_episode(eval_args, seed)
        rows.append(row)
        if not args.quiet:
            print(
                f"  result success={row['success_rate']:.6g} "
                f"reward={row['normalized_reward']:.6g} "
                f"env_time={row['env_time']}",
                flush=True,
            )
    reward_mean = sum(row["normalized_reward"] for row in rows) / float(len(rows))
    success_mean = sum(row["success_rate"] for row in rows) / float(len(rows))
    min_success = min(row["success_rate"] for row in rows)
    max_env_time = max(row["env_time"] for row in rows)
    return {
        "scene": scene,
        "mode": mode,
        "runtime_scene": None if runtime_scene == "__none__" else runtime_scene,
        "episodes": len(rows),
        "seed_start": args.seed,
        "seed_end": args.seed + len(rows) - 1,
        "reward_mean": reward_mean,
        "success_mean": success_mean,
        "min_success": min_success,
        "max_env_time": max_env_time,
        "rows": rows,
    }


def evaluate_gate(args: argparse.Namespace, blocks: list[dict[str, Any]]) -> list[str]:
    failures: list[str] = []
    for block in blocks:
        scene = block["scene"]
        mode = block["mode"]
        success = float(block["success_mean"])

        if success < args.hard_block_min_success:
            failures.append(
                f"{mode}/{scene}: success {success:.4f} < hard block "
                f"{args.hard_block_min_success:.4f}"
            )
        if success < args.min_scene_success:
            failures.append(
                f"{mode}/{scene}: success {success:.4f} < minimum "
                f"{args.min_scene_success:.4f}"
            )
        if scene in CRITICAL_SCENES and success < args.critical_min_success:
            failures.append(
                f"{mode}/{scene}: critical success {success:.4f} < "
                f"{args.critical_min_success:.4f}"
            )
        if scene in MAPF_OPPORTUNITY_SCENES and success < args.mapf_min_success:
            failures.append(
                f"{mode}/{scene}: MAPF opportunity success {success:.4f} < "
                f"{args.mapf_min_success:.4f}"
            )

    if blocks:
        overall_success = sum(float(block["success_mean"]) for block in blocks) / float(
            len(blocks)
        )
        if overall_success < args.overall_min_success:
            failures.append(
                f"overall: success {overall_success:.4f} < "
                f"{args.overall_min_success:.4f}"
            )
    return failures


def target_warnings(args: argparse.Namespace, blocks: list[dict[str, Any]]) -> list[str]:
    warnings = []
    for block in blocks:
        scene = block["scene"]
        mode = block["mode"]
        success = float(block["success_mean"])
        target = args.mapf_target_success if scene in MAPF_OPPORTUNITY_SCENES else args.target_scene_success
        if success < target:
            warnings.append(
                f"{mode}/{scene}: success {success:.4f} below target {target:.4f}"
            )
    return warnings


def print_table(blocks: list[dict[str, Any]]) -> None:
    print(
        "mode,scene,runtime_scene,episodes,seeds,success_mean,min_success,"
        "reward_mean,max_env_time"
    )
    for block in blocks:
        print(
            f"{block['mode']},{block['scene']},{block['runtime_scene']},"
            f"{block['episodes']},{block['seed_start']}-{block['seed_end']},"
            f"{block['success_mean']:.6g},{block['min_success']:.6g},"
            f"{block['reward_mean']:.6g},{block['max_env_time']}"
        )


def parse_args() -> argparse.Namespace:
    evaluator = load_evaluate_sampled()
    parser = argparse.ArgumentParser(
        description="Strict local gate before Docker/competition submission."
    )
    parser.add_argument(
        "--base-state-pkl",
        type=Path,
        default=ROOT / evaluator.DEFAULT_BASE_STATE,
    )
    parser.add_argument("--policy", default="submission.scene_aware_competition_policy.MyPolicy")
    parser.add_argument("--policy-checkpoint", type=Path)
    parser.add_argument("--obs-builder", default=evaluator.DEFAULT_OBS_BUILDER)
    parser.add_argument("--rewards", default=evaluator.DEFAULT_REWARDS)
    parser.add_argument("--scenes", type=parse_scene_list, default=list(SCENES))
    parser.add_argument(
        "--modes",
        type=parse_mode_list,
        default=["known", "unknown", "level"],
        help="Comma-separated: known,unknown,level.",
    )
    parser.add_argument("--episodes", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7410)
    parser.add_argument("--num-agents", type=int, default=80)
    parser.add_argument("--line-length", type=int, default=2)
    parser.add_argument("--hard-block-min-success", type=float, default=0.30)
    parser.add_argument("--min-scene-success", type=float, default=0.35)
    parser.add_argument("--critical-min-success", type=float, default=0.35)
    parser.add_argument("--mapf-min-success", type=float, default=0.55)
    parser.add_argument("--overall-min-success", type=float, default=0.45)
    parser.add_argument("--target-scene-success", type=float, default=0.45)
    parser.add_argument("--mapf-target-success", type=float, default=0.60)
    parser.add_argument("--fail-on-target", action="store_true")
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    blocks = []
    for mode in args.modes:
        for scene in args.scenes:
            if not args.quiet:
                print(f"BLOCK START mode={mode} scene={scene}", flush=True)
            block = run_block(args, scene, mode)
            blocks.append(block)
            if not args.quiet:
                print(
                    f"BLOCK DONE mode={mode} scene={scene} "
                    f"success_mean={block['success_mean']:.6g} "
                    f"reward_mean={block['reward_mean']:.6g}",
                    flush=True,
                )
    print_table(blocks)

    failures = evaluate_gate(args, blocks)
    warnings = target_warnings(args, blocks)
    if args.fail_on_target:
        failures.extend(warnings)

    summary = {
        "policy": args.policy,
        "episodes": args.episodes,
        "seed": args.seed,
        "num_agents": args.num_agents,
        "line_length": args.line_length,
        "blocks": blocks,
        "warnings": warnings,
        "failures": failures,
        "passed": not failures,
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")

    for warning in warnings:
        print(f"TARGET-MISS: {warning}")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        print("Gate: FAIL")
        return 1
    print("Gate: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
