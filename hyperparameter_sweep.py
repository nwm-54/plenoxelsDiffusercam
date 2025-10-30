#!/usr/bin/env python3
"""
Hyperparameter sweep for 3DGS with 3000 iterations.
Optimized for quick training cycles.
"""

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

from arguments.camera_presets import OPTIMIZATION_PRESETS

# Base configuration
BASE_CONFIG = {
    "data_path": "/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego_gen12",
    "n_train_images": 3,
    "iterations": 3000,
}


@dataclass(frozen=True)
class SweepTask:
    """Represents a single sweep experiment."""
    camera_model: str
    config_name: str
    params: Dict[str, Any]


# Camera-specific command-line arguments
CAMERA_SPECS = {
    "base": {},  # Single-view baseline (no special flags)
    "stereo": {"flags": ["--use_stereo"]},
    "iphone": {"flags": ["--use_iphone"]},
    "multiplexing": {"flags": ["--use_multiplexing"], "extra_args": ["--dls", "20"]},
    "lightfield": {"flags": ["--use_multiplexing"], "extra_args": ["--dls", "12"]},
}

DEFAULT_CAPTURE_ANGLE = "10"


def build_experiment_command(
    config_name: str,
    params: Dict[str, Any],
    camera_model: str,
    output_dir: str,
) -> List[str]:
    """Build command for running an experiment."""
    cmd = [
        "python", "train_sim_multiviews.py",
        "-s", BASE_CONFIG["data_path"],
        "-m", output_dir,
        "--n_train_images", str(BASE_CONFIG["n_train_images"]),
        "--iterations", str(BASE_CONFIG["iterations"]),
        "--angle_deg", DEFAULT_CAPTURE_ANGLE,
    ]

    # Add camera-specific flags
    camera_spec = CAMERA_SPECS.get(camera_model, {})
    cmd.extend(camera_spec.get("flags", []))
    cmd.extend(camera_spec.get("extra_args", []))

    # Add hyperparameters
    for param, value in params.items():
        cmd.extend([f"--{param}", str(value)])

    return cmd


def write_task_summary(
    result: Dict[str, Any], summary_dir: Path, task_index: int, task: SweepTask
) -> None:
    """Write individual task summary to JSON file."""
    summary_path = summary_dir / f"{task_index:02d}_{task.camera_model}_{task.config_name}.json"
    with summary_path.open("w") as f:
        json.dump(result, f, indent=2)


def run_experiment(
    config_name: str,
    params: Dict[str, Any],
    camera_model: str,
    output_base: str = "./output5/sweep",
    dry_run: bool = False,
    task_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Run a single experiment with given hyperparameters."""
    output_dir = f"{output_base}/{camera_model}_{config_name}"
    cmd = build_experiment_command(config_name, params, camera_model, output_dir)

    print(f"\n{'='*80}")
    print(f"{'[DRY RUN] ' if dry_run else ''}Running: {camera_model} - {config_name}")
    print(f"Output: {output_dir}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    result = {
        "config_name": config_name,
        "camera_model": camera_model,
        "params": params,
        "output_dir": output_dir,
        "command": " ".join(cmd),
        "success": False,
    }
    if task_index is not None:
        result["task_index"] = task_index

    if dry_run:
        result["success"] = True
        result["dry_run"] = True
        return result

    try:
        subprocess.run(cmd, check=True)
        result["success"] = True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Experiment failed with return code {e.returncode}")
        result["error"] = str(e)

    return result


def build_tasks(camera_models: List[str], configs: List[str]) -> List[SweepTask]:
    """Generate all tasks for the sweep."""
    tasks: List[SweepTask] = []
    for camera_model in camera_models:
        for config_name in configs:
            if config_name not in OPTIMIZATION_PRESETS:
                raise ValueError(f"Unknown config: {config_name}")
            tasks.append(
                SweepTask(
                    camera_model=camera_model,
                    config_name=config_name,
                    params=dict(OPTIMIZATION_PRESETS[config_name]),
                )
            )
    return tasks


def print_task_table(tasks: List[SweepTask]) -> None:
    """Print mapping between task indices and configurations."""
    print("Sweep task mapping:")
    print("-" * 80)
    for idx, task in enumerate(tasks):
        print(
            f"{idx:02d}: camera_model={task.camera_model}, "
            f"config={task.config_name}"
        )
    print("-" * 80)
    print(
        f"Total tasks: {len(tasks)} | "
        f"Use SBATCH --array=0-{max(len(tasks) - 1, 0)}"
    )


def get_task_index() -> Optional[int]:
    """Determine task index from CLI args or SLURM environment."""
    import sys

    # Check for --task_id flag
    if "--task_id" in sys.argv:
        idx = sys.argv.index("--task_id")
        if idx + 1 < len(sys.argv):
            return int(sys.argv[idx + 1])

    # Check SLURM environment
    env_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if env_task_id:
        return int(env_task_id)

    return None


def run_single_task(
    task: SweepTask,
    task_index: int,
    args,
    summary_dir: Path,
) -> Dict[str, Any]:
    """Execute a single sweep task and save results."""
    print(
        f"\n[TASK {task_index}] Running: {task.camera_model} - {task.config_name}"
    )

    result = run_experiment(
        config_name=task.config_name,
        params=task.params,
        camera_model=task.camera_model,
        output_base=args.output_base,
        dry_run=args.dry_run,
        task_index=task_index,
    )

    write_task_summary(result, summary_dir, task_index, task)
    return result


def main():
    """Run hyperparameter sweep."""
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter sweep for 3DGS")
    parser.add_argument(
        "--camera_models",
        nargs="+",
        choices=list(CAMERA_SPECS.keys()),
        default=["base"],
        help="Camera models to test"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(OPTIMIZATION_PRESETS.keys()),
        default=list(OPTIMIZATION_PRESETS.keys()),
        help="Configurations to test"
    )
    parser.add_argument(
        "--output_base",
        type=str,
        default="./output5/sweep",
        help="Base output directory"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them"
    )
    parser.add_argument(
        "--task_id",
        type=int,
        default=None,
        help="Specific task index to run (overrides SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--list_tasks",
        action="store_true",
        help="Print task mapping for SLURM job array and exit",
    )

    args = parser.parse_args()
    tasks = build_tasks(args.camera_models, args.configs)

    if args.list_tasks:
        print_task_table(tasks)
        return

    # Setup output directories
    os.makedirs(args.output_base, exist_ok=True)
    summary_dir = Path(args.output_base) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # SLURM array mode: run single task
    task_index = args.task_id if args.task_id is not None else get_task_index()

    if task_index is not None:
        if not 0 <= task_index < len(tasks):
            raise IndexError(f"Task index {task_index} out of range (0-{len(tasks)-1})")

        run_single_task(tasks[task_index], task_index, args, summary_dir)
        return

    # Sequential mode: run all tasks
    print(
        f"Running {len(tasks)} experiments sequentially.\n"
        f"To parallelize with SLURM: sbatch --array=0-{len(tasks)-1} hyperparameter_sweep.slurm\n"
    )

    results = []
    for idx, task in enumerate(tasks):
        result = run_single_task(task, idx, args, summary_dir)
        results.append(result)

    # Save overall summary
    summary_path = Path(args.output_base) / "sweep_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    successful = sum(r["success"] for r in results)
    print(f"\n{'='*80}")
    print(f"SWEEP COMPLETE: {successful}/{len(results)} experiments succeeded")
    print(f"Results: {summary_path}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
