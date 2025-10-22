#!/usr/bin/env python3
"""
Hyperparameter sweep for 3DGS with 3000 iterations.
Optimized for quick training cycles.
"""

import copy
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Optional

# Base configuration
BASE_CONFIG = {
    "data_path": "/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego_gen12",
    "n_train_images": 3,
    "iterations": 3000,
}

# Hyperparameters to sweep
# Focusing on parameters most impactful for reduced iteration count
SWEEP_CONFIGS = {
    # Current defaults; baseline reference
    "baseline": {
        "position_lr_init": 0.00016,
        "densify_grad_threshold": 0.000015,
        "densification_interval": 50,
        "densify_from_iter": 300,
        "opacity_reset_interval": 1000,
        "lambda_dssim": 0.1,
        "tv_weight": 0.0,
        "tv_unseen_weight": 0.0,
    },

    # Balanced schedule tuned for stereo/multiplexing gains
    "balanced_window": {
        "position_lr_init": 0.00024,
        "position_lr_final": 8e-06,
        "densify_grad_threshold": 0.000013,
        "densification_interval": 35,
        "densify_from_iter": 220,
        "densify_until_iter": 2600,
        "opacity_reset_interval": 600,
        "lambda_dssim": 0.16,
        "tv_weight": 0.0025,
        "tv_unseen_weight": 0.0003,
        "opacity_lr": 0.07,
    },

    # Softer TV weights for base/iPhone while keeping balanced cadence
    "bw_plus": {
        "position_lr_init": 0.00024,
        "position_lr_final": 8e-06,
        "densify_grad_threshold": 0.000012,
        "densification_interval": 32,
        "densify_from_iter": 200,
        "densify_until_iter": 2550,
        "opacity_reset_interval": 600,
        "lambda_dssim": 0.16,
        "tv_weight": 0.002,
        "tv_unseen_weight": 0.00025,
        "opacity_lr": 0.07,
    },

    # Lightest TV / later densification variant to protect baseline PSNR
    "bw_light": {
        "position_lr_init": 0.00024,
        "position_lr_final": 1e-05,
        "densify_grad_threshold": 0.0000125,
        "densification_interval": 35,
        "densify_from_iter": 220,
        "densify_until_iter": 2600,
        "opacity_reset_interval": 620,
        "lambda_dssim": 0.14,
        "tv_weight": 0.0015,
        "tv_unseen_weight": 0.0002,
    },

    # Heavier TV/early densification for multiplexing-style lift
    "tv_mix": {
        "position_lr_init": 0.00027,
        "position_lr_final": 8e-06,
        "densify_grad_threshold": 0.000011,
        "densification_interval": 30,
        "densify_from_iter": 190,
        "densify_until_iter": 2550,
        "opacity_reset_interval": 540,
        "lambda_dssim": 0.17,
        "tv_weight": 0.0035,
        "tv_unseen_weight": 0.00035,
    },

    # Mid LR taper to bridge stability and speed
    "lr_mid": {
        "position_lr_init": 0.00028,
        "position_lr_final": 7e-06,
        "densify_grad_threshold": 0.0000115,
        "densification_interval": 30,
        "densify_from_iter": 200,
        "densify_until_iter": 2500,
        "opacity_reset_interval": 560,
        "lambda_dssim": 0.16,
        "tv_weight": 0.0025,
        "tv_unseen_weight": 0.00025,
        "scaling_lr": 0.007,
        "opacity_lr": 0.075,
    },
}


@dataclass(frozen=True)
class SweepTask:
    """Represents a single sweep experiment."""

    camera_model: str
    config_name: str
    params: Dict[str, Any]


CAMERA_SPECS = {
    "stereo": {"flags": ["--use_stereo"]},
    "iphone": {"flags": ["--use_iphone"]},
    "multiplexing": {"flags": ["--use_multiplexing"], "extra_args": ["--dls", "20"]},
    "lightfield": {"flags": ["--use_multiplexing"], "extra_args": ["--dls", "12"]},
}

DEFAULT_CAPTURE_ANGLE = "10"


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

    # Build command
    cmd = [
        "python", "train_sim_multiviews.py",
        "-s", BASE_CONFIG["data_path"],
        "-m", output_dir,
        "--n_train_images", str(BASE_CONFIG["n_train_images"]),
        "--iterations", str(BASE_CONFIG["iterations"]),
    ]

    # Add camera model flag
    camera_spec = CAMERA_SPECS.get(camera_model, {})
    cmd.extend(camera_spec.get("flags", []))
    cmd.extend(camera_spec.get("extra_args", []))
    # "base" doesn't need a flag

    # Keep capture angle consistent across sweeps
    cmd.extend(["--angle_deg", DEFAULT_CAPTURE_ANGLE])

    # Add hyperparameters
    for param, value in params.items():
        cmd.extend([f"--{param}", str(value)])

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
            tasks.append(
                SweepTask(
                    camera_model=camera_model,
                    config_name=config_name,
                    params=copy.deepcopy(SWEEP_CONFIGS[config_name]),
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


def main():
    """Run hyperparameter sweep."""
    import argparse

    parser = argparse.ArgumentParser(description="Hyperparameter sweep for 3DGS")
    parser.add_argument(
        "--camera_models",
        nargs="+",
        choices=["base", "stereo", "iphone", "multiplexing", "lightfield"],
        default=["base"],
        help="Camera models to test"
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=list(SWEEP_CONFIGS.keys()),
        default=list(SWEEP_CONFIGS.keys()),
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

    # Create output directory
    os.makedirs(args.output_base, exist_ok=True)
    summary_dir = Path(args.output_base) / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Determine execution mode
    env_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    task_index: Optional[int] = None

    if args.task_id is not None:
        task_index = args.task_id
    elif env_task_id is not None:
        try:
            task_index = int(env_task_id)
        except ValueError as exc:
            raise ValueError(
                f"Invalid SLURM_ARRAY_TASK_ID value: {env_task_id}"
            ) from exc

    if task_index is not None:
        if task_index < 0 or task_index >= len(tasks):
            raise IndexError(
                f"Task index {task_index} out of range for {len(tasks)} tasks"
            )

        task = tasks[task_index]
        print(
            f"\n[SLURM MODE] Running task {task_index} / {len(tasks) - 1}: "
            f"{task.camera_model} - {task.config_name}"
        )

        result = run_experiment(
            config_name=task.config_name,
            params=task.params,
            camera_model=task.camera_model,
            output_base=args.output_base,
            dry_run=args.dry_run,
            task_index=task_index,
        )

        summary_path = summary_dir / (
            f"{task_index:02d}_{task.camera_model}_{task.config_name}.json"
        )
        with summary_path.open("w") as f:
            json.dump(result, f, indent=2)

        print(f"\nResult summary written to: {summary_path}")
        return

    # Sequential / local execution path
    results = []
    total_experiments = len(tasks)
    print(
        "Running sequential sweep. To parallelize, submit as a SLURM array with:\n"
        f"  sbatch hyperparameter_sweep.slurm --configs ... --camera_models ...\n"
        f"Array range: 0-{max(total_experiments - 1, 0)}\n"
    )

    for idx, task in enumerate(tasks, start=1):
        print(f"\n\n[Experiment {idx}/{total_experiments}]")

        result = run_experiment(
            config_name=task.config_name,
            params=task.params,
            camera_model=task.camera_model,
            output_base=args.output_base,
            dry_run=args.dry_run,
            task_index=idx - 1,
        )
        results.append(result)
        summary_path = summary_dir / (
            f"{idx - 1:02d}_{task.camera_model}_{task.config_name}.json"
        )
        with summary_path.open("w") as f:
            json.dump(result, f, indent=2)

    # Save results summary
    summary_path = os.path.join(args.output_base, "sweep_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\n{'='*80}")
    print("SWEEP COMPLETE")
    print(f"Results saved to: {summary_path}")
    print(f"Successful: {sum(r['success'] for r in results)}/{len(results)}")
    print(f"{'='*80}\n")

    # Print configuration details
    print("\nConfiguration Details:")
    print("="*80)
    for config_name, params in SWEEP_CONFIGS.items():
        print(f"\n{config_name}:")
        for param, value in params.items():
            print(f"  {param}: {value}")


if __name__ == "__main__":
    main()
