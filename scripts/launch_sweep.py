#!/usr/bin/env python3
"""
Unified launcher for wandb sweeps across multiple camera models.

Usage:
    # Launch all camera models for a sweep type
    python scripts/launch_sweep.py --sweep n_train_images --camera_models all

    # Launch specific camera models
    python scripts/launch_sweep.py --sweep angles --camera_models stereo iphone

    # Dry run to preview configs
    python scripts/launch_sweep.py --sweep angles --camera_models all --dry_run

See scripts/README.md for full documentation.
"""

import argparse
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

# Base datasets
DATASETS = [
    "/home/wl757/multiplexed-pixels/plenoxels/blender_data/chair",
    "/home/wl757/multiplexed-pixels/plenoxels/blender_data/drums",
    "/home/wl757/multiplexed-pixels/plenoxels/blender_data/materials",
    "/home/wl757/multiplexed-pixels/plenoxels/blender_data/mic",
    "/home/wl757/multiplexed-pixels/plenoxels/blender_data/ship",
    "/home/wl757/multiplexed-pixels/plenoxels/blender_data/ficus",
    "/home/wl757/multiplexed-pixels/plenoxels/blender_data/lego_gen12",
    "/home/wl757/multiplexed-pixels/plenoxels/blender_data/hotdog",
]

# Sweep type configurations
SWEEP_CONFIGS = {
    "n_train_images": {
        "program": "train_sim_multiviews.py",
        "method": "grid",
        "parameters": {
            "tv_weight": {"values": [0.0]},
            "tv_unseen_weight": {"values": [0.0]},
            "resolution": {"values": [1]},
            "source_path": {"values": DATASETS},
            "n_train_images": {"values": [1, 2, 3, 5, 10, 15, 20, 30, 40, 50]},
        },
        "command": [
            "${interpreter}",
            "${program}",
            "--iterations", "3000",
            "--output-id", "7",
            "${args_no_boolean_flags}",
        ],
    },
    "angles": {
        "program": "train_sim_multiviews.py",
        "method": "grid",
        "parameters": {
            "tv_weight": {"values": [0.0]},
            "tv_unseen_weight": {"values": [0.0]},
            "resolution": {"values": [1]},
            "source_path": {"values": DATASETS},
            "angle_deg": {"values": [0.2, 0.5, 0.8, 1, 2, 5, 8, 10, 15, 20]},
            "camera_offset": {"values": [0]},
            "n_train_images": {"values": [1, 3]},
        },
        "command": [
            "${interpreter}",
            "${program}",
            "--iterations", "3000",
            "--use_blender",
            "--output-id", "9",
            "${args_no_boolean_flags}",
        ],
    },
    "offset": {
        "program": "train_sim_multiviews.py",
        "method": "grid",
        "parameters": {
            "tv_weight": {"values": [0.0]},
            "tv_unseen_weight": {"values": [0.0]},
            "resolution": {"values": [1]},
            "source_path": {"values": DATASETS},
            "camera_offset": {"values": [2, 0, -2, -5, -10, -15, -20, -25, -30, -35, -40, -45, -50, -75, -100, -125, -150, -175, -200, -250, -500]},
            "n_train_images": {"values": [1]},
        },
        "command": [
            "${interpreter}",
            "${program}",
            "--iterations", "3000",
            "--output-id", "8",
            "${args_no_boolean_flags}",
        ],
    },
}

# Camera model configurations
CAMERA_MODELS = {
    "stereo": {
        "flag": "use_stereo",
        "params": {},
    },
    "multiplexing": {
        "flag": "use_multiplexing",
        "params": {
            "dls": {"values": [12, 20]},
        },
        # Note: use_multiplexing can be true or false (lightfield vs multiplexing)
        "flag_values": [True, False],
    },
    "iphone": {
        "flag": "use_iphone",
        "params": {
            "iphone_same_focal_length": {"values": [True]},
        },
    },
}

# Sweep-specific overrides for camera configurations
SWEEP_SPECIFIC_OVERRIDES = {
    "offset": {
        # offset sweeps use "singleview" in the name
        "name_pattern": "run_{sweep_type}_singleview_{camera_model}",
        "iphone": {
            # offset/iphone uses both false and true for iphone_same_focal_length
            "params": {
                "iphone_same_focal_length": {"values": [False, True]},
            },
        },
    },
    "angles": {
        "iphone": {
            # angles/iphone doesn't include iphone_same_focal_length
            "params": {},
        },
    },
}


def get_base_config(sweep_type: str) -> Dict[str, Any]:
    """Get the base configuration for a sweep type."""
    if sweep_type not in SWEEP_CONFIGS:
        raise ValueError(
            f"Unknown sweep type: {sweep_type}. "
            f"Available: {', '.join(SWEEP_CONFIGS.keys())}"
        )

    # Return a deep copy of the config
    import copy
    return copy.deepcopy(SWEEP_CONFIGS[sweep_type])


def generate_camera_config(
    base_config: Dict[str, Any],
    sweep_type: str,
    camera_model: str,
) -> Dict[str, Any]:
    """Generate a camera-specific config from the base config."""
    import copy

    config = base_config.copy()

    # Deep copy parameters
    config["parameters"] = base_config["parameters"].copy()

    # Get camera model configuration
    cam_config = copy.deepcopy(CAMERA_MODELS[camera_model])

    # Apply sweep-specific overrides
    if sweep_type in SWEEP_SPECIFIC_OVERRIDES:
        sweep_overrides = SWEEP_SPECIFIC_OVERRIDES[sweep_type]

        # Check for name pattern override
        name_pattern = sweep_overrides.get(
            "name_pattern", "run_{sweep_type}_{camera_model}"
        )
        config["name"] = name_pattern.format(
            sweep_type=sweep_type, camera_model=camera_model
        )

        # Check for camera-specific overrides for this sweep
        if camera_model in sweep_overrides:
            camera_overrides = sweep_overrides[camera_model]
            # Override params if specified
            if "params" in camera_overrides:
                cam_config["params"] = copy.deepcopy(camera_overrides["params"])
    else:
        # Default name pattern
        config["name"] = f"run_{sweep_type}_{camera_model}"

    # Remove any existing camera flags
    for other_cam in CAMERA_MODELS.keys():
        flag = CAMERA_MODELS[other_cam]["flag"]
        if flag in config["parameters"]:
            del config["parameters"][flag]
        # Also remove model-specific params
        for param in CAMERA_MODELS[other_cam].get("params", {}).keys():
            if param in config["parameters"]:
                del config["parameters"][param]

    # Add camera-specific flag
    flag_values = cam_config.get("flag_values", [True])
    config["parameters"][cam_config["flag"]] = {"values": flag_values}

    # Add camera-specific parameters
    for param, value in cam_config.get("params", {}).items():
        config["parameters"][param] = value

    return config


def create_sweep(
    config: Dict[str, Any],
    entity: str = None,
    project: str = None,
) -> str:
    """Create a wandb sweep and return the sweep ID."""
    # Write config to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        temp_config = f.name

    try:
        # Build wandb sweep init command
        cmd = ["wandb", "sweep"]
        if entity:
            cmd.extend(["--entity", entity])
        if project:
            cmd.extend(["--project", project])
        cmd.append(temp_config)

        # Create sweep
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Extract sweep ID from output
        # Output typically looks like: "wandb: Created sweep with ID: abc123xyz"
        for line in result.stdout.split("\n") + result.stderr.split("\n"):
            if "Created sweep" in line and "ID:" in line:
                sweep_id = line.split("ID:")[-1].strip()
                return sweep_id

        # Fallback: look for sweep URL
        for line in result.stdout.split("\n") + result.stderr.split("\n"):
            if "wandb.ai" in line and "sweeps" in line:
                # Extract sweep ID from URL
                sweep_id = line.split("sweeps/")[-1].split()[0].strip()
                return sweep_id

        raise RuntimeError("Could not extract sweep ID from wandb output")

    finally:
        # Clean up temp file
        os.unlink(temp_config)


def launch_agent(
    sweep_id: str, entity: str = None, project: str = None, count: int = 1
):
    """Launch a wandb agent for the sweep."""
    # Build sweep path
    if entity and project:
        sweep_path = f"{entity}/{project}/{sweep_id}"
    elif project:
        sweep_path = f"{project}/{sweep_id}"
    else:
        sweep_path = sweep_id

    # Build command
    cmd = ["wandb", "agent", "--count", str(count), sweep_path]

    print(f"Launching agent: {' '.join(cmd)}")
    print(f"To launch more agents, run: wandb agent {sweep_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Unified launcher for wandb sweeps across camera models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--sweep",
        type=str,
        required=True,
        help="Sweep type (e.g., n_train_images, angles, offset)",
    )

    parser.add_argument(
        "--camera_models",
        nargs="+",
        choices=["stereo", "multiplexing", "iphone", "all"],
        default=["all"],
        help="Camera models to launch sweeps for",
    )

    parser.add_argument(
        "--entity", type=str, default=None, help="Wandb entity (username or team)"
    )

    parser.add_argument(
        "--project", type=str, default="multiplexed-pixels", help="Wandb project name (default: multiplexed-pixels)"
    )

    parser.add_argument(
        "--dry_run", action="store_true", help="Print configs without creating sweeps"
    )

    parser.add_argument(
        "--no_agent", action="store_true", help="Create sweeps but do not launch agents"
    )

    parser.add_argument(
        "--agent_count",
        type=int,
        default=1,
        help="Number of runs per agent (default: 1, use -1 for unlimited)",
    )

    args = parser.parse_args()

    # Expand 'all' to all camera models
    if "all" in args.camera_models:
        camera_models = list(CAMERA_MODELS.keys())
    else:
        camera_models = args.camera_models

    # Get base configuration
    try:
        base_config = get_base_config(args.sweep)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    print(f"\n{'=' * 80}")
    print(f"Launching sweeps for: {args.sweep}")
    print(f"Camera models: {', '.join(camera_models)}")
    print(f"Project: {args.project}")
    if args.entity:
        print(f"Entity: {args.entity}")
    print(f"{'=' * 80}\n")

    sweep_ids = []

    for camera_model in camera_models:
        print(f"\n{'=' * 80}")
        print(f"Camera Model: {camera_model}")
        print(f"{'=' * 80}")

        # Generate camera-specific config
        config = generate_camera_config(base_config, args.sweep, camera_model)

        if args.dry_run:
            print("\nGenerated config:")
            print(yaml.dump(config, default_flow_style=False))
            continue

        # Create sweep
        print("Creating sweep...")
        try:
            sweep_id = create_sweep(config, args.entity, args.project)
            sweep_ids.append(
                {
                    "camera_model": camera_model,
                    "sweep_id": sweep_id,
                }
            )
            print(f"✓ Created sweep: {sweep_id}")

            # Launch agent if requested
            if not args.no_agent:
                print("Launching agent...")
                launch_agent(sweep_id, args.entity, args.project, args.agent_count)

        except Exception as e:
            print(f"✗ Failed to create sweep: {e}")
            continue

    # Summary
    if not args.dry_run and sweep_ids:
        print(f"\n\n{'=' * 80}")
        print("SWEEP IDS")
        print(f"{'=' * 80}\n")

        # Print sweep IDs for easy copying
        for item in sweep_ids:
            print(f"{item['sweep_id']}")

        print(f"\n{'=' * 80}")
        print("AGENT COMMANDS")
        print(f"{'=' * 80}\n")

        for item in sweep_ids:
            if args.entity and args.project:
                sweep_path = f"{args.entity}/{args.project}/{item['sweep_id']}"
            elif args.project:
                sweep_path = f"{args.project}/{item['sweep_id']}"
            else:
                sweep_path = item["sweep_id"]

            print(f"# {item['camera_model']}")
            print(f"wandb agent {sweep_path}")

        print(f"\n{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    exit(main())
