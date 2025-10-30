#!/usr/bin/env python3
"""
Unified launcher for wandb sweeps across multiple camera models.

Usage:
    python scripts/launch_sweep.py --sweep n_train_images --camera_models all
    python scripts/launch_sweep.py --sweep angles --camera_models stereo iphone
    python scripts/launch_sweep.py --sweep angles --camera_models all --dry_run
"""

import argparse
import os
import subprocess
import tempfile
from typing import Any, Dict

import yaml

BLENDER_DATA_BASE = "/home/wl757/multiplexed-pixels/plenoxels/blender_data"
DATASETS = [
    f"{BLENDER_DATA_BASE}/{scene}"
    for scene in [
        "chair",
        "drums",
        "materials",
        "mic",
        "ship",
        "ficus",
        "lego_gen12",
        "hotdog",
    ]
]

# Sweep type configurations
SWEEP_CONFIGS = {
    "n_train_images": {
        "program": "train_sim_multiviews.py",
        "method": "grid",
        "parameters": {
            "resolution": {"values": [1]},
            "source_path": {"values": DATASETS},
            "n_train_images": {"values": [1, 2, 3, 5, 10, 15, 20, 30, 40, 50]},
        },
        "command": [
            "${interpreter}",
            "${program}",
            "--iterations",
            "3000",
            "--use_blender",
            "--output-id",
            "7",
            "${args_no_boolean_flags}",
        ],
    },
    "angles": {
        "program": "train_sim_multiviews.py",
        "method": "grid",
        "parameters": {
            "resolution": {"values": [1]},
            "source_path": {"values": DATASETS},
            "angle_deg": {"values": [0.2, 0.5, 0.8, 1, 2, 5, 8, 10, 15, 20]},
            "camera_offset": {"values": [0]},
            "n_train_images": {"values": [1, 3]},
        },
        "command": [
            "${interpreter}",
            "${program}",
            "--iterations",
            "3000",
            "--use_blender",
            "--output-id",
            "9",
            "${args_no_boolean_flags}",
        ],
    },
    "offset": {
        "program": "train_sim_multiviews.py",
        "method": "grid",
        "parameters": {
            "resolution": {"values": [1]},
            "source_path": {"values": DATASETS},
            "camera_offset": {
                "values": [
                    2,
                    0,
                    -2,
                    -5,
                    -10,
                    -15,
                    -20,
                    -25,
                    -30,
                    -35,
                    -40,
                    -45,
                    -50,
                    -75,
                    -100,
                    -125,
                    -150,
                    -175,
                    -200,
                    -250,
                    -500,
                ]
            },
            "n_train_images": {"values": [1]},
        },
        "command": [
            "${interpreter}",
            "${program}",
            "--iterations",
            "3000",
            "--use_blender",
            "--output-id",
            "8",
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
            "iphone_same_focal_length": {"values": [False]},
        },
    },
}

# Sweep-specific overrides for camera configurations
SWEEP_SPECIFIC_OVERRIDES = {
    "offset": {
        "name_pattern": "run_{sweep_type}_singleview_{camera_model}",
        "iphone": {
            "params": {
                "iphone_same_focal_length": {"values": [False]},
            },
        },
    },
    "angles": {
        "iphone": {
            "params": {
                "iphone_same_focal_length": {"values": [False]},
            },
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


def apply_sweep_overrides(
    cam_config: Dict[str, Any],
    sweep_type: str,
    camera_model: str,
) -> str:
    """Apply sweep-specific overrides and return config name."""
    if sweep_type not in SWEEP_SPECIFIC_OVERRIDES:
        return f"run_{sweep_type}_{camera_model}"

    sweep_overrides = SWEEP_SPECIFIC_OVERRIDES[sweep_type]

    if camera_model in sweep_overrides and "params" in sweep_overrides[camera_model]:
        cam_config["params"] = sweep_overrides[camera_model]["params"].copy()

    name_pattern = sweep_overrides.get(
        "name_pattern", "run_{sweep_type}_{camera_model}"
    )
    return name_pattern.format(sweep_type=sweep_type, camera_model=camera_model)


def clean_camera_params(config: Dict[str, Any]) -> None:
    """Remove all camera-specific flags and params from config."""
    for cam_model in CAMERA_MODELS.values():
        flag = cam_model["flag"]
        if flag in config["parameters"]:
            del config["parameters"][flag]

        for param in cam_model.get("params", {}).keys():
            if param in config["parameters"]:
                del config["parameters"][param]


def generate_camera_config(
    base_config: Dict[str, Any],
    sweep_type: str,
    camera_model: str,
) -> Dict[str, Any]:
    """Generate a camera-specific config from the base config."""
    import copy

    config = {
        "program": base_config["program"],
        "method": base_config["method"],
        "parameters": base_config["parameters"].copy(),
        "command": base_config["command"],
    }

    cam_config = copy.deepcopy(CAMERA_MODELS[camera_model])
    config["name"] = apply_sweep_overrides(cam_config, sweep_type, camera_model)

    clean_camera_params(config)

    flag_values = cam_config.get("flag_values", [True])
    config["parameters"][cam_config["flag"]] = {"values": flag_values}

    for param, value in cam_config.get("params", {}).items():
        config["parameters"][param] = value

    return config


def extract_sweep_id(output: str) -> str:
    """Extract sweep ID from wandb command output."""
    for line in output.split("\n"):
        if "Created sweep" in line and "ID:" in line:
            return line.split("ID:")[-1].strip()
        if "wandb.ai" in line and "sweeps" in line:
            return line.split("sweeps/")[-1].split()[0].strip()

    raise RuntimeError("Could not extract sweep ID from wandb output")


def create_sweep(
    config: Dict[str, Any],
    entity: str = None,
    project: str = None,
) -> str:
    """Create a wandb sweep and return the sweep ID."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config, f, default_flow_style=False)
        temp_config = f.name

    try:
        cmd = ["wandb", "sweep"]
        if entity:
            cmd.extend(["--entity", entity])
        if project:
            cmd.extend(["--project", project])
        cmd.append(temp_config)

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return extract_sweep_id(result.stdout + result.stderr)

    finally:
        os.unlink(temp_config)


def get_wandb_entity() -> str:
    """Get the current wandb entity (username)."""
    try:
        import wandb

        api = wandb.Api()
        # Try to get default entity first, fall back to username
        return api.viewer.entity or api.viewer.username
    except Exception:
        pass
    return None


def build_sweep_path(sweep_id: str, entity: str = None, project: str = None) -> str:
    """Build full sweep path from components."""
    if not entity:
        entity = get_wandb_entity()

    if entity and project:
        return f"{entity}/{project}/{sweep_id}"
    if project:
        return f"{project}/{sweep_id}"
    return sweep_id


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
        "--project",
        type=str,
        default="multiplexed-pixels",
        help="Wandb project name (default: multiplexed-pixels)",
    )

    parser.add_argument(
        "--dry_run", action="store_true", help="Print configs without creating sweeps"
    )

    args = parser.parse_args()

    # Auto-detect entity if not provided
    entity = args.entity
    if not entity:
        entity = get_wandb_entity()
        if not entity:
            print(
                "WARNING: Could not auto-detect wandb entity. Agent commands may require manual entity specification."
            )

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
    if entity:
        print(f"Entity: {entity}" + (" (auto-detected)" if not args.entity else ""))
    print(f"{'=' * 80}\n")

    sweep_ids = []

    for camera_model in camera_models:
        print(f"\n{'=' * 80}\nCamera Model: {camera_model}\n{'=' * 80}")

        config = generate_camera_config(base_config, args.sweep, camera_model)

        if args.dry_run:
            print("\nGenerated config:")
            print(yaml.dump(config, default_flow_style=False))
            continue

        try:
            print("Creating sweep...")
            sweep_id = create_sweep(config, entity, args.project)
            sweep_ids.append({"camera_model": camera_model, "sweep_id": sweep_id})
            print(f"✓ Created sweep: {sweep_id}")

        except Exception as e:
            print(f"✗ Failed to create sweep: {e}")

    if args.dry_run or not sweep_ids:
        return 0

    print(f"\n\n{'=' * 80}\nSWEEP IDS\n{'=' * 80}\n")
    for item in sweep_ids:
        print(item["sweep_id"])

    print(f"\n{'=' * 80}\nAGENT COMMANDS\n{'=' * 80}\n")
    for item in sweep_ids:
        sweep_path = build_sweep_path(item["sweep_id"], entity, args.project)
        print(f"# {item['camera_model']}\nwandb agent {sweep_path}")

    print(f"\n{'=' * 80}\n")

    return 0


if __name__ == "__main__":
    exit(main())
