#!/usr/bin/env python3
"""
Test that launch_sweep.py generates identical configs to existing YAML files.
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path
import yaml


def get_generated_config(sweep_type: str, camera_model: str) -> dict:
    """Get generated config from launch_sweep.py in dry-run mode."""
    cmd = [
        'python', 'scripts/launch_sweep.py',
        '--sweep', sweep_type,
        '--camera_models', camera_model,
        '--dry_run'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # Parse the YAML from stdout
    # The output includes a "Generated config:" line followed by YAML
    lines = result.stdout.split('\n')
    yaml_start = None
    for i, line in enumerate(lines):
        if 'Generated config:' in line:
            yaml_start = i + 1
            break

    if yaml_start is None:
        raise RuntimeError(f"Could not find generated config in output")

    # Extract YAML portion (everything after "Generated config:" until next separator)
    yaml_lines = []
    for i in range(yaml_start, len(lines)):
        if lines[i].startswith('====='):
            break
        yaml_lines.append(lines[i])

    yaml_str = '\n'.join(yaml_lines)
    return yaml.safe_load(yaml_str)


def load_existing_config(sweep_type: str, camera_model: str) -> dict:
    """Load existing YAML config file."""
    config_path = Path(f'scripts/{sweep_type}/{sweep_type}_{camera_model}.yaml')

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def normalize_config(config: dict) -> dict:
    """Normalize config for comparison (sort keys, handle special cases)."""
    # Make a deep copy
    import copy
    config = copy.deepcopy(config)

    # Sort parameters alphabetically for consistent comparison
    if 'parameters' in config:
        config['parameters'] = dict(sorted(config['parameters'].items()))

    return config


def compare_configs(generated: dict, existing: dict, context: str) -> bool:
    """Compare two configs and report differences."""
    gen_norm = normalize_config(generated)
    exist_norm = normalize_config(existing)

    # Check top-level keys
    gen_keys = set(gen_norm.keys())
    exist_keys = set(exist_norm.keys())

    if gen_keys != exist_keys:
        print(f"  ✗ {context}: Different top-level keys")
        print(f"    Generated: {gen_keys}")
        print(f"    Existing:  {exist_keys}")
        return False

    differences = []

    # Compare each top-level key
    for key in gen_keys:
        if key == 'parameters':
            # Special handling for parameters
            gen_params = gen_norm[key]
            exist_params = exist_norm[key]

            gen_param_keys = set(gen_params.keys())
            exist_param_keys = set(exist_params.keys())

            if gen_param_keys != exist_param_keys:
                differences.append(f"    Parameters keys differ:")
                differences.append(f"      Generated only: {gen_param_keys - exist_param_keys}")
                differences.append(f"      Existing only:  {exist_param_keys - gen_param_keys}")

            # Compare values for common parameters
            for param in gen_param_keys & exist_param_keys:
                if gen_params[param] != exist_params[param]:
                    differences.append(f"    Parameter '{param}' differs:")
                    differences.append(f"      Generated: {gen_params[param]}")
                    differences.append(f"      Existing:  {exist_params[param]}")
        else:
            if gen_norm[key] != exist_norm[key]:
                differences.append(f"    Key '{key}' differs:")
                differences.append(f"      Generated: {gen_norm[key]}")
                differences.append(f"      Existing:  {exist_norm[key]}")

    if differences:
        print(f"  ✗ {context}: Configs differ")
        for diff in differences:
            print(diff)
        return False
    else:
        print(f"  ✓ {context}: Configs match!")
        return True


def main():
    """Run tests for all sweep types and camera models."""

    # Define test cases
    test_cases = [
        ('n_train_images', 'stereo'),
        ('n_train_images', 'multiplexing'),
        ('n_train_images', 'iphone'),
        ('angles', 'stereo'),
        ('angles', 'multiplexing'),
        ('angles', 'iphone'),
        ('offset', 'stereo'),
        ('offset', 'multiplexing'),
        ('offset', 'iphone'),
    ]

    print("\n" + "="*80)
    print("Testing launch_sweep.py Config Generation")
    print("="*80 + "\n")

    passed = 0
    failed = 0

    for sweep_type, camera_model in test_cases:
        context = f"{sweep_type}/{camera_model}"
        print(f"Testing {context}...")

        try:
            generated = get_generated_config(sweep_type, camera_model)
            existing = load_existing_config(sweep_type, camera_model)

            if compare_configs(generated, existing, context):
                passed += 1
            else:
                failed += 1

        except Exception as e:
            print(f"  ✗ {context}: ERROR - {e}")
            failed += 1

        print()

    print("="*80)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    print("="*80 + "\n")

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
