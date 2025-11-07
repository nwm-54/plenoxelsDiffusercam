# Launch sweeps

```bash
# Launch all camera models for a sweep type
python scripts/launch_sweep.py --sweep n_train_images --camera_models all

# Launch all sweeps at once (n_train_images, angles, offset)
./scripts/launch_all_sweeps.sh

# Preview what will be created
python scripts/launch_sweep.py --sweep angles --camera_models all --dry_run
```

## Usage

### Basic Command

```bash
python scripts/launch_sweep.py --sweep <sweep_type> --camera_models <models>
```

### Arguments

- `--sweep`: Sweep type (`n_train_images`, `angles`, `offset`)
- `--camera_models`: Camera models to launch (`stereo`, `multiplexing`, `iphone`, or `all`)
- `--entity`: Wandb entity/team (optional)
- `--project`: Wandb project name (default: `multiplexed-pixels`)
- `--dry_run`: Preview configs without creating sweeps
- `--no_agent`: Create sweeps but don't launch agents
- `--agent_count`: Number of runs per agent (default: 1, -1 for unlimited)

### Examples

**Launch specific camera models:**
```bash
python scripts/launch_sweep.py --sweep n_train_images --camera_models stereo iphone
```

**Create sweeps without launching agents:**
```bash
python scripts/launch_sweep.py --sweep angles --camera_models all --no_agent
```

**Launch with custom entity/project:**
```bash
python scripts/launch_sweep.py \
    --sweep offset \
    --camera_models all \
    --entity shamus-team \
    --project gs7
```

## Available Sweeps

Sweep configurations are defined in `launch_sweep.py` (no YAML files needed).

| Sweep Type | Description | Parameters |
|------------|-------------|------------|
| `n_train_images` | Vary number of training images | 1-50 training images across 8 datasets |
| `angles` | Vary camera angles | 0.2-20 degrees with blender rendering |
| `offset` | Vary camera offsets | -500 to 2mm offsets for stereo baseline |

## Camera Models

| Model | Description |
|-------|-------------|
| `stereo` | Stereo camera setup |
| `multiplexing` | Multiplexing/lightfield cameras |
| `iphone` | iPhone multi-camera setup |
| `all` | All camera models |

## How It Works

The script:
1. Reads your existing YAML files as templates
2. Generates camera-specific configs automatically
3. Creates wandb sweeps via `wandb sweep`
4. Optionally launches agents via `wandb agent`

Camera-specific configurations are handled automatically:

| Model | Flag | Additional Parameters |
|-------|------|----------------------|
| stereo | `use_stereo: true` | None |
| multiplexing | `use_multiplexing: [true, false]` | `dls: [12, 20]` |
| iphone | `use_iphone: true` | `iphone_same_focal_length` |

## Running Agents

After launching sweeps, the sweep IDs are printed to stdout. Copy them into your SLURM script or run agents manually.

### SLURM

Copy the printed sweep IDs into your existing SLURM script (e.g., `multiviews_slurm.slurm`):

```bash
# Example: Add sweep IDs to your SLURM script
SWEEP_IDS=(
    "abc123xyz"
    "def456uvw"
    "ghi789rst"
)
```

Then submit: `sbatch multiviews_slurm.slurm`

### Manual Agents

Run agents using the printed commands:

```bash
wandb agent shamus-team/multiplexed-pixels/<sweep_id>
```

Or in parallel across terminals/tmux sessions.

## Environment Variables

Set these for convenience:

```bash
export WANDB_ENTITY="shamus-team"
export WANDB_PROJECT="multiplexed-pixels"
```

Then omit `--entity` and `--project` flags.

## Example Output

When you launch sweeps, you'll see:

```
================================================================================
SWEEP IDS
================================================================================

abc123xyz
def456uvw
ghi789rst

================================================================================
AGENT COMMANDS
================================================================================

# stereo
wandb agent shamus-team/multiplexed-pixels/abc123xyz
# multiplexing
wandb agent shamus-team/multiplexed-pixels/def456uvw
# iphone
wandb agent shamus-team/multiplexed-pixels/ghi789rst

================================================================================
```

Copy the sweep IDs into your SLURM script as needed.

## Testing

Validate that generated configs match existing YAMLs:

```bash
python test_sweep_launcher.py
```

All 9 configurations (3 sweep types × 3 camera models) are tested.

## Adding New Sweep Types

Add a new entry to `SWEEP_CONFIGS` in `launch_sweep.py`:

```python
SWEEP_CONFIGS = {
    # ... existing configs ...
    "my_new_sweep": {
        "program": "train_sim_multiviews.py",
        "method": "grid",
        "parameters": {
            "source_path": {"values": DATASETS},
            "my_param": {"values": [1, 2, 3]},
            # ... other parameters
        },
        "command": [
            "${interpreter}",
            "${program}",
            "--iterations", "3000",
            "${args_no_boolean_flags}",
        ],
    },
}
```

Then launch: `python scripts/launch_sweep.py --sweep my_new_sweep --camera_models all`

## Shared Reconstruction Pipeline

Use `preprocess_for_gsplat.py` with an optional `--eval-dir` to rebuild training
and evaluation captures into a single COLMAP/VGGT reconstruction so every
subset shares the same pose frame. It accepts either pre-extracted `images/`, a
folder of videos (frames are extracted with ffmpeg), or a single `.lfr` file
(decoded via `lf_to_colmap.py`). In shared mode the combined `images/` and
`sparse/` live at the output root, and two subset folders `train/` and `test/`
are created with their own `images/` and `sparse/`.

```
python scripts/preprocess_for_gsplat.py \
    --input ../gs7/input_data/dog/iphone4 \
    --eval-dir ../gs7/input_data/dog/eval/test \
    --output-dir ../gs7/input_data/dog/iphone4_shared \
    --train-prefix train \
    # Covisible masks are computed by default when --eval-dir is present.
```
Or run on a cluster via SLURM:

```
sbatch scripts/preprocess_for_gsplat.slurm \
    --input ../gs7/input_data/dog/iphone4 \
    --eval-dir ../gs7/input_data/dog/eval/test
```

Key features:

- Reuses `preprocess_for_gsplat` helpers to run VGGT on the union of all frames.
- Flattens images into `images/` with subset prefixes (`train__`, `test__`, …).
- Produces per-subset dataset folders `train/{images,sparse}` and
  `test/{images,sparse}` that only contain the relevant views while keeping the
  shared reference frame.
- Records split metadata in `metadata/summary.json` and `splits/<subset>.txt`.
- Optionally precomputes covisible masks under `<output>/covisible/<subset>` that can be
  symlinked into run directories.

### Training with shared datasets

After preprocessing, launch the dual trainer using the combined root as `DATA_DIR`
and the `test/` directory for external evaluation:

```
MATCH=train
DATA=../gs7/input_data/dog/iphone4_shared
EVAL=$DATA/test
sbatch examples/dual_training.slurm "$DATA" "$MATCH" "$EVAL"
```

- `MATCH` should match the `train-prefix` used during preprocessing (see
  `metadata/summary.json` for the generated token).
- When `EVAL_DIR` differs from `DATA_DIR`, the SLURM script automatically clears
  the evaluation match string so the hold-out subset is consumed in full. Set
  `EVAL_MATCH_STRING=<token>` if you need to override this behaviour.
- Any precomputed covisible masks under `$DATA/covisible/<subset>` are symlinked
  into each run (`$RESULT_DIR/covisible/<subset>`), so the trainer reuses them
  without re-running RAFT. Dual-training jobs now generate alignment `.npz` files
  on demand inside `$RESULT_DIR/alignments`.

Run `python scripts/preprocess_for_gsplat.py --help` for the full list of
options (copy mode, include lists, chunk sizing, etc.).
