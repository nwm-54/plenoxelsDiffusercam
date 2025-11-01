# WandB Hyperparameter Sweep IDs

All sweeps launched to project: `shamus-team/hyperparameters`

## Sweep IDs by Configuration

| Sweep Name | Config File | Sweep ID | Agent Command | Views | Metric |
|------------|-------------|----------|---------------|-------|--------|
| **Stereo 1v** | `sweep_stereo_1v.yaml` | `q0obf4ky` | `wandb agent shamus-team/hyperparameters/q0obf4ky` | 1 | psnr/adjacent test camera |
| **Stereo 3v** | `sweep_stereo_3v.yaml` | `q74vg6fo` | `wandb agent shamus-team/hyperparameters/q74vg6fo` | 3 | psnr/full test camera |
| **iPhone 1v** | `sweep_iphone_1v.yaml` | `4usvwnzn` | `wandb agent shamus-team/hyperparameters/4usvwnzn` | 1 | psnr/adjacent test camera |
| **iPhone 3v** | `sweep_iphone_3v.yaml` | `vy0q2anm` | `wandb agent shamus-team/hyperparameters/vy0q2anm` | 3 | psnr/full test camera |
| **LightField DLS12 1v** | `sweep_lightfield_dls12_1v.yaml` | `d53voo7q` | `wandb agent shamus-team/hyperparameters/d53voo7q` | 1 | psnr/adjacent test camera |
| **LightField DLS12 3v** | `sweep_lightfield_dls12_3v.yaml` | `vxfnyelf` | `wandb agent shamus-team/hyperparameters/vxfnyelf` | 3 | psnr/full test camera |
| **Multiplexing DLS20 1v** | `sweep_multiplexing_dls20_1v.yaml` | `mi5rnyyz` | `wandb agent shamus-team/hyperparameters/mi5rnyyz` | 1 | psnr/adjacent test camera |
| **Multiplexing DLS20 3v** | `sweep_multiplexing_dls20_3v.yaml` | `f95l04pd` | `wandb agent shamus-team/hyperparameters/f95l04pd` | 3 | psnr/full test camera |

## Direct Sweep URLs

- Stereo 1v: https://wandb.ai/shamus-team/hyperparameters/sweeps/q0obf4ky
- Stereo 3v: https://wandb.ai/shamus-team/hyperparameters/sweeps/q74vg6fo
- iPhone 1v: https://wandb.ai/shamus-team/hyperparameters/sweeps/4usvwnzn
- iPhone 3v: https://wandb.ai/shamus-team/hyperparameters/sweeps/vy0q2anm
- LightField DLS12 1v: https://wandb.ai/shamus-team/hyperparameters/sweeps/d53voo7q
- LightField DLS12 3v: https://wandb.ai/shamus-team/hyperparameters/sweeps/vxfnyelf
- Multiplexing DLS20 1v: https://wandb.ai/shamus-team/hyperparameters/sweeps/mi5rnyyz
- Multiplexing DLS20 3v: https://wandb.ai/shamus-team/hyperparameters/sweeps/f95l04pd

## Running the SLURM Job

To launch all 40 agents (5 per sweep):

```bash
sbatch scripts/run_all_sweeps.slurm
```

This will launch an array job with 40 tasks (indices 0-39), where:
- Tasks 0-4: 5 agents for Stereo 1v
- Tasks 5-9: 5 agents for Stereo 3v
- Tasks 10-14: 5 agents for iPhone 1v
- Tasks 15-19: 5 agents for iPhone 3v
- Tasks 20-24: 5 agents for LightField DLS12 1v
- Tasks 25-29: 5 agents for LightField DLS12 3v
- Tasks 30-34: 5 agents for Multiplexing DLS20 1v
- Tasks 35-39: 5 agents for Multiplexing DLS20 3v

## Manual Agent Launch

To manually launch an agent for a specific sweep:

```bash
wandb agent shamus-team/hyperparameters/<sweep_id>
```

For example:
```bash
wandb agent shamus-team/hyperparameters/vaoq70p1
```
