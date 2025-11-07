  #!/usr/bin/env bash
  set -euo pipefail

  # stereo (fps 20)

  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/action-figure/stereo-train --eval-dir ../gs7/dataset/action-figure/stereo-eval --output-dir ../gs7/dataset/action-figure/stereo --fps 20
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/ball/stereo-train          --eval-dir ../gs7/dataset/ball/stereo-eval          --output-dir ../gs7/dataset/ball/stereo          --fps 20
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/chicken/stereo-train       --eval-dir ../gs7/dataset/chicken/stereo-eval       --output-dir ../gs7/dataset/chicken/stereo       --fps 20
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/dog/stereo-train           --eval-dir ../gs7/dataset/dog/stereo-eval           --output-dir ../gs7/dataset/dog/stereo           --fps 20
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/espresso/stereo-train      --eval-dir ../gs7/dataset/espresso/stereo-eval      --output-dir ../gs7/dataset/espresso/stereo      --fps 20
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/optics/stereo-train        --eval-dir ../gs7/dataset/optics/stereo-eval        --output-dir ../gs7/dataset/optics/stereo        --fps 20
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/salt-pepper/stereo-train   --eval-dir ../gs7/dataset/salt-pepper/stereo-eval   --output-dir ../gs7/dataset/salt-pepper/stereo   --fps 20
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/shelf/stereo-train         --eval-dir ../gs7/dataset/shelf/stereo-eval         --output-dir ../gs7/dataset/shelf/stereo         --fps 20

  # iphone (fps 10)

  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/action-figure/iphone-train --eval-dir ../gs7/dataset/action-figure/iphone-eval --output-dir ../gs7/dataset/action-figure/iphone --fps 10
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/ball/iphone-train          --eval-dir ../gs7/dataset/ball/iphone-eval          --output-dir ../gs7/dataset/ball/iphone --fps 10
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/chicken/iphone-train       --eval-dir ../gs7/dataset/chicken/iphone-eval       --output-dir ../gs7/dataset/chicken/iphone --fps 10
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/dog/iphone-train           --eval-dir ../gs7/dataset/dog/iphone-eval           --output-dir ../gs7/dataset/dog/iphone --fps 10
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/espresso/iphone-train      --eval-dir ../gs7/dataset/espresso/iphone-eval      --output-dir ../gs7/dataset/espresso/iphone --fps 10
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/optics/iphone-train        --eval-dir ../gs7/dataset/optics/iphone-eval        --output-dir ../gs7/dataset/optics/iphone --fps 10
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/salt-pepper/iphone-train   --eval-dir ../gs7/dataset/salt-pepper/iphone-eval   --output-dir ../gs7/dataset/salt-pepper/iphone --fps 10
  sbatch --requeue scripts/preprocess_for_gsplat.slurm --input ../gs7/dataset/shelf/iphone-train         --eval-dir ../gs7/dataset/shelf/iphone-eval         --output-dir ../gs7/dataset/shelf/iphone --fps 10