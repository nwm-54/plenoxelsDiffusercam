#!/bin/bash

# Define the parameters for each flag
r_values=(1 2 4 8)
# r_values=(8)
d_values=(14 18 22)
# d_values=()
# d_values=(24)
s_values=("chair" "drums" "materials" "mic" "ship" "ficus" "lego_gen12" "hotdog") 
tv_values=(0.01 1.0 0.0)
tv_unseen_values=(0.01 0.1 0.0)
# s_values=("mic" "ship" "ficus" "lego_gen12") 
# s_values=("hotdog") 
# s_values=("lego_gen12") 
# log_file="/home/vitran/gs7/nerf_synthetic_execution_log2.txt"

# Ensure the log file is empty before starting
> "$log_file"

# Iterate over all combinations of parameters
source ~/.bashrc
cd /home/wl757/multiplexed-pixels/gs7
conda activate gaussian_splatting


for s in "${s_values[@]}"; do
  log_file="/share/monakhova/shamus_data/multiplexed_pixels/logs/nerf_${s}.txt"
  touch "$log_file"
  echo "created ${log_file}" 

  for d in "${d_values[@]}"; do
    for r in "${r_values[@]}"; do
      for tv in "${tv_values[@]}"; do
        for tv_unseen in "${tv_unseen_values[@]}"; do
          # Construct the model name and command
          m_value="${s}_$((800/r))_dls${d}_tv${tv}_tv_unseen${tv_unseen}"
          cmd="python3 train_sim_plain.py -s /home/wl757/multiplexed-pixels/plenoxels/blender_data/${s} --iterations 3000 -r ${r} --dls ${d} --tv_weight ${tv} --tv_unseen_weight ${tv_unseen}"

          # Log the command being executed
          echo "Executing: $cmd" | tee -a "$log_file"

          # Execute the command and append stdout and stderr to the log file
          eval "$cmd" >> "$log_file" 2>&1

          # Check if the command succeeded
          if [[ $? -ne 0 ]]; then
            echo "Command failed: $cmd" | tee -a "$log_file"
          else
            echo "Command completed successfully: $cmd" | tee -a "$log_file"
          fi

          # Add a separator in the log file for readability
          echo "------------------------------------------------------------" | tee -a "$log_file"
        done
      done
  done
done

echo "Execution completed. Logs are available in $log_file"
