#!/bin/bash

# Define the parameters for each flag
r_values=(1 2 4 8)
# r_values=(8)
d_values=(10 12 14 16 18 20 22 24)
# d_values=()
# d_values=(24)
# s_values=("chair" "drums" "materials" "mic" "ship" "ficus" "lego_gen12" "hotdog") 
# s_values=("mic" "ship" "ficus" "lego_gen12") 
# s_values=("hotdog") 
s_values=("lego_gen12") 
# log_file="/home/vitran/gs7/nerf_synthetic_execution_log2.txt"

# Ensure the log file is empty before starting
# > "$log_file"

# Iterate over all combinations of parameters
for s in "${s_values[@]}"; do
  log_file="/home/wl757/multiplexed-pixels/gs7/logs2/nerf_${s}_execution_log2_multiviews.txt"
  touch "$log_file"
  echo "created ${log_file}" 

  for d in "${d_values[@]}"; do
    for r in "${r_values[@]}"; do
      m_value="${s}_$((800/r))_dls${d}_multiviews"
      cmd="python3 train_sim_multiviews.py -s /home/wl757/multiplexed-pixels/plenoxels/blender_data/${s} -m ./output5/${m_value} --iterations 3001 -r ${r} --dls ${d} --tv_weight 1 --tv_unseen_weight 0.05  --views_index 50 59 60 70 90 "

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

echo "Execution completed. Logs are available in $log_file"
