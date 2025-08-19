import subprocess
import os

cuda_device = "0"  # Change this to specify a different GPU if needed
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
# Define the parameters for each flag
r_values = [1, 2, 4, 8, 16]
s_values = ["chair", "drums", "materials", "mic", "ship", "ficus"]
m_values = [f"{item}_800" for item in s_values]

# Define the log file path
log_file = "/home/vitran/gs7/nerf_synthetic_execution.log"

# Ensure the log file is empty before starting
with open(log_file, "w") as f:
    f.write("")

# Iterate over all combinations of parameters
for s in s_values:
    for m in m_values:
        if not m.startswith(s):
            continue  # Ensure `-m` aligns with `-s` (e.g., chair with chair_800)
        for r in r_values:
            cmd = [
                "python3",
                "train_sim_plain.py",
                "-s",
                f"/home/vitran/plenoxels/blender_data/{s}",
                "-m",
                f"./output/{m}",
                "--iterations",
                "3000",
                "-r",
                str(r),
            ]

            # Combine Conda activation and the command
            full_cmd = [
                "bash",
                "-c",
                f"source /home/vitran/miniconda3/etc/profile.d/conda.sh && conda activate gaussian_splatting && {' '.join(cmd)}",
            ]

            # Execute the command and write stdout incrementally
            try:
                with open(log_file, "a") as log:
                    log.write(f"Command: {' '.join(cmd)}\\n")
                    process = subprocess.Popen(
                        full_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    for line in process.stdout:
                        log.write(line)  # Write stdout incrementally
                        log.flush()  # Ensure it's written to the file immediately

                    stderr = process.communicate()[1]
                    log.write(f"STDERR:\\n{stderr}\\n")
                    log.write("-" * 80 + "\\n")

            except Exception as e:
                with open(log_file, "a") as log:
                    log.write(f"Failed to execute: {' '.join(cmd)}\\n")
                    log.write(f"Error: {str(e)}\\n")
                    log.write("-" * 80 + "\\n")

print(f"Execution completed. Logs are available in {log_file}")
