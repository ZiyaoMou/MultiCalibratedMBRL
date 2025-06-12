import os
import csv

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

base_dir = "/workspace/CalibratedModelBasedRL/log"

def find_log_file(experiment_subdir):
    subdir_path = os.path.join(base_dir, experiment_subdir)
    if not os.path.exists(subdir_path):
        raise FileNotFoundError(f"Experiment folder not found: {subdir_path}")
    
    time_dirs = [d for d in os.listdir(subdir_path) 
                 if os.path.isdir(os.path.join(subdir_path, d))]
    
    if not time_dirs:
        raise FileNotFoundError(f"No runs found under: {subdir_path}")
    
    time_dirs.sort()
    
    for d in reversed(time_dirs): 
        log_path = os.path.join(subdir_path, d, "logs.mat")
        if os.path.exists(log_path):
            return log_path
    raise FileNotFoundError(f"No logs.mat found in {experiment_subdir}")

log_dirs = {
    "PE-DS-calibrate": find_log_file("halfcheetah_calibrated-1"),
    "PE-DS-no-calibrate": find_log_file("halfcheetah_uncalibrated-1"),
}


min_num_trials = 10 
plt.figure(figsize=(8, 5))

for label, path in log_dirs.items():
    if not os.path.exists(path):
        print(f"Missing: {path}")
        continue

    data = loadmat(path)
    returns = data["returns"]
    mean = np.mean(returns, axis=0) 
    std = np.std(returns, axis=0)
    x = np.arange(1, len(data["returns"][0]) + 1) * 1000
    plt.plot(x, mean, label=label)
    plt.fill_between(x, mean - std, mean + std, alpha=0.2)

plt.axhline(y=16000, linestyle="--", color="gray", label="SAC at convergence")

plt.title("HalfCheetah")
plt.xlabel("Number of Timesteps")
plt.ylabel("Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("images/halfcheetah_comparison.png")
plt.show()