import os
import csv

import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

base_dir = "/workspace/CalibratedModelBasedRL/log"

def find_latest_log_in_dir(dir_path):
    if not os.path.exists(dir_path):
        return None
    
    time_dirs = [d for d in os.listdir(dir_path) 
                 if os.path.isdir(os.path.join(dir_path, d))]
    
    if not time_dirs:
        return None
    
    time_dirs.sort()
    
    for d in reversed(time_dirs): 
        log_path = os.path.join(dir_path, d, "logs.mat")
        if os.path.exists(log_path):
            return log_path
    return None

def find_log_files(experiment_base):
    log_files = []
    
    # Try numbered directories from 0 to 9
    for i in range(4):  # 0 through 9
        numbered_dir = f"{experiment_base}-{i}"
        dir_path = os.path.join(base_dir, numbered_dir)
        log_path = find_latest_log_in_dir(dir_path)
        if log_path:
            log_files.append(log_path)
    
    if not log_files:
        raise FileNotFoundError(f"No logs.mat found in {experiment_base}-0 through {experiment_base}-9")
    return log_files

log_dirs = {
    "PE-DS-calibrate": find_log_files("1_cartpole_calibrated"),
    "PE-DS-no-calibrate": find_log_files("1_cartpole_uncalibrated"),
}

plt.figure(figsize=(8, 5))

for label, paths in log_dirs.items():
    all_max_returns = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        data = loadmat(path)
        returns = data["returns"]
        returns = returns[:, :15]
        
        # Calculate maximum reward up to each point
        max_returns = np.maximum.accumulate(returns[0])
        all_max_returns.append(max_returns)
    
    if all_max_returns:
        # Stack all max returns from different runs
        stacked_max_returns = np.vstack(all_max_returns)
        mean = np.mean(stacked_max_returns, axis=0)
        ste = np.std(stacked_max_returns, axis=0) / np.sqrt(stacked_max_returns.shape[0])
        x = np.arange(1, len(mean) + 1) * 200
        
        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - ste, mean + ste, alpha=0.2)

plt.title("cartpole")
plt.xlabel("Number of Timesteps")
plt.ylabel("Maximum Reward")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig("images/1-cartpole_comparison-max.png")
plt.show()