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
    for i in range(0,1):  # 0 through 9
        numbered_dir = f"{experiment_base}-{i}"
        dir_path = os.path.join(base_dir, numbered_dir)
        log_path = find_latest_log_in_dir(dir_path)
        if log_path:
            log_files.append(log_path)
        print(dir_path)
    if not log_files:
        raise FileNotFoundError(f"No logs.mat found in {experiment_base}-0 through {experiment_base}-9")
    return log_files

log_dirs = {
    "Multi-Domain Calibration": find_log_files("halfcheetah_multical"),
    "Single-Domain Calibration": find_log_files("halfcheetah_singlecal"),
    "No Calibration": find_log_files("halfcheetah_uncalibrated"),
}

plt.figure(figsize=(10, 6))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green

for i, (label, paths) in enumerate(log_dirs.items()):
    all_max_returns = []
    for path in paths:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        data = loadmat(path)
        returns = data["returns"]
        returns = returns[:, :180]
        
        # Calculate maximum reward up to each point
        max_returns = np.maximum.accumulate(returns[0])
        all_max_returns.append(max_returns)
    
    if all_max_returns:
        # Find the maximum length among all runs
        max_length = max(len(arr) for arr in all_max_returns)
        print(f"{label}: max_length = {max_length}, num_runs = {len(all_max_returns)}")
        
        # Calculate mean and standard error for each timestep
        means = []
        stes = []
        
        for t in range(max_length):
            # Collect values at timestep t from all runs that have data at this timestep
            values_at_t = []
            for arr in all_max_returns:
                if t < len(arr):
                    values_at_t.append(arr[t])
            
            if values_at_t:
                mean_val = np.mean(values_at_t)
                ste_val = np.std(values_at_t) / np.sqrt(len(values_at_t))
                means.append(mean_val)
                stes.append(ste_val)
            else:
                means.append(np.nan)
                stes.append(np.nan)
        
        means = np.array(means)
        stes = np.array(stes)
        
        # Remove NaN values
        valid_mask = ~np.isnan(means)
        means = means[valid_mask]
        stes = stes[valid_mask]
        x = np.arange(1, len(means) + 1) * 1000
        
        plt.plot(x, means, label=label, color=colors[i], linewidth=2)
        plt.fill_between(x, means - stes, means + stes, alpha=0.2, color=colors[i])

plt.title("HalfCheetah: Comparison of Calibration Methods", fontsize=14, fontweight='bold')
plt.xlabel("Number of Timesteps", fontsize=12)
plt.ylabel("Maximum Reward", fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Create images directory if it doesn't exist
os.makedirs("images/multi-cal", exist_ok=True)
plt.savefig("images/multi-cal/halfcheetah_calibration_comparison.png", dpi=300, bbox_inches='tight')
plt.show()