import subprocess
from multiprocessing import Pool
import os

noise_scales = [0.05, 0.1, 0.15, 0.2, 0.25]

def run_experiment(noise_scale):
    logdir = f"log/half_noise_{noise_scale}"
    logfile = f"{logdir}/terminal.log"
    os.makedirs(logdir, exist_ok=True)
    
    command = [
        "python", "scripts/mbexp.py",
        "-env", "halfcheetah_v4",
        "-ca", "model-type", "PE",
        "-ca", "prop-type", "DS",
        "-calibrate",
        "-noise_scale", str(noise_scale),
        "-logdir", logdir,
        "-o", "exp_cfg.exp_cfg.ntrain_iters", "180"
    ]

    with open(logfile, "w") as f:
        process = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
        process.wait()

if __name__ == "__main__":
    with Pool(processes=len(noise_scales)) as pool:
        pool.map(run_experiment, noise_scales)