import subprocess

BASE_CMD = "python3 scripts/mbexp.py -env cartpole -ca model-type PE -ca prop-type DS"
NUM_ITER = 15

def run_job(logdir, calibrated):
    cmd = f"{BASE_CMD} -logdir {logdir}"
    if calibrated:
        cmd += " -calibrate"
    cmd += f" -o exp_cfg.exp_cfg.ntrain_iters {NUM_ITER}"
    print(f"Launching: {cmd}")
    return subprocess.Popen(cmd, shell=True)

def main():
    for i in range(1, 10):
        print(f"\n=== Starting run {i}: launching uncalibrated and calibrated in parallel ===")

        proc_uncal = run_job(f"log/1_cartpole_uncalibrated-{i}", calibrated=False)
        proc_cal = run_job(f"log/1_cartpole_calibrated-{i}", calibrated=True)

        proc_uncal.wait()
        proc_cal.wait()

    print("\n All training runs completed.")

if __name__ == "__main__":
    main()