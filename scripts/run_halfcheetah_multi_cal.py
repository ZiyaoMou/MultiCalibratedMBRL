import subprocess, os, time
ENV         = "halfcheetah"
BASE_CMD    = f"python3 scripts/mbexp.py -env {ENV} -ca model-type PE -ca prop-type DS"
ROLLOUT_LEN = 1000
N_ITERS     = 180
EMB_DIM     = 3
N_RUNS      = 1
LOG_ROOT    = "log"

OVR = (f"-o exp_cfg.sim_cfg.task_hor {ROLLOUT_LEN} "
       f"-o exp_cfg.exp_cfg.ntrain_iters {N_ITERS}")

def launch(job_name: str, extra_flags: str):
    logdir = os.path.join(LOG_ROOT, job_name)
    cmd = f"{BASE_CMD} -logdir {logdir} {OVR} {extra_flags}"
    print("Launching:", cmd)
    return subprocess.Popen(cmd, shell=True)

def main():
    os.makedirs(LOG_ROOT, exist_ok=True)
    processes = []

    for i in range(N_RUNS):
        print(f"\n=== Launching run {i} ===")

        processes.append(launch(f"halfcheetah_uncalibrated-{i}", ""))
        processes.append(launch(f"halfcheetah_singlecal-{i}", "-calibrate"))
        processes.append(launch(f"halfcheetah_multical-{i}", f"-calibrate -emb_dim {EMB_DIM}"))

    print(f"\nAll {3 * N_RUNS} jobs launched. Waiting for them to finish...")
    for p in processes:
        p.wait()

    print("\nAll experiments completed.")

if __name__ == "__main__":
    main()