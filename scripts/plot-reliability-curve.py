import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from scipy.stats import norm
from dotmap import DotMap

from dmbrl.modeling.models import BNN
from dmbrl.modeling.layers import FC, RecalibrationLayer
import dmbrl.env  
import gym

MODEL_DIR  = "log/1_cartpole_uncalibrated-0/2025-07-01--21:29:47"
MODEL_NAME = "model"
MAT_FILE   = os.path.join(MODEL_DIR, MODEL_NAME + ".mat")
NUM_STEPS  = 500
SEED       = 0

def collect_random_data(env, n_steps, seed=0):
    rng = np.random.default_rng(seed)
    obs_buf, act_buf, next_obs_buf = [], [], []

    # reset
    reset_out = env.reset()
    o = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    for _ in range(n_steps):
        a = env.action_space.sample()
        step_out = env.step(a)

        if len(step_out) == 5:                           # Gym ≥0.26
            o2, _, terminated, truncated, _ = step_out
            done = terminated or truncated
        else:                                           # Gym ≤0.25
            o2, _, done, _ = step_out

        obs_buf.append(o)
        act_buf.append(a)
        next_obs_buf.append(o2)

        if done:
            reset_out = env.reset()
            o = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        else:
            o = o2

    return (np.asarray(obs_buf, dtype=np.float32),
            np.asarray(act_buf, dtype=np.float32),
            np.asarray(next_obs_buf, dtype=np.float32))

def obs_preproc(obs):
    return np.concatenate([
        np.sin(obs[:, 1:2]),
        np.cos(obs[:, 1:2]),
        obs[:, :1],
        obs[:, 2:],
    ], axis=1)


def compute_calibration_error(cdf_pred, cdf_emp, n_bins=15):
    N, D = cdf_pred.shape
    ece_list, mce_list = [], []

    for d in range(D):
        prob_pred = cdf_pred[:, d]
        prob_true = cdf_emp[:, d]

        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(prob_pred, bins) - 1

        ece = 0.0
        mce = 0.0

        for i in range(n_bins):
            idx = bin_indices == i
            if np.any(idx):
                avg_conf = prob_pred[idx].mean()
                avg_acc = prob_true[idx].mean()
                bin_error = np.abs(avg_acc - avg_conf)
                ece += (idx.sum() / N) * bin_error
                mce = max(mce, bin_error)

        ece_list.append(ece)
        mce_list.append(mce)

    return ece_list, mce_list

def test_calibration_sampling(bnn, X, calibrate=True):
    with bnn.sess.as_default():
        expected_dim = bnn.sess.run(bnn.scaler.mu).shape[1]

    if X.shape[1] > expected_dim:
        X = X[:, :expected_dim]
    elif X.shape[1] < expected_dim:
        pad = np.zeros((X.shape[0], expected_dim - X.shape[1]))
        X = np.concatenate([X, pad], axis=1)
    inputs = tf.placeholder(shape=X.shape, dtype=tf.float32, name="X")
    mean_tf, var_tf = bnn.create_prediction_tensors(inputs)
    preds_tf = bnn.sample_predictions(mean_tf, var_tf, calibrate=calibrate)

    with bnn.sess.as_default():
        preds, mean, var = bnn.sess.run([preds_tf, mean_tf, var_tf],
                                        feed_dict={inputs: X})

    cdf_pred = norm.cdf(preds, loc=mean, scale=np.sqrt(var))
    
    # Apply calibration if enabled
    if calibrate:
        if bnn.recalibrator.emb_dim is None:
            # Single-domain calibration: pass zeros as domain vector (consistent with training)
            domain_vec = np.zeros((cdf_pred.shape[0], 1), dtype=np.float32)
            cdf_pred = bnn.sess.run(bnn.recalibrator(cdf_pred, domain_vec=domain_vec))
        else:
            # Multi-domain calibration: need domain vectors
            cdf_pred = bnn.sess.run(bnn.recalibrator(cdf_pred))

    N, D = cdf_pred.shape
    cdf_emp = np.zeros_like(cdf_pred)
    for d in range(D):
        sorted_p = np.sort(cdf_pred[:, d])
        ranks = np.searchsorted(sorted_p, cdf_pred[:, d], side='right')
        cdf_emp[:, d] = ranks / N

    return cdf_pred, cdf_emp

def plot_reliability_curve(cdf_pred, cdf_emp, dim=0):
    os.makedirs("images/multi-cal", exist_ok=True)
    
    ps = np.linspace(0.01, 0.99, 50)
    emp = [np.mean(cdf_emp[:, dim][cdf_pred[:, dim] <= q]) for q in ps]
    plt.figure(figsize=(5,3))
    plt.plot(ps, emp, 'o-', label="Empirical")
    plt.plot([0,1],[0,1],'k--', label="Ideal")
    plt.title(f"Cartpole: uncalibrated BNN Reliability Dim {dim}")
    plt.xlabel("Predicted CDF level q")
    plt.ylabel("Pr(y ≤ quantile(q))")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/multi-cal/cartpole_reliability_curve_dim{dim}-uncalibrated.png")
    plt.close()

def main():
    
    env = gym.make("MBRLCartpole-v0")
    obss, acts, next_obss = collect_random_data(env, NUM_STEPS, SEED)
    print(f"Collected {len(obss)} data points")

    params = DotMap(
        name=MODEL_NAME,
        model_dir=MODEL_DIR,
        load_model=True,
        num_networks=5,
        emb_dim=None,
        cal_hidden=16
    )
    model = BNN(params)
    model.finalize(tf.train.AdamOptimizer, {"learning_rate":1e-3})
    print("Loaded model from", MAT_FILE)

    X_obs = obs_preproc(obss)
    acts = acts.squeeze() 
    if acts.ndim == 1:
        acts = acts[:, None]
    else:
        acts = acts.reshape(acts.shape[0], -1) 
    X = np.concatenate([X_obs, acts], axis=1)
    print(f"Input shape: {X.shape}")

    # For uncalibrated model, use calibrate=False
    cdf_pred, cdf_emp = test_calibration_sampling(model, X, calibrate=False)
    print(f"CDF prediction shape: {cdf_pred.shape}")
    
    for d in range(2):
        plot_reliability_curve(cdf_pred, cdf_emp, dim=d)
        print(f"Saved reliability curve for dimension {d}")

    ece_list, mce_list = compute_calibration_error(cdf_pred, cdf_emp)
    print(f"ECE: {ece_list}")
    print(f"MCE: {mce_list}")
    print("Reliability curves saved to images/multi-cal/")

if __name__ == "__main__":
    main()