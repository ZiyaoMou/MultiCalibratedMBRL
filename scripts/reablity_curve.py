import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from scipy.stats import norm
from dotmap import DotMap

from dmbrl.modeling.models import BNN
import dmbrl.env  
import gym

UNCAL_MODEL_DIR = "log/1_cartpole_uncalibrated-0/2025-07-02--02:38:44"
CAL_MODEL_DIR   = "log/1_cartpole_calibrated-0/2025-07-02--02:38:44"
MODEL_NAME      = "model"
NUM_STEPS       = 500
SEED            = 0

def collect_random_data(env, n_steps, seed=0):
    np.random.seed(seed)
    obs_buf, act_buf, next_obs_buf = [], [], []
    o = env.reset()
    for _ in range(n_steps):
        a = env.action_space.sample()
        o2, _, done, _ = env.step(a)
        obs_buf.append(o); act_buf.append(a); next_obs_buf.append(o2)
        o = env.reset() if done else o2
    return np.array(obs_buf), np.array(act_buf), np.array(next_obs_buf)

def obs_preproc(obs):
    return np.concatenate([
        np.sin(obs[:, 1:2]),
        np.cos(obs[:, 1:2]),
        obs[:, :1],
        obs[:, 2:],
    ], axis=1)

def compute_calibration(model, X, y_true, calibrate):
    with model.sess.graph.as_default():
        inputs = tf.placeholder(shape=X.shape, dtype=tf.float32, name="x_in")
        mean_tf, var_tf = model.create_prediction_tensors(inputs)
        preds_tf = model.sample_predictions(mean_tf, var_tf, calibrate=calibrate)
    with model.sess.as_default():
        preds, mean, var = model.sess.run(
            [preds_tf, mean_tf, var_tf], feed_dict={inputs: X}
        )

    if y_true.ndim == 1:
        y_true = y_true[:, None] 
        
    std = np.sqrt(np.maximum(var, 1e-9))        
    cdf_pred = norm.cdf(y_true, loc=mean, scale=std)

    N, D = cdf_pred.shape
    cdf_emp = np.zeros_like(cdf_pred)
    for d in range(D):
        sorted_p = np.sort(cdf_pred[:, d])
        ranks = np.searchsorted(sorted_p, cdf_pred[:, d], side='right')
        cdf_emp[:, d] = ranks / N

    return cdf_pred, cdf_emp

def compute_ece(cdf_pred, cdf_emp, n_bins=15):
    N, D = cdf_pred.shape
    ece_list = []
    for d in range(D):
        pred = cdf_pred[:, d]
        true = cdf_emp[:, d]
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            idx = (pred >= bins[i]) & (pred < bins[i+1])
            if np.any(idx):
                acc = true[idx].mean()
                conf = pred[idx].mean()
                ece += (len(idx.nonzero()[0]) / N) * np.abs(acc - conf)
        ece_list.append(ece)
    return ece_list

def plot_reliability_curve(pred1, emp1, pred2, emp2, dim=0):
    os.makedirs("images", exist_ok=True)
    ps = np.linspace(0.01, 0.99, 50)
    emp_uncal = [np.mean(emp1[:, dim][pred1[:, dim] <= q]) for q in ps]
    emp_cal   = [np.mean(emp2[:, dim][pred2[:, dim] <= q]) for q in ps]

    plt.figure(figsize=(4.5,3.5))
    plt.plot([0,1], [0,1], 'k--', label='Ideal')
    plt.plot(ps, emp_uncal, 'o-', label='Uncalibrated')
    plt.plot(ps, emp_cal,   's-', label='Calibrated')
    plt.xlabel("Predicted CDF")
    plt.ylabel("Empirical CDF")
    plt.title(f"Reliability – Dim {dim}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"images/reliability_comparison_dim{dim}.png")
    plt.close()

def compute_ece_scalar(cdf_pred, cdf_emp, n_bins=15):
    N, D = cdf_pred.shape
    total_ece = 0.0

    for d in range(D):
        pred = cdf_pred[:, d]
        true = cdf_emp[:, d]
        bins = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        for i in range(n_bins):
            idx = (pred >= bins[i]) & (pred < bins[i+1])
            if np.any(idx):
                acc = true[idx].mean()
                conf = pred[idx].mean()
                ece += (len(idx.nonzero()[0]) / N) * np.abs(acc - conf)
        total_ece += ece

    return total_ece / D

def main():
    env = gym.make("MBRLCartpole-v0")
    obss, acts, next_obss = collect_random_data(env, NUM_STEPS, SEED)
    X_obs = obs_preproc(obss)
    acts = acts.squeeze()
    if acts.ndim == 1:
        acts = acts[:, None]
    X = np.concatenate([X_obs, acts], axis=1)
    
    y_true = (next_obss - obss)
    
    print(f"[Info] X shape: {X.shape}")
    print(f"[Info] y_true shape: {y_true.shape}")

    graph1 = tf.Graph()
    graph2 = tf.Graph()

    with graph1.as_default():
        sess1 = tf.Session()
        with sess1.as_default():
            params_uncal = DotMap(name=MODEL_NAME, model_dir=UNCAL_MODEL_DIR, load_model=True, num_networks=5)
            model_uncal = BNN(params_uncal)
            model_uncal.finalize(tf.train.AdamOptimizer, {"learning_rate":1e-3})
            pred_uncal, emp_uncal = compute_calibration(model_uncal, X, y_true, calibrate=False)
        sess1.close()

    with graph2.as_default():
        sess2 = tf.Session()
        with sess2.as_default():
            params_cal = DotMap(name=MODEL_NAME, model_dir=CAL_MODEL_DIR, load_model=True, num_networks=5)
            model_cal = BNN(params_cal)
            model_cal.finalize(tf.train.AdamOptimizer, {"learning_rate":1e-3})
            pred_cal, emp_cal = compute_calibration(model_cal, X, y_true, calibrate=True)
        sess2.close()

    for d in range(pred_uncal.shape[1]):
        plot_reliability_curve(pred_uncal, emp_uncal, pred_cal, emp_cal, dim=d)

    ece_uncal = compute_ece(pred_uncal, emp_uncal)
    ece_cal   = compute_ece(pred_cal, emp_cal)
    for d in range(len(ece_uncal)):
        print(f"[Dim {d}] ECE (Uncal): {ece_uncal[d]:.4f} | ECE (Cal): {ece_cal[d]:.4f}")
    print(f"Overall ECE (Uncal): {compute_ece_scalar(pred_uncal, emp_uncal):.4f} | Overall ECE (Cal): {compute_ece_scalar(pred_cal, emp_cal):.4f}")

if __name__ == "__main__":
    main()