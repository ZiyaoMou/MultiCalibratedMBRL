import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow_probability as tfp
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from dotmap import DotMap

from dmbrl.modeling.models import BNN
from dmbrl.modeling.layers import FC

NUM_SAMPLES = 1024
IN_DIM, OUT_DIM, DOMAIN_DIM, HIDDEN_DIM = 100, 2, 5, 10
ECE_BINS = 20      # number of bins for Expected Calibration Error

def stub_data_with_domains():
    X = np.random.uniform(-1, 1, size=(NUM_SAMPLES, IN_DIM))
    W_hidden, W_last = np.random.randn(IN_DIM, HIDDEN_DIM), np.random.randn(HIDDEN_DIM, OUT_DIM)
    y_mid = np.matmul(X, W_hidden) + 5
    y_mid[y_mid < 0] = 0
    y = np.matmul(y_mid, W_last) + 2
    domains = np.random.uniform(-1, 1, size=(NUM_SAMPLES, DOMAIN_DIM))
    return X, y, domains


def create_bnn_with_domain(X, y, domains):
    params = DotMap(name="bnn_domain", domain_emb_dim=DOMAIN_DIM, cal_hidden=16)
    model = BNN(params)
    model.add(FC(OUT_DIM, input_dim=IN_DIM, weight_decay=5e-4))
    model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.03})
    model.train(X, y, epochs=1000, batch_size=64)
    model.calibrate(X, y, domains=domains)
    return model

def compute_ece(pred_cdf, true_emp, n_bins=ECE_BINS):
    """
    pred_cdf, true_emp: shape (N,) – predicted CDF & empirical CDF at same points
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (pred_cdf >= lo) & (pred_cdf < hi)
        if mask.any():
            gap = abs(true_emp[mask].mean() - ((lo + hi) / 2.0))
            ece += gap * mask.mean()         # weight by bin proportion
    return ece


def test_domain_calibration_sampling(bnn, X, domains):
    inputs = tf.placeholder(shape=X.shape, dtype=tf.float32)
    dom_vec = tf.placeholder(shape=domains.shape, dtype=tf.float32)
    mean_tf, var_tf = bnn.create_prediction_tensors(inputs)

    # three variants
    preds_uncal = bnn.sample_predictions(mean_tf, var_tf, calibrate=False)
    preds_cal   = bnn.sample_predictions(mean_tf, var_tf, calibrate=True)
    preds_multi = bnn.sample_predictions(mean_tf, var_tf, calibrate=True, domain_vec=dom_vec)

    preds_uncal, preds_cal, preds_multi, mean, var = bnn.sess.run(
        [preds_uncal, preds_cal, preds_multi, mean_tf, var_tf],
        feed_dict={inputs: X, dom_vec: domains}
    )

    def prepare_cdf(preds):
        return norm.cdf(preds, loc=mean, scale=np.sqrt(var))

    cdf_uncal  = prepare_cdf(preds_uncal)
    cdf_cal    = prepare_cdf(preds_cal)
    cdf_multi  = prepare_cdf(preds_multi)
    cdf_cal    = bnn.sess.run(bnn.recalibrator(cdf_cal))                       # global
    cdf_multi  = bnn.sess.run(bnn.recalibrator(cdf_multi, domain_vec=domains)) # multi

    for d in range(OUT_DIM):
        emp_uncal  = np.array([np.mean(cdf_uncal[:, d]  < p) for p in cdf_uncal[:, d]])
        emp_cal    = np.array([np.mean(cdf_cal[:, d]    < p) for p in cdf_cal[:, d]])
        emp_multi  = np.array([np.mean(cdf_multi[:, d]  < p) for p in cdf_multi[:, d]])

        ece_uncal  = compute_ece(cdf_uncal[:, d],  emp_uncal)
        ece_cal    = compute_ece(cdf_cal[:, d],    emp_cal)
        ece_multi  = compute_ece(cdf_multi[:, d],  emp_multi)

        print(f"[Dim {d}]  ECE  Uncal={ece_uncal:.4f} | Single={ece_cal:.4f} | Multi={ece_multi:.4f}")

        plt.figure(figsize=(5, 5))
        plt.title(f"Empirical vs Predicted CDF (dim {d})")
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
        for label, cdf, emp in [("Uncal", cdf_uncal, emp_uncal),
                                ("Single", cdf_cal,  emp_cal),
                                ("Multi",  cdf_multi, emp_multi)]:
            plt.scatter(cdf[:, d], emp, alpha=0.4, label=label)
        plt.xlabel("Predicted CDF"); plt.ylabel("Empirical CDF")
        plt.legend(); plt.tight_layout()
        plt.savefig(f"calibration_dim{d}.png")
        plt.close()

def reliability_curve(prob, n_bins=ECE_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_conf, bin_acc = [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (prob >= lo) & (prob < hi)
        if mask.any():
            conf = prob[mask].mean()
            acc  = (prob[mask] < prob[mask][:, None]).mean()  # empirical 为自身比较
            bin_conf.append(conf)
            bin_acc.append(acc)
    return np.array(bin_conf), np.array(bin_acc)


def test_reliability_curve(bnn, X, domains, n_bins=ECE_BINS):
    inputs = tf.placeholder(shape=X.shape, dtype=tf.float32)
    dom_vec = tf.placeholder(shape=domains.shape, dtype=tf.float32)
    mean_tf, var_tf = bnn.create_prediction_tensors(inputs)

    preds_uncal = bnn.sample_predictions(mean_tf, var_tf, calibrate=False)
    preds_cal   = bnn.sample_predictions(mean_tf, var_tf, calibrate=True)
    preds_multi = bnn.sample_predictions(mean_tf, var_tf, calibrate=True, domain_vec=dom_vec)

    preds_uncal, preds_cal, preds_multi, mean, var = bnn.sess.run(
        [preds_uncal, preds_cal, preds_multi, mean_tf, var_tf],
        feed_dict={inputs: X, dom_vec: domains}
    )

    def to_cdf(p):  # 预测→CDF 概率
        return norm.cdf(p, loc=mean, scale=np.sqrt(var))

    cdf_uncal  = to_cdf(preds_uncal)
    cdf_cal    = to_cdf(preds_cal)
    cdf_multi  = to_cdf(preds_multi)
    cdf_cal    = bnn.sess.run(bnn.recalibrator(cdf_cal))
    cdf_multi  = bnn.sess.run(bnn.recalibrator(cdf_multi, domain_vec=domains))

    for d in range(OUT_DIM):
        p_uncal  = cdf_uncal[:, d]
        p_cal    = cdf_cal[:, d]
        p_multi  = cdf_multi[:, d]

        conf_u, acc_u = reliability_curve(p_uncal,  n_bins)
        conf_c, acc_c = reliability_curve(p_cal,    n_bins)
        conf_m, acc_m = reliability_curve(p_multi,  n_bins)

        plt.figure(figsize=(5, 5))
        plt.plot([0, 1], [0, 1], "k--", lw=1, label="Ideal")
        plt.plot(conf_u, acc_u, "o-", label="Uncalibrated")
        plt.plot(conf_c, acc_c, "s-", label="Single Calibrated")
        plt.plot(conf_m, acc_m, "d-", label="Multi Calibrated")
        plt.xlabel("Mean predicted probability")
        plt.ylabel("Empirical probability")
        plt.title(f"Reliability Curve (dim {d})")
        plt.grid(alpha=.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"images/bnn_multi_domain_reliability_dim{d}.png")
        plt.close()

def main():
    X, y, domains = stub_data_with_domains()
    bnn = create_bnn_with_domain(X, y, domains)
    test_domain_calibration_sampling(bnn, X, domains)
    test_reliability_curve(bnn, X, domains)



if __name__ == "__main__":
    main()