import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tensorflow_probability as tfp
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

from dotmap import DotMap

from dmbrl.modeling.models import BNN
from dmbrl.modeling.layers import FC, RecalibrationLayer

NUM_SAMPLES = 1024
IN_DIM = 100
HIDDEN_DIM = 10
OUT_DIM = 2


def sigmoid(x):
    return 1/(1 + np.exp(-x))


def stub_data():
    X = np.random.random(size=(NUM_SAMPLES, IN_DIM))
    # W_tru = np.random.random(size=(IN_DIM, OUT_DIM))
    # b_tru = 5
    # y = np.matmul(X, W_tru) + b_tru

    W_hidden = np.random.random(size=(IN_DIM, HIDDEN_DIM))
    W_last = np.random.random(size=(HIDDEN_DIM, OUT_DIM))

    y_mid = np.matmul(X, W_hidden) + 5
    y_mid[y_mid < 0] = 0
    # y_mid = sigmoid(y_mid)
    y = np.matmul(y_mid, W_last) + 2

    return (X, y)


def create_bnn(X, y):
    params = DotMap({"name": "test"})
    model = BNN(params)
    model.add(FC(OUT_DIM, input_dim=IN_DIM, weight_decay=0.0005)) # linear model for simplicity
    model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.03})

    model.train(X, y, epochs=1000, batch_size=64)
    model.calibrate(X, y)
    # model.plot_calibration(X, y)

    return model

def test_calibration_sampling(bnn, X, n_bins=15, recalib=True):
    """Draw reliability (calibration) curves for each output dim."""
    inputs = tf.placeholder(shape=X.shape, dtype=tf.float32)
    mean_tf, var_tf = bnn.create_prediction_tensors(inputs)
    preds_tf = bnn.sample_predictions(mean_tf, var_tf,
                                      calibrate=recalib)

    preds, mean, var = bnn.sess.run([preds_tf, mean_tf, var_tf],
                                    feed_dict={inputs: X})

    cdf_pred = norm.cdf(preds, loc=mean, scale=np.sqrt(var))
    if recalib:
        cdf_pred = bnn.sess.run(bnn.recalibrator(cdf_pred))

    for d in range(cdf_pred.shape[1]):
        prob_pred = cdf_pred[:, d]
        bins = np.linspace(0, 1, n_bins + 1)
        binids = np.digitize(prob_pred, bins) - 1 

        bin_acc, bin_conf, bin_cnt = [], [], []
        for b in range(n_bins):
            idx = binids == b
            if np.any(idx):
                conf = prob_pred[idx].mean()        
                acc  = np.mean(prob_pred[idx] < prob_pred[idx][:, None],
                               axis=(0, 1))
                bin_conf.append(conf)
                bin_acc.append(acc)
                bin_cnt.append(idx.sum())

        plt.figure(figsize=(4.5, 4.5))
        plt.plot([0, 1], [0, 1], 'k--', lw=1, label='ideal')
        plt.plot(bin_conf, bin_acc, 'o-', label='reliability')
        plt.xlabel('Predicted probability')
        plt.ylabel('Empirical probability')
        plt.title(f'Reliability Curve • dim {d}')
        plt.grid(alpha=.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig(f'images/bnn_reliability_curve_dim_{d}.png')


def main():
    X, y = stub_data()
    bnn = create_bnn(X, y)
    test_calibration_sampling(bnn, X)


if __name__ == '__main__':
    main()