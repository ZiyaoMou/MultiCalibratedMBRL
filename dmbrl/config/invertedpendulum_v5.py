import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from dotmap import DotMap
import gymnasium as gym

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC


class InvertedPendulumConfigModule:
    ENV_NAME = "MBRLInvertedPendulum-v5"
    TASK_HORIZON = 1000
    NTRAIN_ITERS = 300
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25

    # observation dim = 4, action dim = 1
    MODEL_IN, MODEL_OUT = 4 + 1, 4  # (obs + action, delta_obs)
    GP_NINDUCING_POINTS = 200


    def __init__(self, noise_scale=0.01):
        self.ENV = gym.make(self.ENV_NAME, reset_noise_scale=noise_scale)
        self.ENV.reset(seed=None)
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)

        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2000
            },
            "CEM": {
                "popsize": 400,
                "num_elites": 40,
                "max_iters": 5,
                "alpha": 0.1
            }
        }

    @staticmethod
    def obs_preproc(obs):
        return obs

    @staticmethod
    def obs_postproc(obs, pred):
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        return next_obs - obs

    @staticmethod
    def obs_cost_fn(obs):
        if isinstance(obs, np.ndarray):
            return 1e-2 * np.square(obs[:, 0]) + np.square(obs[:, 1])
        else:
            return 1e-2 * tf.square(obs[:, 0]) + tf.square(obs[:, 1])

    @staticmethod
    def ac_cost_fn(acs):
        return 0.001 * np.sum(np.square(acs), axis=1) if isinstance(acs, np.ndarray) else 0.001 * tf.reduce_sum(tf.square(acs), axis=1)

    def nn_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model",
            num_networks=get_required_argument(model_init_cfg, "num_nets", "Must provide ensemble size"),
            sess=self.SESS,
            load_model=model_init_cfg.get("load_model", False),
            model_dir=model_init_cfg.get("model_dir", None)
        ))

        if not model_init_cfg.get("load_model", False):
            model.add(FC(200, input_dim=self.MODEL_IN, activation="swish", weight_decay=0.000025))
            model.add(FC(200, activation="swish", weight_decay=0.00005))
            model.add(FC(200, activation="swish", weight_decay=0.000075))
            model.add(FC(200, activation="swish", weight_decay=0.000075))
            model.add(FC(self.MODEL_OUT, weight_decay=0.0001))

        model.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})
        return model

    def gp_constructor(self, model_init_cfg):
        model = get_required_argument(model_init_cfg, "model_class", "Must provide model class")(DotMap(
            name="model",
            kernel_class=get_required_argument(model_init_cfg, "kernel_class", "Must provide kernel class"),
            kernel_args=model_init_cfg.get("kernel_args", {}),
            num_inducing_points=get_required_argument(
                model_init_cfg, "num_inducing_points", "Must provide number of inducing points."
            ),
            sess=self.SESS
        ))
        return model


CONFIG_MODULE = InvertedPendulumConfigModule