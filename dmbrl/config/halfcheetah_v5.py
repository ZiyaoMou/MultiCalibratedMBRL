import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from dotmap import DotMap
import gym

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.modeling.layers import FC


class HalfCheetahConfigModule:
    ENV_NAME = "MBRLHalfCheetah-v5"
    TASK_HORIZON = 1000
    NTRAIN_ITERS = 300
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 30

    # New dimensions for v4: observation dim = 17, action dim = 6
    MODEL_IN, MODEL_OUT = 17 + 6, 17  # (obs + action, delta_obs)
    GP_NINDUCING_POINTS = 300

    def __init__(self, noise_scale=0.1):
        self.ENV = gym.make(self.ENV_NAME, reset_noise_scale=noise_scale)
        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)

        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {
                "popsize": 2500
            },
            "CEM": {
                "popsize": 500,
                "num_elites": 50,
                "max_iters": 5,
                "alpha": 0.1
            }
        }
        

    @staticmethod
    def obs_preproc(obs):
        # Keep raw observation (shape: [B, 17])
        return obs

    @staticmethod
    def obs_postproc(obs, pred):
        # pred: delta_obs, add to original obs
        return obs + pred

    @staticmethod
    def targ_proc(obs, next_obs):
        # Predict delta obs
        return next_obs - obs

    @staticmethod
    def obs_cost_fn(obs):
        # Reward is +velocity, so cost = -velocity
        return -obs[:, 8]  # x_velocity

    @staticmethod
    def ac_cost_fn(acs):
        return 0.1 * np.sum(np.square(acs), axis=1) if isinstance(acs, np.ndarray) else 0.1 * tf.reduce_sum(tf.square(acs), axis=1)

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


CONFIG_MODULE = HalfCheetahConfigModule