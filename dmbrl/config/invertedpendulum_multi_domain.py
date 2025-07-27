import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gymnasium as gym

from dmbrl.config.invertedpendulum_v5 import InvertedPendulumConfigModule


class InvertedPendulumMultiDomainConfigModule(InvertedPendulumConfigModule):
    ENV_NAME = "MBRLInvertedPendulum-v5"
    TASK_HORIZON = 1000
    NTRAIN_ITERS = 300
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 4 + 1, 4
    GP_NINDUCING_POINTS = 200

    def __init__(self, domain_id=0, noise_scale=0.01):
        self.domain_configs = {
            0: {"reset_noise_scale": 0.01},
            1: {"reset_noise_scale": 0.02},
            2: {"reset_noise_scale": 0.03},
            3: {"reset_noise_scale": 0.04},
            4: {"reset_noise_scale": 0.05},
        }

        domain_cfg = self.domain_configs[domain_id]

        print(domain_cfg)

        self.ENV = gym.make(
            self.ENV_NAME,
            # domain_id=domain_id,
            # domain_configs=self.domain_configs,
            reset_noise_scale=domain_cfg["reset_noise_scale"]
        )

        cfg = tf.ConfigProto()
        cfg.gpu_options.allow_growth = True
        self.SESS = tf.Session(config=cfg)

        self.NN_TRAIN_CFG = {"epochs": 5}
        self.OPT_CFG = {
            "Random": {"popsize": 2000},
            "CEM": {"popsize": 400, "num_elites": 40, "max_iters": 5, "alpha": 0.1}
        }

    def get_all_domain_configs(self):
        return self.domain_configs


CONFIG_MODULE = InvertedPendulumMultiDomainConfigModule