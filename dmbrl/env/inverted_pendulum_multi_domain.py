__credits__ = ["Rushiv Arora"]

import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
from dmbrl.env.inverted_pendulum_v4 import InvertedPendulumEnv

class MultiDomainInvertedPendulumEnv(InvertedPendulumEnv):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25,
    }
    def __init__(self, domain_id=0, domain_configs=None, **kwargs):
            self.domain_id = domain_id
            self.domain_configs = domain_configs or {}
            super().__init__(**kwargs)
    
    def _apply_domain_randomization(self):
        if self.domain_id in self.domain_configs:
            config = self.domain_configs[self.domain_id]
            if "reset_noise_scale" in config:
                self.reset_noise_scale = config["reset_noise_scale"]

    def reset_model(self):
        self._apply_domain_randomization()
        return super().reset_model()
