import os
import numpy as np
from gym import utils, spaces
from gym.envs.mujoco import mujoco_env


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 20
    }
    def __init__(self, **kwargs):
        self.prev_qpos = None
        dir_path = os.path.dirname(os.path.realpath(__file__))
        xml_path = os.path.join(dir_path, "assets/half_cheetah.xml")
        frame_skip = 5

        self.forward_reward_weight = kwargs.pop('forward_reward_weight', 1.0)
        self.ctrl_cost_weight = kwargs.pop('ctrl_cost_weight', 0.1)
        self.reset_noise_scale = kwargs.pop('reset_noise_scale', 0.1)
        self.exclude_current_positions_from_observation = kwargs.pop(
            'exclude_current_positions_from_observation', True
        )
        # Domain randomization parameters
        self.enable_domain_randomization = kwargs.pop('enable_domain_randomization', False)
        self.mass_range = kwargs.pop('mass_range', [0.5, 2.0])
        self.friction_range = kwargs.pop('friction_range', [0.3, 1.5])
        self.damping_range = kwargs.pop('damping_range', [0.5, 2.0])
        self.gravity_range = kwargs.pop('gravity_range', [0.8, 1.2])
        self.timestep_range = kwargs.pop('timestep_range', [0.95, 1.05])

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        mujoco_env.MujocoEnv.__init__(self, xml_path, frame_skip, observation_space=self.observation_space)
        utils.EzPickle.__init__(self, **kwargs)

        # Save original physical parameters for domain randomization
        if self.enable_domain_randomization:
            self.init_body_mass = self.model.body_mass.copy()
            self.init_geom_friction = self.model.geom_friction.copy()
            self.init_dof_damping = self.model.dof_damping.copy()
            self.init_gravity = self.model.opt.gravity.copy()
            self.init_timestep = self.model.opt.timestep

    def _get_obs(self):
        return np.concatenate([
            (self.data.qpos.flat[:1] - self.prev_qpos[:1]) / self.dt,
            self.data.qpos.flat[1:],
            self.data.qvel.flat,
        ]).astype(np.float32)

    def reset_model(self):
        # Domain randomization
        if self.enable_domain_randomization:
            # Randomize body masses
            mass_scale = np.random.uniform(*self.mass_range, size=self.model.body_mass.shape)
            self.model.body_mass[:] = self.init_body_mass * mass_scale

            # Randomize friction coefficients
            friction_scale = np.random.uniform(*self.friction_range, size=(self.model.ngeom,))
            self.model.geom_friction[:, 0] = self.init_geom_friction[:, 0] * friction_scale

            # Randomize joint damping
            damping_scale = np.random.uniform(*self.damping_range, size=self.model.dof_damping.shape)
            self.model.dof_damping[:] = self.init_dof_damping * damping_scale

            # Randomize gravity
            gravity_scale = np.random.uniform(*self.gravity_range)
            self.model.opt.gravity[:] = self.init_gravity * gravity_scale

            # Randomize timestep
            timestep_scale = np.random.uniform(*self.timestep_range)
            self.model.opt.timestep = self.init_timestep * timestep_scale

        qpos = self.init_qpos + np.random.normal(loc=0, scale=self.reset_noise_scale, size=self.model.nq)
        qvel = self.init_qvel + np.random.normal(loc=0, scale=self.reset_noise_scale, size=self.model.nv)
        self.set_state(qpos, qvel)
        self.prev_qpos = np.copy(self.data.qpos.flat)
        return self._get_obs()

    def step(self, action):
        self.prev_qpos = np.copy(self.data.qpos.flat)
        self.do_simulation(action, self.frame_skip)
        ob = self._get_obs()

        reward_ctrl = -self.ctrl_cost_weight * np.square(action).sum()
        reward_run = self.forward_reward_weight * ob[0]  # forward velocity
        reward = reward_run + reward_ctrl

        terminated = False
        truncated = False
        return ob.astype(np.float32), reward, terminated, truncated, {}

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.25
        self.viewer.cam.elevation = -55