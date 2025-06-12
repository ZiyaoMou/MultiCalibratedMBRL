from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

env = make_vec_env("HalfCheetah-v2", n_envs=1)

model = SAC("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=180_000)

model.save("sac_halfcheetah")