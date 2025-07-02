from gym.envs.registration import register


register(
    id='MBRLCartpole-v0',
    entry_point='dmbrl.env.cartpole:CartpoleEnv'
)


register(
    id='MBRLReacher3D-v0',
    entry_point='dmbrl.env.reacher:Reacher3DEnv'
)


register(
    id='MBRLPusher-v0',
    entry_point='dmbrl.env.pusher:PusherEnv'
)


register(
    id='MBRLHalfCheetah-v0',
    entry_point='dmbrl.env.half_cheetah:HalfCheetahEnv'
)

register(
    id="MBRLHalfCheetah-v4",
    entry_point="dmbrl.env.half_cheetah:HalfCheetahEnv",
    max_episode_steps=5000,
    kwargs=dict(
        exclude_current_positions_from_observation=True,
    ),
)

register(
    id="MBRLHalfCheetah-v4-Randomized",
    entry_point="dmbrl.env.half_cheetah:HalfCheetahEnv",
    max_episode_steps=5000,
    kwargs=dict(
        exclude_current_positions_from_observation=True,
        enable_domain_randomization=True,
        mass_range=[0.5, 2.0],
        friction_range=[0.3, 1.5],
        damping_range=[0.5, 2.0],
        gravity_range=[0.8, 1.2],
        timestep_range=[0.95, 1.05],
    ),
)


register(
    id='MBRLAnt-v0',
    entry_point='dmbrl.env.ant:AntEnv'
)
