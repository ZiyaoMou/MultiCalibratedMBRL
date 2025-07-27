from gymnasium.envs.registration import register


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
    id='MBRLAnt-v0',
    entry_point='dmbrl.env.ant:AntEnv'
)

register(
    id="MBRLHalfCheetah-v5",
    entry_point='dmbrl.env.half_cheetah_v5:HalfCheetahEnv'
)

# register(
#     id="MBRLHalfCheetahMultiDomain-v5",
#     entry_point='dmbrl.env.halt_cheetah_multi_domain:MultiDomainHalfCheetahEnv'
# )

register(
    id="MBRLInvertedPendulum-v5",
    entry_point='dmbrl.env.inverted_pendulum_v5:InvertedPendulumEnv'
)

# register(
#     id="MBRLInvertedPendulumMultiDomain-v5",
#     entry_point='dmbrl.env.inverted_pendulum_multi_domain:MultiDomainInvertedPendulumEnv'
# )