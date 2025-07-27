from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
from time import time, localtime, strftime

import numpy as np
from scipy.io import savemat
from dotmap import DotMap

from dmbrl.misc.DotmapUtils import get_required_argument
from dmbrl.misc.Agent import Agent

SAVE_EVERY = 25


class MBExperiment:
    def __init__(self, params, multi_domain=False):
        """Initializes class instance.

        Argument:
            params (DotMap): A DotMap containing the following:
                .sim_cfg:
                    .env (gym.env): Environment for this experiment
                    .task_hor (int): Task horizon
                    .stochastic (bool): (optional) If True, agent adds noise to its actions.
                        Must provide noise_std (see below). Defaults to False.
                    .noise_std (float): for stochastic agents, noise of the form N(0, noise_std^2I)
                        will be added.

                .exp_cfg:
                    .ntrain_iters (int): Number of training iterations to be performed.
                    .nrollouts_per_iter (int): (optional) Number of rollouts done between training
                        iterations. Defaults to 1.
                    .ninit_rollouts (int): (optional) Number of initial rollouts. Defaults to 1.
                    .policy (controller): Policy that will be trained.

                .log_cfg:
                    .logdir (str): Parent of directory path where experiment data will be saved.
                        Experiment will be saved in logdir/<date+time of experiment start>
                    .nrecord (int): (optional) Number of rollouts to record for every iteration.
                        Defaults to 0.
                    .neval (int): (optional) Number of rollouts for performance evaluation.
                        Defaults to 1.
        """
        self.env = get_required_argument(params.sim_cfg, "env", "Must provide environment.")
        self.task_hor = get_required_argument(params.sim_cfg, "task_hor", "Must provide task horizon.")
        if params.sim_cfg.get("stochastic", False):
            self.agent = Agent(DotMap(
                env=self.env, noisy_actions=True,
                noise_scale=params.sim_cfg.get("noise_scale", None),
                noise_stddev=get_required_argument(
                    params.sim_cfg,
                    "noise_std",
                    "Must provide noise standard deviation in the case of a stochastic environment."
                )
            ))
        else:
            self.agent = Agent(DotMap(env=self.env, noisy_actions=False, noise_scale=params.sim_cfg.get("noise_scale", None)))

        self.ntrain_iters = get_required_argument(
            params.exp_cfg, "ntrain_iters", "Must provide number of training iterations."
        )
        self.nrollouts_per_iter = params.exp_cfg.get("nrollouts_per_iter", 1)
        self.ninit_rollouts = params.exp_cfg.get("ninit_rollouts", 1)
        self.policy = get_required_argument(params.exp_cfg, "policy", "Must provide a policy.")

        self.logdir = os.path.join(
            get_required_argument(params.log_cfg, "logdir", "Must provide log parent directory."),
            strftime("%Y-%m-%d--%H:%M:%S", localtime())
        )
        self.nrecord = params.log_cfg.get("nrecord", 0)
        self.neval = params.log_cfg.get("neval", 1)
        self.multi_domain = multi_domain

    def run_experiment(self):
        """Perform experiment.
        """
        os.makedirs(self.logdir, exist_ok=True)

        traj_obs, traj_acs, traj_rets, traj_rews = [], [], [], []

        # Perform initial rollouts
        samples = []
        for i in range(self.ninit_rollouts):
            samples.append(
                self.agent.sample(
                    self.task_hor, self.policy
                )
            )
            traj_obs.append(samples[-1]["obs"])
            traj_acs.append(samples[-1]["ac"])
            traj_rews.append(samples[-1]["rewards"])

        if self.ninit_rollouts > 0:
            self.policy.train(
                [sample["obs"] for sample in samples],
                [sample["ac"] for sample in samples],
                [sample["rewards"] for sample in samples]
            )

        # Training loop
        for i in range(self.ntrain_iters):
            print("####################################################################")
            print("Starting training iteration %d." % (i + 1))

            iter_dir = os.path.join(self.logdir, "train_iter%d" % (i + 1))
            os.makedirs(iter_dir, exist_ok=True)

            samples = []
            for j in range(self.nrecord):
                samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy,
                        os.path.join(iter_dir, "rollout%d.mp4" % j)
                    )
                )
            if self.nrecord > 0:
                for item in filter(lambda f: f.endswith(".json"), os.listdir(iter_dir)):
                    os.remove(os.path.join(iter_dir, item))
            for j in range(max(self.neval, self.nrollouts_per_iter) - self.nrecord):
                samples.append(
                    self.agent.sample(
                        self.task_hor, self.policy
                    )
                )
            print("Rewards obtained:", [sample["reward_sum"] for sample in samples[:self.neval]])
            traj_obs.extend([sample["obs"] for sample in samples[:self.nrollouts_per_iter]])
            traj_acs.extend([sample["ac"] for sample in samples[:self.nrollouts_per_iter]])
            traj_rets.extend([sample["reward_sum"] for sample in samples[:self.neval]])
            traj_rews.extend([sample["rewards"] for sample in samples[:self.nrollouts_per_iter]])
            samples = samples[:self.nrollouts_per_iter]

            self.policy.dump_logs(self.logdir, iter_dir if (i + 1) % SAVE_EVERY == 0 else None)
            savemat(
                os.path.join(self.logdir, "logs.mat"),
                {
                    "observations": traj_obs,
                    "actions": traj_acs,
                    "returns": traj_rets,
                    "rewards": traj_rews
                }
            )

            if i < self.ntrain_iters - 1:
                self.policy.train(
                    [sample["obs"] for sample in samples],
                    [sample["ac"] for sample in samples],
                    [sample["rewards"] for sample in samples],
                    logdir=iter_dir if (i + 1) % SAVE_EVERY == 0 else None
                )

            # Delete iteration directory if not used
            if len(os.listdir(iter_dir)) == 0:
                os.rmdir(iter_dir)

    def run_experiment_multi_domain(self, domains, env):
        """Perform experiment with multiple domains."""
        os.makedirs(self.logdir, exist_ok=True)

        domain_obs_trajs = [[] for _ in domains]
        domain_acs_trajs = [[] for _ in domains]
        domain_rews_trajs = [[] for _ in domains]
        domain_rets = [[] for _ in domains]

        for i in range(self.ntrain_iters):
            print("####################################################################")
            print("Starting multi-domain training iteration %d." % (i + 1))

            iter_dir = os.path.join(self.logdir, f"train_iter{i+1}")
            os.makedirs(iter_dir, exist_ok=True)

            for d_idx, domain_id in enumerate(domains):
                print(f"  Sampling from domain {domain_id}...")
                # Create environment with specific domain configuration
                actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
                domain_env = actual_env.__class__(
                    reset_noise_scale=domains[domain_id]["reset_noise_scale"]
                )
                agent = Agent(DotMap(env=domain_env, noisy_actions=False))

                samples = []
                for _ in range(self.nrollouts_per_iter):
                    samples.append(agent.sample(self.task_hor, self.policy))

                domain_obs_trajs[d_idx].extend([s["obs"] for s in samples])
                domain_acs_trajs[d_idx].extend([s["ac"] for s in samples])
                domain_rews_trajs[d_idx].extend([s["rewards"] for s in samples])
                domain_rets[d_idx].extend([s["reward_sum"] for s in samples])

            print("Domain rewards summary:")
            for d_idx, domain_id in enumerate(domains):
                rets = domain_rets[d_idx]
                print(rets)
                print(f"  Domain {domain_id}: avg return = {np.max(rets):.2f}, min = {np.min(rets):.2f}")

            # Train controller on all domain data
            self.policy.train_with_domains(domain_obs_trajs, domain_acs_trajs, domain_rews_trajs, logdir=iter_dir)

            self.policy.dump_logs(self.logdir, iter_dir if (i + 1) % SAVE_EVERY == 0 else None)
            savemat(
                os.path.join(self.logdir, f"logs_iter{i+1}.mat"),
                {
                    "observations": domain_obs_trajs,
                    "actions": domain_acs_trajs,
                    "rewards": domain_rews_trajs,
                    "returns": domain_rets
                }
            )