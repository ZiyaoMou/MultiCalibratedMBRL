from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from dotmap import DotMap

import time


class Agent:
    """An general class for RL agents.
    """
    def __init__(self, params):
        """Initializes an agent.

        Arguments:
            params: (DotMap) A DotMap of agent parameters.
                .env: (OpenAI gym environment) The environment for this agent.
                .noisy_actions: (bool) Indicates whether random Gaussian noise will 
                    be added to the actions of this agent.
                .noise_stddev: (float) The standard deviation to be used for the 
                    action noise if params.noisy_actions is True.
        """
        self.env = params.env
        self.noise_stddev = params.noise_stddev if params.get("noisy_actions", False) else None

        if isinstance(self.env, DotMap):
            raise ValueError("Environment must be provided to the agent at initialization.")
        if (not isinstance(self.noise_stddev, float)) and params.get("noisy_actions", False):
            raise ValueError("Must provide standard deviation for noise for noisy actions.")

        if self.noise_stddev is not None:
            self.dU = self.env.action_space.shape[0]

    def sample(self, horizon, policy, record_fname=None):
        """Samples a rollout from the agent.

        Arguments: 
            horizon: (int) The length of the rollout to generate from the agent.
            policy: (policy) The policy that the agent will use for actions.
            record_fname: (str/None) The name of the file to which a recording of the rollout
                will be saved. If None, the rollout will not be recorded.

        Returns: (dict) A dictionary containing data from the rollout.
            The keys of the dictionary are 'obs', 'ac', and 'reward_sum'.
        """
        video_record = record_fname is not None
        recorder = None if not video_record else VideoRecorder(self.env, record_fname)

        times, rewards = [], []
        # Get domain vector from environment if available
        domain_vec = None
        if hasattr(self.env, 'get_domain_vector'):
            domain_vec = self.env.get_domain_vector()
        elif hasattr(self.env, 'domain_vector'):
            domain_vec = self.env.domain_vector
        # If no domain vector is available, don't add any
        
        # Initialize with reset observation
        reset_result = self.env.reset()
        if isinstance(reset_result, tuple):
            obs = reset_result[0]  # Gym 0.26+: (obs, info)
        else:
            obs = reset_result     # Gym < 0.26

        # Convert observation to float32
        obs = obs.astype(np.float32)

        # Concatenate domain vector with observation if available
        if domain_vec is not None and len(domain_vec) > 0:
            obs = np.concatenate([obs, domain_vec], axis=-1)

        O, A, reward_sum, done = [obs.copy()], [], 0, False

        policy.reset()
        for t in range(horizon):
            if video_record:
                recorder.capture_frame()
            start = time.time()
            A.append(policy.act(O[t], t))
            times.append(time.time() - start)

            if self.noise_stddev is None:
                next_obs, reward, terminated, truncated, info = self.env.step(A[t])
            else:
                action = A[t] + np.random.normal(loc=0, scale=self.noise_stddev, size=[self.dU])
                action = np.minimum(np.maximum(action, self.env.action_space.low), self.env.action_space.high)
                next_obs, reward, terminated, truncated, info = self.env.step(action)
            
            # Combine terminated and truncated into done for backward compatibility
            done = terminated or truncated
            
            # Convert next observation to float32
            next_obs = next_obs.astype(np.float32)
            
            # Concatenate domain vector with next observation if available
            if domain_vec is not None and len(domain_vec) > 0:
                next_obs = np.concatenate([next_obs, domain_vec], axis=-1)
            
            O.append(next_obs.copy())  # Save the next observation
            reward_sum += reward
            rewards.append(reward)
            if done:
                break

        if video_record:
            recorder.capture_frame()
            recorder.close()

        print("Average action selection time: ", np.mean(times))
        print("Rollout length: ", len(A))

        return {
            "obs": np.array(O, dtype=np.float32),
            "ac": np.array(A, dtype=np.float32),
            "reward_sum": reward_sum,
            "rewards": np.array(rewards, dtype=np.float32),
            "domain_vec": domain_vec,
        }
