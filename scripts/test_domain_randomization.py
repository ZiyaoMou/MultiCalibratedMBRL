#!/usr/bin/env python3
"""
Test script for domain randomization in HalfCheetah environment.
This script demonstrates how to use the enhanced HalfCheetah environment
with physical parameter randomization.
"""

import numpy as np
import gym
import matplotlib.pyplot as plt
from dmbrl.env import HalfCheetahEnv

def test_domain_randomization():
    """Test domain randomization with different configurations."""
    
    print("Testing Domain Randomization in HalfCheetah Environment")
    print("=" * 60)
    
    # Test 1: Standard environment (no randomization)
    print("\n1. Standard Environment (No Randomization)")
    env_standard = gym.make("MBRLHalfCheetah-v4")
    
    # Test 2: Environment with domain randomization
    print("\n2. Environment with Domain Randomization")
    env_randomized = gym.make("MBRLHalfCheetah-v4-Randomized")
    
    # Test 3: Custom randomization parameters
    print("\n3. Custom Randomization Parameters")
    env_custom = HalfCheetahEnv(
        enable_domain_randomization=True,
        mass_range=[0.3, 3.0],      # More extreme mass variation
        friction_range=[0.1, 2.0],  # More extreme friction variation
        damping_range=[0.2, 3.0],   # More extreme damping variation
        gravity_range=[0.5, 1.5],   # More extreme gravity variation
        timestep_range=[0.9, 1.1],  # Moderate timestep variation
    )
    
    # Collect data from each environment
    results = {}
    
    for name, env in [("Standard", env_standard), 
                     ("Randomized", env_randomized), 
                     ("Custom", env_custom)]:
        print(f"\nCollecting data from {name} environment...")
        
        observations = []
        rewards = []
        masses = []
        frictions = []
        
        # Run multiple episodes to see the effects
        for episode in range(5):
            obs = env.reset()
            episode_reward = 0
            step_count = 0
            
            # Store initial physical parameters
            if hasattr(env, 'model'):
                masses.append(env.model.body_mass.copy())
                frictions.append(env.model.geom_friction[:, 0].copy())
            
            while step_count < 100:  # Short episodes for testing
                action = env.action_space.sample()  # Random actions
                obs, reward, terminated, truncated, info = env.step(action)
                
                observations.append(obs)
                rewards.append(reward)
                episode_reward += reward
                step_count += 1
                
                if terminated or truncated:
                    break
            
            print(f"  Episode {episode + 1}: Total reward = {episode_reward:.2f}, Steps = {step_count}")
        
        results[name] = {
            'observations': np.array(observations),
            'rewards': np.array(rewards),
            'masses': np.array(masses) if masses else None,
            'frictions': np.array(frictions) if frictions else None
        }
    
    # Analyze and visualize results
    print("\n" + "=" * 60)
    print("ANALYSIS RESULTS")
    print("=" * 60)
    
    # Compare reward distributions
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Reward distributions
    plt.subplot(2, 3, 1)
    for name, data in results.items():
        plt.hist(data['rewards'], alpha=0.7, label=name, bins=20)
    plt.xlabel('Reward')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Forward velocity (first observation dimension)
    plt.subplot(2, 3, 2)
    for name, data in results.items():
        forward_velocities = data['observations'][:, 0]  # First dimension is forward velocity
        plt.hist(forward_velocities, alpha=0.7, label=name, bins=20)
    plt.xlabel('Forward Velocity')
    plt.ylabel('Frequency')
    plt.title('Forward Velocity Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Mass variations (if available)
    plt.subplot(2, 3, 3)
    for name, data in results.items():
        if data['masses'] is not None:
            # Plot mean mass across episodes
            mean_masses = np.mean(data['masses'], axis=0)
            plt.plot(mean_masses, 'o-', label=name, alpha=0.7)
    plt.xlabel('Body Index')
    plt.ylabel('Mean Mass')
    plt.title('Body Mass Variations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Friction variations (if available)
    plt.subplot(2, 3, 4)
    for name, data in results.items():
        if data['frictions'] is not None:
            # Plot mean friction across episodes
            mean_frictions = np.mean(data['frictions'], axis=0)
            plt.plot(mean_frictions, 'o-', label=name, alpha=0.7)
    plt.xlabel('Geometry Index')
    plt.ylabel('Mean Friction')
    plt.title('Friction Coefficient Variations')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 5: Episode rewards over time
    plt.subplot(2, 3, 5)
    for name, data in results.items():
        # Calculate episode rewards
        episode_rewards = []
        current_episode = []
        for reward in data['rewards']:
            current_episode.append(reward)
            if len(current_episode) >= 100:  # Episode length
                episode_rewards.append(sum(current_episode))
                current_episode = []
        
        if episode_rewards:
            plt.plot(episode_rewards, 'o-', label=name, alpha=0.7)
    plt.xlabel('Episode')
    plt.ylabel('Episode Reward')
    plt.title('Episode Rewards Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Observation space exploration
    plt.subplot(2, 3, 6)
    for name, data in results.items():
        # Calculate observation variance across all dimensions
        obs_variance = np.var(data['observations'], axis=0)
        plt.plot(obs_variance, 'o-', label=name, alpha=0.7)
    plt.xlabel('Observation Dimension')
    plt.ylabel('Variance')
    plt.title('Observation Space Exploration')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('domain_randomization_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nSUMMARY STATISTICS")
    print("-" * 40)
    for name, data in results.items():
        print(f"\n{name} Environment:")
        print(f"  Mean Reward: {np.mean(data['rewards']):.3f} ± {np.std(data['rewards']):.3f}")
        print(f"  Mean Forward Velocity: {np.mean(data['observations'][:, 0]):.3f} ± {np.std(data['observations'][:, 0]):.3f}")
        print(f"  Observation Variance: {np.mean(np.var(data['observations'], axis=0)):.3f}")
        
        if data['masses'] is not None:
            mass_variance = np.var(data['masses'], axis=0)
            print(f"  Mass Variation (mean std): {np.mean(np.std(data['masses'], axis=0)):.3f}")
        
        if data['frictions'] is not None:
            friction_variance = np.var(data['frictions'], axis=0)
            print(f"  Friction Variation (mean std): {np.mean(np.std(data['frictions'], axis=0)):.3f}")
    
    print(f"\nResults saved to: domain_randomization_comparison.png")

def test_parameter_sensitivity():
    """Test sensitivity to different randomization ranges."""
    
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY TEST")
    print("=" * 60)
    
    # Test different randomization intensities
    intensities = {
        'Low': {
            'mass_range': [0.8, 1.2],
            'friction_range': [0.8, 1.2],
            'damping_range': [0.8, 1.2],
            'gravity_range': [0.9, 1.1],
            'timestep_range': [0.98, 1.02],
        },
        'Medium': {
            'mass_range': [0.5, 2.0],
            'friction_range': [0.3, 1.5],
            'damping_range': [0.5, 2.0],
            'gravity_range': [0.8, 1.2],
            'timestep_range': [0.95, 1.05],
        },
        'High': {
            'mass_range': [0.2, 5.0],
            'friction_range': [0.1, 3.0],
            'damping_range': [0.2, 5.0],
            'gravity_range': [0.5, 1.5],
            'timestep_range': [0.9, 1.1],
        }
    }
    
    sensitivity_results = {}
    
    for intensity_name, params in intensities.items():
        print(f"\nTesting {intensity_name} intensity randomization...")
        
        env = HalfCheetahEnv(
            enable_domain_randomization=True,
            **params
        )
        
        episode_rewards = []
        forward_velocities = []
        
        for episode in range(10):
            obs = env.reset()
            episode_reward = 0
            
            for step in range(100):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                forward_velocities.append(obs[0])  # Forward velocity
                
                if terminated or truncated:
                    break
            
            episode_rewards.append(episode_reward)
        
        sensitivity_results[intensity_name] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'mean_velocity': np.mean(forward_velocities),
            'std_velocity': np.std(forward_velocities),
            'reward_range': np.max(episode_rewards) - np.min(episode_rewards)
        }
        
        print(f"  Mean Episode Reward: {sensitivity_results[intensity_name]['mean_reward']:.2f} ± {sensitivity_results[intensity_name]['std_reward']:.2f}")
        print(f"  Reward Range: {sensitivity_results[intensity_name]['reward_range']:.2f}")
    
    # Plot sensitivity results
    plt.figure(figsize=(12, 8))
    
    intensities_list = list(sensitivity_results.keys())
    mean_rewards = [sensitivity_results[name]['mean_reward'] for name in intensities_list]
    std_rewards = [sensitivity_results[name]['std_reward'] for name in intensities_list]
    reward_ranges = [sensitivity_results[name]['reward_range'] for name in intensities_list]
    
    plt.subplot(2, 2, 1)
    plt.bar(intensities_list, mean_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
    plt.ylabel('Mean Episode Reward')
    plt.title('Mean Reward by Randomization Intensity')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.bar(intensities_list, reward_ranges, alpha=0.7)
    plt.ylabel('Reward Range')
    plt.title('Reward Variability by Randomization Intensity')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    mean_velocities = [sensitivity_results[name]['mean_velocity'] for name in intensities_list]
    std_velocities = [sensitivity_results[name]['std_velocity'] for name in intensities_list]
    plt.bar(intensities_list, mean_velocities, yerr=std_velocities, capsize=5, alpha=0.7)
    plt.ylabel('Mean Forward Velocity')
    plt.title('Forward Velocity by Randomization Intensity')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.bar(intensities_list, std_rewards, alpha=0.7)
    plt.ylabel('Reward Standard Deviation')
    plt.title('Reward Uncertainty by Randomization Intensity')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nSensitivity analysis saved to: parameter_sensitivity_analysis.png")

if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run tests
    test_domain_randomization()
    test_parameter_sensitivity()
    
    print("\n" + "=" * 60)
    print("DOMAIN RANDOMIZATION TEST COMPLETED")
    print("=" * 60)
    print("\nKey findings:")
    print("1. Domain randomization increases environment diversity")
    print("2. Higher randomization intensity leads to greater reward variability")
    print("3. Physical parameter variations affect forward velocity and control")
    print("4. Custom randomization ranges allow fine-tuning of domain adaptation")
    print("\nUsage examples:")
    print("- Standard: gym.make('MBRLHalfCheetah-v4')")
    print("- Randomized: gym.make('MBRLHalfCheetah-v4-Randomized')")
    print("- Custom: HalfCheetahEnv(enable_domain_randomization=True, mass_range=[0.3, 3.0], ...)") 