import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from isaac_env import IsaacSimEnv
import time
import logging
import matplotlib.pyplot as plt
import os

def main():
    logging.getLogger('pegasus.simulator.logic.backends.px4_mavlink_backend').setLevel(logging.ERROR)
    env = make_vec_env(lambda: IsaacSimEnv(), n_envs=1)
    model = PPO('MlpPolicy', env, verbose=1)
    
    # Train the model and log metrics
    total_timesteps = 1000  # Start with a smaller number of timesteps
    log_interval = 10  # Log metrics every 100 timesteps
    rewards = []
    episode_lengths = []

    for i in range(0, total_timesteps, log_interval):
        model.learn(total_timesteps=log_interval, reset_num_timesteps=False)
        obs = env.reset()
        episode_reward = 0
        episode_length = 0
        max_steps_per_episode = 1000  # Add a maximum number of steps per episode to prevent infinite loops

        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            if done or episode_length >= max_steps_per_episode:
                break

        rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        print(f"Episode {i // log_interval + 1}: Reward = {episode_reward}, Length = {episode_length}")

    # Save the model
    model.save("ppo_hover")

    # Plot the rewards and episode lengths
    # Create a directory to save the plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)

    # Plot the rewards and save as an image
    plt.figure(figsize=(12, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    plt.savefig('plots/episode_rewards.png')

    # Plot the episode lengths and save as an image
    plt.figure(figsize=(12, 5))
    plt.plot(episode_lengths)
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.title('Episode Lengths')
    plt.savefig('plots/episode_lengths.png')

if __name__ == "__main__":
    main()