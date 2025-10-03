# Training loop for the RL model.

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from isaac_env import IsaacSimEnv

def main():
    # Create a vectorized environment for training.
    env = make_vec_env(lambda: IsaacSimEnv(), n_envs=1)
    
    # Initialize the PPO model with desired hyperparameters.
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./tensorboard_logs/"
    )
    
    # Train for a specified number of timesteps.
    model.learn(total_timesteps=500000)
    
    # Save the trained model.
    model.save("ppo_hover_trained.zip")

if __name__ == "__main__":
    main()