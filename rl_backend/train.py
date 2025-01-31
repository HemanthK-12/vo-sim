import gymnasium as gym
from isaac_env import IsaacSimEnv
import time
def main():
    env = IsaacSimEnv()
    obs, info = env.reset()
    print("Initial Observation:", obs)

    for _ in range(10000):
        action = env.action_space.sample()  # Random action for testing
        obs, reward, terminated, truncated, info = env.step(action)
        print("Observation:", obs)
        print("Reward:", reward)
        print("Terminated:", terminated)
        print("Truncated:", truncated)
        if terminated or truncated:
            obs, info = env.reset()
            print("Reset Observation:", obs)
        time.sleep(0.1)
if __name__ == "__main__":
    main()