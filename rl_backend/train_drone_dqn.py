import gym
import numpy as np
from rlenv_controller import RLController
from dqn_model import DQNAgent
import torch

env = RLController()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = DQNAgent(state_dim, action_dim)

episodes = 1000
for e in range(episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_dim])
    for time in range(500):
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_dim])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            agent.update_target_model()
            print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
            break
        agent.replay()

torch.save(agent.model.state_dict(), "dqn_drone_hover.pth")