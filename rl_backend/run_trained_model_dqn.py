import gym
import numpy as np
from rlenv_controller import RLController
from dqn_model import DQNAgent

env = RLController()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
agent = DQNAgent(state_dim, action_dim)

# Load the trained model
agent.model.load_state_dict(torch.load("dqn_drone_hover.pth"))
agent.update_target_model()

state = env.reset()
state = np.reshape(state, [1, state_dim])
for _ in range(1000):
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    next_state = np.reshape(next_state, [1, state_dim])
    state = next_state
    env.render()
    if done:
        break