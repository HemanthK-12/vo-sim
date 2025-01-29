from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.state import State
import numpy as np
from scipy.spatial.transform import Rotation
from stable_baselines3 import DQN

class RLController(Backend):
    def __init__(self, desired_altitude, Kp, Kd):
        self.desired_altitude = desired_altitude
        self.Kp = Kp
        self.Kd = Kd
        self.input_ref = [0.0, 0.0, 0.0, 0.0]
        self.p = np.zeros((3,)) # position
        self.v = np.zeros((3,)) # velocity
        self.R = Rotation.identity() # orientation
        self.w = np.zeros((3,)) # angular velocity
        self.received_first_state = False
        self.m = 1.50        # Mass in Kg
        self.g = 9.81       # The gravity acceleration ms^-2
        self.Kp_attitude = np.array([1.0, 1.0, 0.0])  # Proportional gains for attitude control
        self.Kd_attitude = np.array([0.1, 0.1, 0.0])  # Derivative gains for attitude control
        self.model = DQN.load("dqn_cartpole")  # Load the trained model

    def start(self):
        pass

    def stop(self):
        pass

    def update_sensor(self, sensor_type, data):
        pass

    def update_state(self, state: State):
        self.p = state.position
        self.v = state.linear_velocity
        self.R = Rotation.from_quat(state.attitude)
        self.w = state.angular_velocity
        self.received_first_state = True

    def input_reference(self):
        return self.input_ref

    def update(self, dt: float):
        if not self.received_first_state:
            return

        obs = np.concatenate([self.p, self.v, self.R.as_quat(), self.w])
        action, _states = self.model.predict(obs, deterministic=True)
        self.input_ref = action

    def reset(self):
        pass