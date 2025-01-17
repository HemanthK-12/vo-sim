#Making my own custom control backend

#First include Backend Class in pegasus.simulator.logic.backends
import gym
from gym import spaces
import numpy as np
from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.state import State
# from pegasus.simulator.logic.sensors import imu
import numpy as np
from scipy.spatial.transform import Rotation

#then, make class which inherits this Backend class, this class name will only be imported in the main simulation file and will be used in configuring the multirotor vehicle

class RLController(Backend, gym.Env):
    # 1) start() method is invoked when simulation starts. Performing initialization should be main here.
    # 2) Similarly, stop() is invoked when simulation stops. do cleanup here.
    # 3) update_sensor() is invoked by every sensor giving data, like imu, gps, etc. I/P : string containing type of string and the data it is sending to you. generally we use if statements to check for the sensor needed and do work accordingly.
    # 4) update_state() gets the current state of the vehicle and is used to change/update it's state . It gives in the state, info given as (in order) : position, altitude, linear velocity, linear body-velocity, angular velocity, linear acceleration. All use Inertial Frame = East-North-Down, Vehicle's Frame = Front-Left-Up.
    # 5) input_reference() gives the simulation the list of target angular velocities/rpm to give to each rotor. This can access the specifics of the vehicle also through thevehicle class, so if you want to not hardcode values, this'll be good approach. think of this as todo.
    # 6) update() method updates to the simulation, the list of target angular velocities/rpm to give to each rotor, using the values returned by the input_reference() as reference. This will contain core logic generally.
    def __init__(self, desired_altitude=2.0):
        super(RLController, self).__init__()
        self.desired_altitude = desired_altitude
        self.action_space = spaces.Box(low=0, high=1000, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)
        self.state = None
        self.reset()

    def start(self):
        pass

    def stop(self):
        pass

    def update_sensor(self, sensor_type, data):
        if sensor_type == "IMU":
            print("IMU Orientation = ", data['orientation'])
            print("IMU Angular Velocity = ", data['angular_velocity'])
            print("IMU Linear Acceleration = ", data['linear_acceleration'])

    def update_state(self, state: State):
        self.state = {
            'position': state.position,
            'orientation': Rotation.from_quat(state.attitude),
            'angular_velocity': state.angular_velocity,
            'rpm': np.zeros((4,))
        }
        self.received_first_state = True
        print("Position = ", state.position)
        print("Orientation = ", state.attitude)
        print("Linear Velocity = ", state.linear_velocity)
        print("Angular Velocity = ", state.angular_velocity)

    def input_reference(self):
        return self.state['rpm']

    def update(self, dt: float):
        if not self.received_first_state:
            return

        # Placeholder for DQN action
        action = self.state['rpm']
        self.state['rpm'] = action

    def reset(self):
        self.state = {
            'position': np.zeros((3,)),
            'orientation': Rotation.identity(),
            'angular_velocity': np.zeros((3,)),
            'rpm': np.zeros((4,))
        }
        return self._get_obs()

    def step(self, action):
        self.state['rpm'] = action
        # Simulate the drone's response to the action here
        # Update self.state['position'], self.state['orientation'], self.state['angular_velocity']
        reward = -abs(self.state['position'][2] - self.desired_altitude)
        done = False
        info = {}
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        position = self.state['position']
        orientation = self.state['orientation'].as_quat()
        angular_velocity = self.state['angular_velocity']
        rpm = self.state['rpm']
        return np.concatenate([position, orientation, angular_velocity, rpm])

    def render(self, mode='human'):
        pass

    def close(self):
        pass