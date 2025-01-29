#Making my own custom control backend

#First include Backend Class in pegasus.simulator.logic.backends
from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.state import State
# from pegasus.simulator.logic.sensors import imu
import numpy as np
from scipy.spatial.transform import Rotation

#then, make class which inherits this Backend class, this class name will only be imported in the main simulation file and will be used in configuring the multirotor vehicle

class RLController(Backend):
    # 1) start() method is invoked when simulation starts. Performing initialization should be main here.
    # 2) Similarly, stop() is invoked when simulation stops. do cleanup here.
    # 3) update_sensor() is invoked by every sensor giving data, like imu, gps, etc. I/P : string containing type of string and the data it is sending to you. generally we use if statements to check for the sensor needed and do work accordingly.
    # 4) update_state() gets the current state of the vehicle and is used to change/update it's state . It gives in the state, info given as (in order) : position, altitude, linear velocity, linear body-velocity, angular velocity, linear acceleration. All use Inertial Frame = East-North-Down, Vehicle's Frame = Front-Left-Up.
    # 5) input_reference() gives the simulation the list of target angular velocities/rpm to give to each rotor. This can access the specifics of the vehicle also through the vehicle class, so if you want to not hardcode values, this'll be good approach. think of this as todo.
    # 6) update() method updates to the simulation, the list of target angular velocities/rpm to give to each rotor, using the values returned by the input_reference() as reference. This will contain core logic generally.
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
        print(state.position)

    def input_reference(self):
        return self.input_ref

    def update(self, dt: float):
        if not self.received_first_state:
            return

        if(self.p[2]<self.desired_altitude):
            thrust = self.m * self.g * 300
        else:
            thrust = 0
        # Combine thrust and attitude control
        self.input_ref = [
            thrust / 4 ,
            thrust / 4 ,
            thrust / 4 ,
            thrust / 4 
        ]

    def reset(self):
        pass

    def update_graphical_sensor(self, sensor_type, data):
        pass