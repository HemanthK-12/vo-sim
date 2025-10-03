from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.state import State
import numpy as np
from omni.isaac.core.world import World
from pegasus.simulator.logic.sensors.imu import IMU
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

class RLController(Backend):
    def __init__(self, desired_altitude, Kp, Kd):
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Initialize controller parameters
        self.desired_altitude = desired_altitude
        self.Kp = Kp
        self.Kd = Kd
        self.input_ref = [0.0, 0.0, 0.0, 0.0]  # Four rotor speeds
        
        # State variables
        self.p = np.zeros((3,))  # position
        self.v = np.zeros((3,))  # velocity
        self.R = Rotation.identity()  # orientation
        self.w = np.zeros((3,))  # angular velocity
        self.received_first_state = False
        
        # IMU setup
        self.imu = IMU()
        self.imu_data = None

        # Load the trained PPO model
        try:
            self.model = PPO.load("ppo_hover.zip")
            print("Successfully loaded PPO model")
        except Exception as e:
            print(f"Error loading PPO model: {e}")
            self.model = None

    def update_state(self, state: State):
        # Update internal state
        self.p = state.position
        self.v = state.linear_velocity
        self.R = Rotation.from_quat(state.attitude)
        self.w = state.angular_velocity
        self.imu_data = self.imu.update(state, self.world.get_physics_dt())
        self.received_first_state = True

    def update(self, dt: float):
        if not self.received_first_state or self.model is None:
            return

        # Prepare observation vector for the model
        obs = np.concatenate([
            self.p,                  # position (3)
            self.v,                  # velocity (3)
            self.R.as_quat(),       # attitude quaternion (4)
            self.w,                  # angular velocity (3)
            self.imu_data['orientation'],        # IMU orientation (4)
            self.imu_data['angular_velocity'],   # IMU angular velocity (3)
            self.imu_data['linear_acceleration'] # IMU acceleration (3)
        ])

        try:
            # Get action from PPO model
            action, _ = self.model.predict(obs, deterministic=True)
            
            # Scale action from [-1, 1] to [0, 1] for rotor speeds
            scaled_action = (action + 1.0) * 0.5
            
            # Update rotor speeds
            self.input_ref = scaled_action.tolist()
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            self.input_ref = [0.0, 0.0, 0.0, 0.0]  # Safe fallback

    def input_reference(self):
        return self.input_ref

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        self.received_first_state = False
        self.input_ref = [0.0, 0.0, 0.0, 0.0]

    def update_sensor(self, sensor_type, data):
        pass

    def update_graphical_sensor(self, sensor_type, data):
        pass