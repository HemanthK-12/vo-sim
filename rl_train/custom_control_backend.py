from pegasus.simulator.logic.backends import Backend
from pegasus.simulator.logic.state import State
import numpy as np
from omni.isaac.core.world import World
from pegasus.simulator.logic.sensors.imu import IMU
from scipy.spatial.transform import Rotation
from stable_baselines3 import PPO
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

class RLController(Backend):
    def __init__(self, Kp, Kd):
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.Kp = Kp
        self.Kd = Kd
        self.input_ref = [0.0, 0.0, 0.0, 0.0]  # rotor commands
        self.p = np.zeros((3,))              # position
        self.v = np.zeros((3,))              # velocity
        self.R = Rotation.identity()         # orientation
        self.w = np.zeros((3,))              # angular velocity
        self.received_first_state = False
        self.imu = IMU()
        self.imu_data = None

        # Load the trained PPO model (if available)
        try:
            self.model = PPO.load("ppo_hover.zip")
            print("PPO model loaded successfully in backend.")
        except Exception as e:
            print(f"Error loading PPO model: {e}")
            self.model = None

    def start(self):
        # Optionally add code to start the controller
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
        # Update IMU state using the current state and simulation dt
        self.imu_data = self.imu.update(state, self.world.get_physics_dt())
        self.received_first_state = True
        print(f"Current position: {self.p}")

    def update(self, dt: float):
        # Compute rotor commands from current state using the PPO model.
        if not self.received_first_state:
            return
        if self.model is None:
            print("No model loaded; cannot predict action.")
            return

        # Build observation: position (3), velocity (3), attitude (4),
        # angular velocity (3), and IMU data (orientation (4),
        # angular velocity (3), linear acceleration (3)) => 23 values.
        obs = np.concatenate([
            self.p,
            self.v,
            self.R.as_quat(),
            self.w,
            self.imu_data.get('orientation', np.zeros(4)),
            self.imu_data.get('angular_velocity', np.zeros(3)),
            self.imu_data.get('linear_acceleration', np.zeros(3))
        ])
        # Predict action from the model (assumed to output values in [-1, 1])
        action, _ = self.model.predict(obs, deterministic=True)
        # Scale predicted action from [-1, 1] to rotor RPM range [4000, 10000]
        scaled_action = 4000.0 + (action + 1.0) * 0.5 * (10000.0 - 4000.0)
        self.input_ref = scaled_action.tolist()

    def input_reference(self):
        return self.input_ref

    def reset(self):
        self.received_first_state = False
        self.input_ref = [0.0, 0.0, 0.0, 0.0]

    def update_graphical_sensor(self, sensor_type, data):
        pass