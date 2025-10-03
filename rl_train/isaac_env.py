"""
Custom gym environment for connecting with ISAACSIM via Pegasus.
This version:
 - Sets up the simulation world.
 - Creates a drone with a backend (RLController).
 - Provides observation, reward, step, reset functions.
"""
from gymnasium import spaces
import numpy as np
import gymnasium as gym
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
from omni.isaac.core.world import World
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.sensors.imu import IMU
from scipy.spatial.transform import Rotation
from custom_control_backend import RLController

class IsaacSimEnv(gym.Env):
    def __init__(self):
        super(IsaacSimEnv, self).__init__()
        
        # Create a basic world instance (or use PegasusInterface's world if needed)
        self.pg_world = World()  
        
        # Create backend controller with chosen parameters
        self.rl_controller = RLController(Kp=2.0, Kd=0.5)
        
        # Multirotor configuration: assign the backend in _backends.
        config_multirotor = MultirotorConfig()
        config_multirotor._backends = [self.rl_controller]
        
        # Create the drone with an initial elevated position (e.g., 5 m) for stability.
        initial_position = [0.0, 0.0, 5.0]
        initial_attitude = Rotation.from_euler("XYZ", [0, 0, 0], degrees=True).as_quat()
        self.drone = Multirotor(
            "/World/quadrotor1",
            ROBOTS['Iris'],
            0,
            initial_position,
            initial_attitude,
            config=config_multirotor,
        )
        
        # Use the created world. (Adjust if your simulation uses a different world.)
        self.world = self.pg_world
        
        # Define normalized action space [-1, 1] for each rotor; scaling will occur later.
        self.min_rotor_speed = 4000.0
        self.max_rotor_speed = 10000.0
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        
        # Observation: 3 (position) + 3 (linear velocity) + 4 (attitude) + 3 (angular velocity)
        # + 4 (IMU orientation) + 3 (IMU angular velocity) + 3 (IMU linear acceleration) = 23.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)
    
    def _get_obs(self):
        # Get state from backend for the drone.
        position = self.rl_controller.p
        linear_velocity = self.rl_controller.v
        attitude = self.rl_controller.R.as_quat()
        angular_velocity = self.rl_controller.w
        imu_data = self.rl_controller.imu_data
        if imu_data is None:
            imu_orientation = np.zeros(4)
            imu_angular_velocity = np.zeros(3)
            imu_linear_acceleration = np.zeros(3)
        else:
            imu_orientation = imu_data.get('orientation', np.zeros(4))
            imu_angular_velocity = imu_data.get('angular_velocity', np.zeros(3))
            imu_linear_acceleration = imu_data.get('linear_acceleration', np.zeros(3))
        obs = np.concatenate([
            position,
            linear_velocity,
            attitude,
            angular_velocity,
            imu_orientation,
            imu_angular_velocity,
            imu_linear_acceleration
        ])
        info = {}
        return obs, info

    def _compute_reward(self, obs):
        # Define target states.
        target_altitude = 10.0
        target_attitude = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Extract observation components (assumed indices by construction).
        altitude = obs[2]                # z-coordinate from position.
        linear_velocity = obs[3:6]
        attitude = obs[6:10]
        angular_velocity = obs[10:13]
        
        # Use linear penalty for altitude error.
        altitude_error = altitude - target_altitude
        altitude_reward = -np.abs(altitude_error)
        
        # Stability penalties.
        velocity_penalty = -0.5 * np.clip(np.linalg.norm(linear_velocity), 0, 5)
        angular_velocity_penalty = -0.5 * np.clip(np.linalg.norm(angular_velocity), 0, 5)
        
        # Attitude penalty.
        attitude_error = np.linalg.norm(attitude - target_attitude)
        attitude_reward = -attitude_error
        
        # Combined weighted reward.
        reward = (5.0 * altitude_reward +
                  2.0 * attitude_reward +
                  1.0 * velocity_penalty +
                  1.0 * angular_velocity_penalty)
        return reward

    def _is_terminated(self, obs):
        # Terminate if the drone flies too low or too high.
        altitude = obs[2]
        return altitude < 0 or altitude > 30

    def step(self, action):
        # Scale action from normalized [-1,1] to the rotor speed range.
        scaled_action = self.min_rotor_speed + (action + 1.0) * 0.5 * (self.max_rotor_speed - self.min_rotor_speed)
        # Add a base thrust to help maintain altitude.
        base_thrust = 6000.0
        scaled_action = scaled_action + base_thrust
        # Clip commands within valid RPM.
        scaled_action = np.clip(scaled_action, self.min_rotor_speed, self.max_rotor_speed)
        # Set the input reference for the drone's backend.
        self.rl_controller.input_ref = scaled_action.tolist()
        
        # Step simulation.
        self.world.step(render=False)
        
        obs, info = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = self._is_terminated(obs)
        truncated = False
        
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        # Reset world and backend state.
        self.world.reset()
        self.rl_controller.reset()
        obs, info = self._get_obs()
        return obs, info