import gymnasium as gym
from gymnasium import spaces
import numpy as np
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
import omni.timeline
from omni.isaac.core.world import World
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from scipy.spatial.transform import Rotation

class ControlAction:
    def __init__(self, rotor_speeds):
        self.joint_positions = None
        self.joint_velocities = None
        self.joint_efforts = None
        self.rotor_speeds = rotor_speeds
        self.joint_indices=None

class IsaacSimEnv(gym.Env):
    def __init__(self):
        super(IsaacSimEnv, self).__init__()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Default Environment"])

        config_multirotor1 = MultirotorConfig()
        self.drone = Multirotor(
            "/World/quadrotor1",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.0],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor1,
        )
        self.world.reset()

        # Define action and observation space
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

    def reset(self):
        self.world.reset()
        obs, info = self._get_obs()
        return obs, info

    def step(self, action):
        control_action = ControlAction(rotor_speeds=action)
        self.drone.apply_action(control_action)
        self.world.step(render=False)
        obs, info = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = self._is_terminated(obs)
        truncated = False
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        position,attitude = self.drone.get_world_pose()
        obs = np.concatenate([position, self.drone.get_linear_velocity(), attitude, self.drone.get_angular_velocity()])
        info = {}
        return obs, info

    def _compute_reward(self, obs):
        # Simple reward function for testing
        reward = -np.linalg.norm(obs[:3])  # Negative distance from origin
        return reward

    def _is_terminated(self, obs):
        # Simple termination condition for testing
        if np.linalg.norm(obs[:3]) > 10.0:  # Terminate if the drone is too far from the origin
            return True
        return False