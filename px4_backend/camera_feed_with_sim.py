from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
#this is written in the middle as omni.isaac has only kit module before starting omniverse's isaac sim

#but after the simulation app is started and isaac sim is running, omni.isaac has all these modules(i used dir(omni.isaac) in python to get all these) : IsaacSensorSchema', 'RangeSensorSchema','block_world', 'cloner', 'core', 'core_archive', 'core_nodes', 'cortex', 'debug_draw', 'dynamic_control', 'franka', 'kit', 'lula', 'manipulators', 'menu', 'ml_archive', 'motion_generation', 'nucleus', 'occupancy_map', 'quadruped', 'range_sensor', 'scene_blox', 'sensor', 'surface_gripper', 'ui', 'universal_robots', 'utils', 'version', 'wheeled_robots
import omni.timeline #this manages the playing simulation in isaacsim
from omni.isaac.core.world import World #this creates the scene

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS#robots for importing the iris quadcopter,simulation_environment is to import environments like default, hospital,office,etc.
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig#same, this too fro importing iris quadcopter

from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig#to configure px4 autopilot to isaac sim through pegasus, this uses mavlink as the comm protocol

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface#we need to import this to manage all interfaces between pegasus and isaac sim

from omni.isaac.sensor import Camera#to keep the cameras of third-person view and drone-first person view
from omni.isaac.core.utils.prims import move_prim#to make the camera the child of the drone
from scipy.spatial.transform import Rotation
import numpy as np

class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Warehouse"])
        #Out of : 
        # Default Environment
        # Black Gridroom
        # Curved Gridroom
        # Hospital
        # Office
        # Simple Room
        # Warehouse
        # Warehouse with Forklifts
        # Warehouse with Shelves
        # Full Warehouse
        # Flat Plane
        # Rough Plane
        # Slope Plane
        # Stairs Plane

        # setting all attributes of quadcopter
        config_multirotor = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]
        self.drone = Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )
        #camera to see the drone, i.e. third person view
        camera_path = "/World/quadrotor/Camera"
        self.camera = Camera(
            prim_path=camera_path,
            position=np.array([0.0, 0.0, 0.5]),  # i kept the drone 0.5 units above drone
            orientation=np.array([0.0, 0.0, 0.0, 1.0])  # normal orientation
        )
        self.camera.initialize()

        self.world.reset()
        self.stop_sim = False
    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and not self.stop_sim:
            drone_position, drone_orientation = self.drone.get_world_pose()
            self.camera.set_world_pose(drone_position + np.array([0.0, 0.0, 0.5]), drone_orientation)
            self.world.step(render=True)

        self.timeline.stop()
        simulation_app.close()

pg_app = PegasusApp()
pg_app.run()
#TODO: should see some way to divide the __init__ part of the class into another file, maybe setup.py and call it, will be simpler to write core logic here
