from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
import omni.timeline
from omni.isaac.core.world import World
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
import numpy as np
from scipy.spatial.transform import Rotation
from custom_control_backend import RLController
from omni.isaac.sensor import Camera

class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Default Environment"])
        
        config_multirotor1 = MultirotorConfig()
        config_multirotor1.backends = [RLController(
            desired_altitude=0.5,  # Set the desired altitude
            Kp=2.0,                 # Proportional gain
            Kd=0.5                  # Derivative gain
        )]
        self.drone=Multirotor(
            "/World/quadrotor1",
            ROBOTS['Iris'],
            0,
            [0.0,0.0,0.0],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor1,
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

    def run(self):
        self.timeline.play()
        while simulation_app.is_running():
            drone_position, drone_orientation = self.drone.get_world_pose()
            self.camera.set_world_pose(drone_position + np.array([0.0, 0.0, 0.5]), drone_orientation)
            self.world.step(render=True)
        self.timeline.stop()
        simulation_app.close()

def main():
    pg_app = PegasusApp()
    pg_app.run()

if __name__ == "__main__":
    main()
