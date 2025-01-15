from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
#this is needed as other apps
import omni.timeline
from omni.isaac.core.world import World
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from scipy.spatial.transform import Rotation

class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Default Environment"])
        config_multirotor = MultirotorConfig()
        mavlink_config = PX4MavlinkBackendConfig({
            "vehicle_id": 0,
            "px4_autolaunch": True,
            "px4_dir": self.pg.px4_path,
            "px4_vehicle_model": self.pg.px4_default_airframe
        })
        config_multirotor.backends = [PX4MavlinkBackend(mavlink_config)]
        Multirotor(
            "/World/quadrotor",
            ROBOTS['Iris'],
            0,
            [0.0, 0.0, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor,
        )
        self.world.reset()
        self.stop_sim = False
    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
        self.timeline.stop()
        simulation_app.close()
pg_app = PegasusApp()
pg_app.run()