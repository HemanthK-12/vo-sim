from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})
import omni.timeline
from omni.isaac.core.world import World
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.multirotor import Multirotor, MultirotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
import os
from scipy.spatial.transform import Rotation

class PegasusApp:
    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])
        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())
        
        config_multirotor1 = MultirotorConfig()
        config_multirotor1.backends = [NonlinearController(
            trajectory_file=self.curr_dir + "/trajectories/pitch_relay_90_deg_2.csv",
            results_file=self.curr_dir + "/results/single_statistics.npz",
            Ki=[0.5, 0.5, 0.5],
            Kr=[2.0, 2.0, 2.0]
        )]
        Multirotor(
            "/World/quadrotor1",
            ROBOTS['Iris'],
            0,
            [2.3, -1.5, 0.07],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config_multirotor1,
        )
        self.world.reset()

    def run(self):
        self.timeline.play()
        while simulation_app.is_running():
            self.world.step(render=True)
        self.timeline.stop()
        simulation_app.close()

def main():
    pg_app = PegasusApp()
    pg_app.run()

if __name__ == "__main__":
    main()
