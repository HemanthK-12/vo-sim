#Making my own custom control backend

#First include Backend Class in pegasus.simulator.logic.backends
from pegasus.simulator.logic.backends import Backend

#then, make class which inherits this Backend class, this class name will only be imported in the main simulation file and will be used in configuring the multirotor vehicle

class RLController(Backend):
    # 1) start() method is invoked when simulation starts. Performing initialization should be main here.
    # 2) Similarly, stop() is invoked when simulation stops. do cleanup here.
    # 3) update_sensor() is invoked by every sensor giving data, like imu, gps, etc. I/P : string containing type of string and the data it is sending to you. generally we use if statements to check for the sensor needed and do work accordingly.
    # 4) update_state() gets the current state of the vehicle and is used to change/update it's state . It gives in the state, info given as (in order) : position, altitude, linear velocity, linear body-velocity, angular velocity, linear acceleration. All use Inertial Frame = East-North-Down, Vehicle's Frame = Front-Left-Up.
    # 5) input_reference() gives the simulation the list of target angular velocities/rpm to give to each rotor. This can access the specifics of the vehicle also through thevehicle class, so if you want to not hardcode values, this'll be good approach. think of this as todo.
    # 6) update() method updates to the simulation, the list of target angular velocities/rpm to give to each rotor, using the values returned by the input_refernce() as reference. This will contain core logic generally.
    def start(self):
        pass
    def stop(self):
        pass
    def update_sensor(self,sensor_type,data):
        pass

