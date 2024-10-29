import pybullet as p
import pybullet_data
import time

# Initialize PyBullet
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load the plane and drone URDF
plane_id = p.loadURDF("plane.urdf")
drone_id = p.loadURDF("./cf2/cf2.urdf", [0, 0, 1])

# Define the control parameters

move_step = 0.01

# Main simulation loop
while True:
    keys = p.getKeyboardEvents()
    
    # Get the current position and orientation of the drone
    current_position, current_orientation = p.getBasePositionAndOrientation(drone_id)
    
    # Update the position based on key input
    if ord('w') in keys and keys[ord('w')] & p.KEY_IS_DOWN:
        current_position = [current_position[0], current_position[1] + move_step, current_position[2]]
    if ord('s') in keys and keys[ord('s')] & p.KEY_IS_DOWN:
        current_position = [current_position[0], current_position[1] - move_step, current_position[2]]
    if ord('a') in keys and keys[ord('a')] & p.KEY_IS_DOWN:
        current_position = [current_position[0] - move_step, current_position[1], current_position[2]]
    if ord('d') in keys and keys[ord('d')] & p.KEY_IS_DOWN:
        current_position = [current_position[0] + move_step, current_position[1], current_position[2]]
    if ord('q') in keys and keys[ord('q')] & p.KEY_IS_DOWN:
        current_position = [current_position[0], current_position[1], current_position[2] + move_step]
    if ord('e') in keys and keys[ord('e')] & p.KEY_IS_DOWN:
        current_position = [current_position[0], current_position[1], current_position[2] - move_step]
    
    # Set the new position of the drone
    p.resetBasePositionAndOrientation(drone_id, current_position, current_orientation)
    
    # Step the simulation
    p.stepSimulation()
    time.sleep(time_step)

# Disconnect from PyBullet
p.disconnect()