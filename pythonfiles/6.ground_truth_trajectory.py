
#This code spawns the drone and the environment in the pybullet space and records the drone's movement through keyboard events and later plots the trajectory of the drone in a matplotlib 3d graph.

import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
physicsClient = p.connect(p.GUI)


quat = p.getQuaternionFromEuler([1.57, 0, 0])
drone_quat = p.getQuaternionFromEuler([1.57, 0, 0])
drone = p.loadURDF('./cf2/cf2.urdf', globalScaling=2, baseOrientation=drone_quat, basePosition=[0, 0, 2])
forest = p.loadURDF("./forest/forest.urdf", basePosition=[0, 0, 0], baseOrientation=quat, useFixedBase=True)
texture_id = p.loadTexture("./forest/forest.png")
p.changeVisualShape(forest, -1, textureUniqueId=texture_id)

fov = 60  # Field of view in degrees
aspect = 1  # Aspect ratio
near = 0.001  # Near clipping plane
far = 1000  # Far clipping plane
width, height = 1920, 1088  # Image size

# Camera position and orientation relative to the drone (looking down)
camera_position = [0, 0, -0.1]  # Slightly below the drone
camera_target = [-1, 0, 0]  # Pointing downward

# Camera projection matrix
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

frame_rate = 10  # Frames per second
duration = 10  # Duration in seconds
total_frames = frame_rate * duration

keys = {
    'w': [0.1, 0, 0],
    's': [-0.1, 0, 0],
    'a': [0, -0.1, 0],
    'd': [0, 0.1, 0],
    'q': [0, 0, 0.1],
    'e': [0, 0, -0.1]
}
positions=[]
for i in range(total_frames):
    # Get the drone's position and orientation
    drone_position, drone_orientation = p.getBasePositionAndOrientation(drone)
    positions.append(drone_position)
    # Check for keyboard input
    keys_pressed = p.getKeyboardEvents()
    for key, movement in keys.items():
        if ord(key) in keys_pressed and keys_pressed[ord(key)] & p.KEY_WAS_TRIGGERED:
            drone_position = [drone_position[j] + movement[j] for j in range(3)]
            p.resetBasePositionAndOrientation(drone, drone_position, drone_orientation)
    
    # Compute camera's view matrix (camera positioned below drone, looking downward)
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[drone_position[0] + camera_position[0],
                           drone_position[1] + camera_position[1],
                           drone_position[2] + camera_position[2]],
        cameraTargetPosition=[drone_position[0] + camera_target[0],
                              drone_position[1] + camera_target[1],
                              drone_position[2] + camera_target[2]],
        cameraUpVector=[1, 0, 0]  # Assuming the up direction is along the x-axis
    )
    
    # Render the camera images
    images = p.getCameraImage(width, height, view_matrix, projection_matrix)
    
    # Extract the RGB image
    rgba_img = np.reshape(images[2], (height, width, 4))  # RGBA image
    rgb_img = rgba_img[:, :, :3]  # Convert to RGB by ignoring the alpha channel
    
    time.sleep(1 / frame_rate)
out.release()
p.disconnect()

positions = list(zip(*positions))  # Transpose to get X, Y, Z coordinates separately
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[0], positions[1], positions[2], label='Drone Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
# plt.savefig('./trajectories/2.png')  # Save the figure as a PNG image
plt.show()
