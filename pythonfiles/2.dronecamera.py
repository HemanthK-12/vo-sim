import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time

# Connect to PyBullet
physicsClient = p.connect(p.GUI)

# Load the URDF model
quat = p.getQuaternionFromEuler([1.57, 0, 0])
drone_quat = p.getQuaternionFromEuler([1.57, 0, 0])
drone = p.loadURDF('./cf2/cf2.urdf', globalScaling=2, baseOrientation=drone_quat, basePosition=[0, 0, 2])
forest = p.loadURDF("./forest/forest.urdf", basePosition=[0, 0, 0], baseOrientation=quat, useFixedBase=True)
texture_id = p.loadTexture("./forest/forest.png")
p.changeVisualShape(forest, -1, textureUniqueId=texture_id)

# Camera setup
fov = 60  # Field of view in degrees
aspect = 1  # Aspect ratio
near = 0.001  # Near clipping plane
far = 10  # Far clipping plane
width, height = 1920, 1088  # Image size

# Camera position and orientation relative to the drone (looking down)
camera_position = [0, 0, -0.1]  # Slightly below the drone
camera_target = [0, 0, -1]  # Pointing downward

# Camera projection matrix
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Simulate and capture video feed
frame_rate = 10  # Frames per second
duration = 10  # Duration in seconds
total_frames = frame_rate * duration

# Initialize VideoWriter
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('drone_camera_feed.avi', fourcc, frame_rate, (width, height))

# Keyboard control setup
keys = {
    'w': [0.1, 0, 0],
    's': [-0.1, 0, 0],
    'a': [0, -0.1, 0],
    'd': [0, 0.1, 0],
    'q': [0, 0, 0.1],
    'e': [0, 0, -0.1]
}

for i in range(total_frames):
    # Get the drone's position and orientation
    drone_position, drone_orientation = p.getBasePositionAndOrientation(drone)
    
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
    
    # Convert RGB to BGR
    bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    
    # Write the frame to the video file
    out.write(bgr_img)
    
    # Add a small delay to control the simulation speed
    time.sleep(1 / frame_rate)

# Release the VideoWriter
out.release()

# Cleanup
p.disconnect()
