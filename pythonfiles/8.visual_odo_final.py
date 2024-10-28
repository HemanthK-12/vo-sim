import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

physicsClient = p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
quat = p.getQuaternionFromEuler([1.57, 0, 0])
drone_quat = p.getQuaternionFromEuler([1.57, 0, 0])
drone = p.loadURDF('./cf2/cf2.urdf', globalScaling=2, baseOrientation=drone_quat, basePosition=[0, 0, 0])
forest = p.loadURDF("./forest/forest.urdf", basePosition=[0, 0, 0], baseOrientation=quat, useFixedBase=True)
texture_id = p.loadTexture("./forest/forest.png")
p.changeVisualShape(forest, -1, textureUniqueId=texture_id)

# Camera and intrinsic parameters
fov = 60
aspect = 1
near = 0.001
far = 1000
width, height = 1920, 1088
camera_position = [0, 0, -0.1]
camera_target = [-1, 0, 0]
fx = width / (2 * (np.tan(np.deg2rad(fov) / 2)))
fy = height / (2 * (np.tan(np.deg2rad(fov) / 2)))

intrinsic_matrix = np.array([[fx, 0, width / 2],
                             [0, fy, height / 2],
                             [0, 0, 1]])
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# ORB and BFMatcher
orb = cv2.ORB_create()
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Keyboard control and settings
keys = {'w': [0.1, 0, 0], 's': [-0.1, 0, 0], 'a': [0, -0.1, 0], 'd': [0, 0.1, 0], 'q': [0, 0, 0.1], 'e': [0, 0, -0.1]}
positions = []
estimated=[]
# Main loop for capturing frames and estimating position
prev_keypoint = None
prev_descriptor = None
while(1):  # Loop for a set number of frames
    estimated_position=[]
    drone_position, drone_orientation = p.getBasePositionAndOrientation(drone)
    positions.append(drone_position)
    prev_position = drone_position
    # Move the drone based on keyboard events
    keys_pressed = p.getKeyboardEvents()
    if ord('x') in keys_pressed and keys_pressed[ord('x')] & p.KEY_WAS_TRIGGERED:
        break
    for key, movement in keys.items():
        if ord(key) in keys_pressed and keys_pressed[ord(key)] & p.KEY_WAS_TRIGGERED:
            drone_position = [drone_position[j] + movement[j] for j in range(3)]
            p.resetBasePositionAndOrientation(drone, drone_position, drone_orientation)
    if np.all(prev_position == drone_position):
        continue
    # Set up the camera view
    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[drone_position[0] + camera_position[0],
                           drone_position[1] + camera_position[1],
                           drone_position[2] + camera_position[2]],
        cameraTargetPosition=[drone_position[0] + camera_target[0],
                              drone_position[1] + camera_target[1],
                              drone_position[2] + camera_target[2]],
        cameraUpVector=[1, 0, 0]
    )

    # Capture image from the camera
    images = p.getCameraImage(width, height, view_matrix, projection_matrix)
    rgba_img = np.reshape(images[2], (height, width, 4))
    rgb_img = rgba_img[:, :, :3]
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

    # Detect ORB keypoints and descriptors
    keypoint_list, descriptor_list = orb.detectAndCompute(rgb_img, None)
    if prev_keypoint is not None and descriptor_list is not None:
        matches = bf_matcher.match(prev_descriptor, descriptor_list)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points
        old_keypoints_matched = np.array([prev_keypoint[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
        new_keypoints_matched = np.array([keypoint_list[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

        # Find the essential matrix and recover pose
        if len(matches) >= 5:
            E, _ = cv2.findEssentialMat(new_keypoints_matched, old_keypoints_matched, intrinsic_matrix)
            _, R, t, _ = cv2.recoverPose(E, new_keypoints_matched, old_keypoints_matched, intrinsic_matrix)
            scale_factor = 0.1 / np.linalg.norm(t)  # Normalize t and scale it to match the 0.1 unit move
            t = (t * scale_factor).ravel()  # Scale and convert to a flat list
            for i in range(0,3):
                estimated_position.append(prev_position[i] + t[i])
            estimated.append(estimated_position)
            # Display the positions
            print("Translation matrix: \n", t)
            print(f"Actual translation : \n{np.array(drone_position)-np.array(prev_position)}")
            print(f"Estimated position : \n{estimated_position}")
            print(f"Actual position : \n{drone_position}")
            print(f"Difference between actual and estimated position: {np.array(drone_position) - np.array(estimated_position)}\n")
    # Update previous keypoints and descriptors
    prev_descriptor = descriptor_list
    prev_keypoint = keypoint_list


p.disconnect()
positions = list(zip(*positions))  # Transpose to get X, Y, Z coordinates separately
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[0], positions[1], positions[2], label='Drone Trajectory')
ax.plot(estimated[0], estimated[1], estimated[2], label='Estimated Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()
# plt.savefig('./trajectories/2.png')  # Save the figure as a PNG image
plt.show()

