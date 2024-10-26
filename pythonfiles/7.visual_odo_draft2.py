
import pybullet as p
import numpy as np
import cv2

# Connect to PyBullet
physicsClient = p.connect(p.GUI)

# Load environment and textures
quat = p.getQuaternionFromEuler([1.57, 0, 0])
drone_quat = p.getQuaternionFromEuler([1.57, 0, 0])
drone = p.loadURDF('./cf2/cf2.urdf', globalScaling=2, baseOrientation=drone_quat, basePosition=[0, 0, 2])
forest = p.loadURDF("./forest/forest.urdf", basePosition=[0, 0, 0], baseOrientation=quat, useFixedBase=True)
texture_id = p.loadTexture("./forest/forest.png")
p.changeVisualShape(forest, -1, textureUniqueId=texture_id)

# Camera parameters
fov = 60
aspect = 1
near = 0.001
far = 1000
width, height = 1920, 1088

# Calculate intrinsic matrix
fx = width / (2 * (np.tan(np.deg2rad(fov) / 2)))
fy = height / (2 * (np.tan(np.deg2rad(fov) / 2)))
intrinsic_matrix = np.array([[fx, 0, width / 2],
                             [0, fy, height / 2],
                             [0, 0, 1]])

projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Keyboard control settings
keys = {
    'w': [0.1, 0, 0],
    's': [-0.1, 0, 0],
    'a': [0, -0.1, 0],
    'd': [0, 0.1, 0],
    'q': [0, 0, 0.1],
    'e': [0, 0, -0.1]
}

# Initialize ORB and BFMatcher
orb = cv2.ORB_create()
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Main loop
positions = []
prev_keypoint = None
prev_descriptor = None

for i in range(100):  # 10 seconds at 10 fps

    # Get the drone's position and update view matrix
    drone_position, drone_orientation = p.getBasePositionAndOrientation(drone)
    positions.append(drone_position)

    # Check keyboard events for drone movement
    keys_pressed = p.getKeyboardEvents()
    for key, movement in keys.items():
        if ord(key) in keys_pressed and keys_pressed[ord(key)] & p.KEY_WAS_TRIGGERED:
            drone_position = [drone_position[j] + movement[j] for j in range(3)]
            p.resetBasePositionAndOrientation(drone, drone_position, drone_orientation)

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[drone_position[0], drone_position[1], drone_position[2] - 0.1],
        cameraTargetPosition=[drone_position[0] - 1, drone_position[1], drone_position[2]],
        cameraUpVector=[1, 0, 0]
    )

    # Capture image from the camera
    images = p.getCameraImage(width, height, view_matrix, projection_matrix)
    rgba_img = np.reshape(images[2], (height, width, 4))
    rgb_img = rgba_img[:, :, :3]
    
    # Detect ORB keypoints and descriptors
    keypoint_list, descriptor_list = orb.detectAndCompute(rgb_img, None)

    # Match with previous frame if exists
    if prev_keypoint is not None and descriptor_list is not None:
        matches = bf_matcher.match(prev_descriptor, descriptor_list)
        matches = sorted(matches, key=lambda x: x.distance)

        # Only proceed if there are enough matches
        if len(matches) >= 5:
            # Extract matched points from keypoints
            old_keypoints_matched = np.array([prev_keypoint[m.queryIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)
            new_keypoints_matched = np.array([keypoint_list[m.trainIdx].pt for m in matches], dtype=np.float32).reshape(-1, 1, 2)

            # Calculate Essential matrix and decompose pose
            E, _ = cv2.findEssentialMat(new_keypoints_matched, old_keypoints_matched, intrinsic_matrix)
            _, R, t, _ = cv2.recoverPose(E, new_keypoints_matched, old_keypoints_matched, intrinsic_matrix)

            # Project 3D points
            old_projection_matrix = np.hstack((np.eye(3), np.zeros((3, 1))))
            new_projection_matrix = np.hstack((R, t))
            homogenous_4d_pts = cv2.triangulatePoints(
                old_projection_matrix, new_projection_matrix,
                old_keypoints_matched.T, new_keypoints_matched.T
            )

            # Convert to 3D coordinates
            estimated_3d_points = homogenous_4d_pts[:3] / homogenous_4d_pts[3]
            print("Drone position:", drone_position, "\nEstimated 3D points:", estimated_3d_points.T)

    # Update previous keypoints and descriptors
    prev_descriptor = descriptor_list
    prev_keypoint = keypoint_list

p.disconnect()
cv2.destroyAllWindows()


# #This code spawns the drone and the environment in the pybullet space and records the drone's value through keyboard events and later plots the trajectory of the drone in a matplotlib 3d graph.

# import pybullet as p
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt

# physicsClient = p.connect(p.GUI)

# env_quat = p.getQuaternionFromEuler([1.57, 0, 0])
# drone_quat = p.getQuaternionFromEuler([1.57, 0, 0])
# drone = p.loadURDF('./cf2/cf2.urdf', globalScaling=2, baseOrientation=drone_quat, basePosition=[0, 0, 0])
# forest = p.loadURDF("./forest/forest.urdf", basePosition=[0, 0, 0], baseOrientation=env_quat, useFixedBase=True)
# texture_id = p.loadTexture("./forest/forest.png")
# p.changeVisualShape(forest, -1, textureUniqueId=texture_id)

# fov = 60  # Field of view(degrees)
# aspect = 1  # Aspect ratio
# near = 0.001  # nearest distance which can be resolved
# far = 1000  # farthest distance till which is rendered in the image
# width, height = 1920, 1088 

# #to define a camera's view, you need :
# # the position of the camera
# # a point(target) in the direction the camera is looking at and
# # up direction of camera.
# camera_position = [0, 0, -0.1]  # Slightly below the drone
# camera_target = [-1, 0, 0]  
# camera_up_vector=[1, 0, 0]

# projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far) #how 3d points are projected onto 2d images

# frame_rate = 10 
# duration_seconds = 10 
# total_frames = frame_rate * duration_seconds

# keys = {
#     'w': [0.1, 0, 0],
#     's': [-0.1, 0, 0],
#     'a': [0, -0.1, 0],
#     'd': [0, 0.1, 0],
#     'q': [0, 0, 0.1],
#     'e': [0, 0, -0.1]
# }
# positions=[]

# for i in range(total_frames):

#     drone_position, drone_orientation = p.getBasePositionAndOrientation(drone)
#     positions.append(drone_position)#for drawing trajectory

#     # seeing if keys are pressed or not
#     keys_pressed = p.getKeyboardEvents()
#     for key, value in keys.items():
#         if ord(key) in keys_pressed and keys_pressed[ord(key)] & p.KEY_WAS_TRIGGERED: #if ascii value of character is in the list of keys pressed
#             for j in range(3):
#                 drone_position[j]+= value[j] #adding 0.1 to drone's position for new position
#             p.resetBasePositionAndOrientation(drone, drone_position, drone_orientation)
    
#     # Compute camera's view matrix (camera positioned below drone, looking downward)
#     view_matrix = p.computeViewMatrix(cameraEyePosition=[drone_position[0] + camera_position[0],
#                                                          drone_position[1] + camera_position[1],
#                                                          drone_position[2] + camera_position[2]],cameraTargetPosition=[drone_position[0] + camera_target[0],
#                                                                                                                         drone_position[1] + camera_target[1],
#                                                                                                                          drone_position[2] + camera_target[2]],cameraUpVector=camera_up_vector)

#     images = p.getCameraImage(width, height, view_matrix, projection_matrix)
    
#     rgba_img = np.reshape(images[2], (height, width, 4))  # RGBA image
#     rgb_img = rgba_img[:, :, :3]  # Convert to RGB by ignoring the alpha channel

# p.disconnect()

# positions = list(zip(*positions))  # Transpose to get X, Y, Z coordinates separately
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(positions[0], positions[1], positions[2], label='Drone Trajectory')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.legend()
# # plt.savefig('./trajectories/2.png')  # Save the figure as a PNG image
# plt.show()
