import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time

physicsClient = p.connect(p.GUI)

quat = p.getQuaternionFromEuler([1.57, 0, 0])
drone_quat = p.getQuaternionFromEuler([1.57, 0, 0])
drone = p.loadURDF('./cf2/cf2.urdf', globalScaling=2, baseOrientation=drone_quat, basePosition=[0, 0, 2])
forest = p.loadURDF("./forest/forest.urdf", basePosition=[0, 0, 0], baseOrientation=quat, useFixedBase=True)
texture_id = p.loadTexture("./forest/forest.png")
p.changeVisualShape(forest, -1, textureUniqueId=texture_id)

fov = 60
aspect = 1
near = 0.001
far = 1000
width, height = 960, 540

camera_position = [0, 0, -0.1]
camera_target = [-1, 0, 0]

projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

frame_rate = 10
duration = 10
total_frames = frame_rate * duration

orb = cv2.ORB_create()
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)  # BFMatcher

prev_descriptors = None
prev_keypoints = None

for i in range(total_frames):
    drone_position, drone_orientation = p.getBasePositionAndOrientation(drone)

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=[drone_position[0] + camera_position[0],
                           drone_position[1] + camera_position[1],
                           drone_position[2] + camera_position[2]],
        cameraTargetPosition=[drone_position[0] + camera_target[0],
                              drone_position[1] + camera_target[1],
                              drone_position[2] + camera_target[2]],
        cameraUpVector=[1, 0, 0]
    )

    images = p.getCameraImage(width, height, view_matrix, projection_matrix)
    rgba_img = np.reshape(images[2], (height, width, 4))
    rgb_img = rgba_img[:, :, :3]
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    keypoints, descriptors = orb.detectAndCompute(gray_img, None)

    if prev_descriptors is not None:
        matches = bf.match(prev_descriptors, descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        img_matches = cv2.drawMatches(prev_img, prev_keypoints, rgb_img, keypoints, matches[:10], None, flags=2)
        cv2.imshow("Feature Matching", img_matches)
        cv2.waitKey(1)

    prev_descriptors = descriptors
    prev_keypoints = keypoints
    prev_img = rgb_img.copy()

p.disconnect()
cv2.destroyAllWindows()
