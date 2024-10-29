import pybullet as p
import pybullet_data
import numpy as np
import cv2
import time
import datetime
import matplotlib.pyplot as plt
plt.ion()
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

drone_quat = p.getQuaternionFromEuler([1.57, 0, 0])
drone = p.loadURDF('./cf2/cf2.urdf', globalScaling=2, baseOrientation=drone_quat, basePosition=[0, 0, 0])
forest = p.loadURDF("./forest/forest.urdf", basePosition=[0, 0, 0], baseOrientation=drone_quat, useFixedBase=True)
texture_id = p.loadTexture("./forest/forest.png")
p.changeVisualShape(forest, -1, textureUniqueId=texture_id)

fov = 60
aspect = 1
near, far = 0.001, 1000
width, height = 640,360 
camera_position = [0, 0, -0.1]
camera_target = [-1, 0, 0]
fx = width / (2 * (np.tan(np.deg2rad(fov) / 2)))
fy = height / (2 * (np.tan(np.deg2rad(fov) / 2)))

intrinsic_matrix = np.array([[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]])
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
orb = cv2.ORB_create()
bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
step=0.1

keys = {
    p.B3G_UP_ARROW: [-step, 0, 0],    # Up Arrow
    p.B3G_DOWN_ARROW: [step, 0, 0],   # Down Arrow
    p.B3G_LEFT_ARROW: [0, -step, 0],  # Left Arrow
    p.B3G_RIGHT_ARROW: [0, step, 0],  # Right Arrow
    ord('1'): [0, 0, -step],          # 'w' key
    ord('2'): [0, 0, step]            # 's' key
}
positions = [[0,0,0]]
estimated = [[0,0,0]]
scale_factor = 0.05
SCALE_ADJUSTMENT_FACTOR = 0.05
prev_keypoint, prev_descriptor = None, None

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

drone_trajectory, = ax.plot([p[0] for p in positions], [p[1] for p in positions], [p[2] for p in positions], label='Drone Trajectory')
estimated_trajectory, = ax.plot([p[0] for p in estimated], [p[1] for p in estimated], [p[2] for p in estimated], label='Estimated Trajectory')
plt.legend()
time_step = 1.0 / 240.0

while True:
    estimated_position = []
    drone_position, drone_orientation = p.getBasePositionAndOrientation(drone)
    positions.append(drone_position)
    prev_position = drone_position

    keys_pressed = p.getKeyboardEvents()
    if ord('x') in keys_pressed and keys_pressed[ord('x')] & p.KEY_IS_DOWN:
        break
    for key, movement in keys.items():
        if key in keys_pressed and keys_pressed[key] & p.KEY_IS_DOWN:
            drone_position = [drone_position[i] + movement[i] for i in range(3)]
            p.resetBasePositionAndOrientation(drone, drone_position, drone_orientation)
    if np.all(prev_position == drone_position):
        continue
    keys_pressed = {}
    p.resetDebugVisualizerCamera(cameraDistance=5, cameraYaw=50, cameraPitch=-35, cameraTargetPosition=drone_position)
    view_matrix = p.computeViewMatrix(cameraEyePosition=[drone_position[0] + camera_position[0], drone_position[1] + camera_position[1], drone_position[2] + camera_position[2]], cameraTargetPosition=[drone_position[0] + camera_target[0], drone_position[1] + camera_target[1], drone_position[2] + camera_target[2]], cameraUpVector=[1, 0, 0])

    images = p.getCameraImage(width, height, view_matrix, projection_matrix)
    rgba_img = np.reshape(images[2], (height, width, 4))
    rgb_img = rgba_img[:, :, :3]
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    keypoint_list, descriptor_list = orb.detectAndCompute(gray_img, None)
    if prev_keypoint is not None and descriptor_list is not None:
        matches = bf_matcher.match(prev_descriptor, descriptor_list)
        matches = sorted(matches, key=lambda x: x.distance)

        quality_matches = [m for m in matches if m.distance < 30]

        old_keypoints_matched = np.array([prev_keypoint[m.queryIdx].pt for m in quality_matches], dtype=np.float32).reshape(-1, 1, 2)
        new_keypoints_matched = np.array([keypoint_list[m.trainIdx].pt for m in quality_matches], dtype=np.float32).reshape(-1, 1, 2)

        if len(quality_matches) >= 8:
            E, mask = cv2.findEssentialMat(new_keypoints_matched, old_keypoints_matched, intrinsic_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, _ = cv2.recoverPose(E, new_keypoints_matched, old_keypoints_matched, intrinsic_matrix)
            t = (t * scale_factor).ravel()

            scale_factor += SCALE_ADJUSTMENT_FACTOR * (0.1 - np.linalg.norm(t))
            estimated_position = [prev_position[i] + t[i] for i in range(3)]
            estimated.append(estimated_position)

            print("Translation matrix:\n", t)
            print(f"Actual translation:\n{np.array(drone_position) - np.array(prev_position)}")
            print(f"Estimated position:\n{estimated_position}")
            print(f"Actual position:\n{drone_position}")
            print(f"Difference between actual and estimated position:{np.array(drone_position) - np.array(estimated_position)}\n\n\n")

    prev_descriptor = descriptor_list
    prev_keypoint = keypoint_list
    ax.set_xlim(min([p[0] for p in positions]) - 1, max([p[0] for p in positions]) + 1)
    ax.set_ylim(min([p[1] for p in positions]) - 1, max([p[1] for p in positions]) + 1)
    ax.set_zlim(min([p[2] for p in positions]) - 1, max([p[2] for p in positions]) + 1)
    drone_trajectory.set_data([p[0] for p in positions], [p[1] for p in positions])
    drone_trajectory.set_3d_properties([p[2] for p in positions])
    estimated_trajectory.set_data([p[0] for p in estimated], [p[1] for p in estimated])
    estimated_trajectory.set_3d_properties([p[2] for p in estimated])

    contact_points = p.getContactPoints()
    if contact_points:
        print("Collision detected!")
        for contact in contact_points:
            print(f"Object A ID: {contact[1]}, Object B ID: {contact[2]}, Position: {contact[5]}")

    plt.draw()
    plt.pause(0.01)
    p.stepSimulation()
    time.sleep(time_step)

p.disconnect()

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'./trajectories/odometry_{timestamp}.png')