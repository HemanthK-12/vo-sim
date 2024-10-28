import pybullet as p
import pybullet_data
import numpy as np
import cv2
import datetime
import matplotlib.pyplot as plt

physicsClient=p.connect(p.GUI)
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

drone_quat=p.getQuaternionFromEuler([1.57, 0, 0])
drone=p.loadURDF('./cf2/cf2.urdf',globalScaling=2,baseOrientation=drone_quat,basePosition=[0, 0, 0])
forest=p.loadURDF("./forest/forest.urdf",basePosition=[0, 0, 0],baseOrientation=drone_quat,useFixedBase=True)
texture_id=p.loadTexture("./forest/forest.png")
p.changeVisualShape(forest,-1,textureUniqueId=texture_id)

fov=60
aspect=1
near,far=0.001,1000
width,height=1920,1088
camera_position=[0,0,-0.1]
camera_target=[-1,0,0]
fx=width/(2*(np.tan(np.deg2rad(fov)/2)))
fy=height/(2*(np.tan(np.deg2rad(fov)/2)))

intrinsic_matrix=np.array(
                            [[fx,0,width/2],
                            [0,fy,height/2],
                            [0,0,1]]
                        )
projection_matrix=p.computeProjectionMatrixFOV(fov,aspect,near,far)
orb = cv2.ORB_create()
bf_matcher=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
keys={
        'w':[-0.2,0,0],
        's':[0.2,0,0], 
        'a':[0,-0.1,0], 
        'd':[0,0.1,0], 
        'q':[0,0,-0.1], 
        'e':[0,0,0.1]
    }
positions=[]
estimated=[]
scale_factor=0.1
SCALE_ADJUSTMENT_FACTOR=0.05
prev_keypoint, prev_descriptor = None, None

while (1):
    estimated_position=[]
    drone_position,drone_orientation=p.getBasePositionAndOrientation(drone)
    positions.append(drone_position)
    prev_position=drone_position

    keys_pressed=p.getKeyboardEvents()
    if ord('x') in keys_pressed and keys_pressed[ord('x')] & p.KEY_WAS_TRIGGERED:
        break
    for key,movement in keys.items():
        if ord(key) in keys_pressed and keys_pressed[ord(key)] & p.KEY_WAS_TRIGGERED:
            drone_position = [drone_position[i] + movement[i] for i in range(3)]
            p.resetBasePositionAndOrientation(drone,drone_position,drone_orientation)
    if np.all(prev_position==drone_position):
        continue

    view_matrix=p.computeViewMatrix(
        cameraEyePosition=[drone_position[0]+camera_position[0],
                           drone_position[1]+camera_position[1],
                           drone_position[2]+camera_position[2]],
        cameraTargetPosition=[drone_position[0]+camera_target[0],
                              drone_position[1]+camera_target[1],
                              drone_position[2]+camera_target[2]],
        cameraUpVector=[1,0,0]
    )

    images=p.getCameraImage(width,height,view_matrix,projection_matrix)
    rgba_img=np.reshape(images[2],(height, width,4))
    rgb_img=rgba_img[:,:,:3]

    keypoint_list,descriptor_list=orb.detectAndCompute(rgb_img,None)
    if prev_keypoint is not None and descriptor_list is not None:
        matches = bf_matcher.match(prev_descriptor, descriptor_list)
        matches = sorted(matches, key=lambda x: x.distance)

        quality_matches = [m for m in matches if m.distance < 30]
        
        old_keypoints_matched = np.array([prev_keypoint[m.queryIdx].pt for m in quality_matches], dtype=np.float32).reshape(-1, 1, 2)
        new_keypoints_matched = np.array([keypoint_list[m.trainIdx].pt for m in quality_matches], dtype=np.float32).reshape(-1, 1, 2)

        if len(quality_matches) >= 8:
            E, mask = cv2.findEssentialMat(new_keypoints_matched, old_keypoints_matched, intrinsic_matrix, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, _ = cv2.recoverPose(E,new_keypoints_matched,old_keypoints_matched,intrinsic_matrix)
            t=(t*scale_factor).ravel()

            scale_factor += SCALE_ADJUSTMENT_FACTOR * (0.1 - np.linalg.norm(t))
            estimated_position=[prev_position[i]+t[i] for i in range(3)]
            estimated.append(estimated_position)

            print("Translation matrix: \n", t)
            print(f"Actual translation : \n{np.array(drone_position)-np.array(prev_position)}")
            print(f"Estimated position : \n{estimated_position}")
            print(f"Actual position : \n{drone_position}")
            print(f"Difference between actual and estimated position: {np.array(drone_position) - np.array(estimated_position)}\n\n\n")

    prev_descriptor=descriptor_list
    prev_keypoint=keypoint_list
p.disconnect()

positions=list(zip(*positions))
estimated=list(zip(*estimated))
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')
ax.plot(positions[0],positions[1],positions[2],label='Drone Trajectory')
ax.plot(estimated[0],estimated[1],estimated[2],label='Estimated Trajectory')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.legend()

timestamp=datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f'./trajectories/odometry_{timestamp}.png')
plt.show()
