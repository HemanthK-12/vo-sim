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
width, height = 1920, 1088

camera_position = [0, 0, -0.1]
camera_target = [-1, 0, 0]
fx=width/(2*(np.tan((np.deg2rad(fov)/2))))
fy=height/(2*(np.tan((np.deg2rad(fov)/2))))

intrinsic_matrix=np.array([[fx,0,width/2],
                          [0,fy,height/2],
                          [0,0,1]])
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

frame_rate = 10
duration = 10
total_frames = frame_rate * duration

keys = {
    'w': [0.1, 0, 0],
    's': [-0.1, 0, 0],
    'a': [0, -0.1, 0],
    'd': [0, 0.1, 0],
    'q': [0, 0, 0.1],
    'e': [0, 0, -0.1]
}

orb = cv2.ORB_create()
bf_matcher=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)

positions=[]
prev_keypoint=None
prev_descriptor=None
for i in range(total_frames):

    old_keypoints_matched=[]
    new_keypoints_matched=[]

    drone_position, drone_orientation = p.getBasePositionAndOrientation(drone)
    positions.append(drone_position)

    keys_pressed = p.getKeyboardEvents()
    for key, movement in keys.items():
        if ord(key) in keys_pressed and keys_pressed[ord(key)] & p.KEY_WAS_TRIGGERED:
           drone_position = [drone_position[j] + movement[j] for j in range(3)]
           p.resetBasePositionAndOrientation(drone, drone_position, drone_orientation)
    
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
        
    keypoint_list, descriptor_list = orb.detectAndCompute(rgb_img, None) 
    if prev_keypoint is not None:
        matches=bf_matcher.match(prev_descriptor,descriptor_list)
        matches=sorted(matches,key=lambda x:x.distance)
        for i in matches:
            old_keypoints_matched.append(prev_keypoint[i.queryIdx].pt)
            new_keypoints_matched.append(keypoint_list[i.trainIdx].pt)
        old_keypoints_matched=np.array(old_keypoints_matched,dtype=np.float32)
        new_keypoints_matched=np.array(new_keypoints_matched,dtype=np.float32)
        E,_=cv2.findEssentialMat(new_keypoints_matched,old_keypoints_matched,intrinsic_matrix)
        _,r,t,_=cv2.recoverPose(E,new_keypoints_matched,old_keypoints_matched,intrinsic_matrix)
        old_projection_matrix=np.hstack((np.eye(3),np.zeros((3,1))))
        new_projection_matrix=np.hstack((r,t))
        homogenous_4d_pts=cv2.triangulatePoints(old_projection_matrix,new_projection_matrix,old_keypoints_matched.T,new_keypoints_matched.T)
        estimated_3d_point=homogenous_4d_pts[:3]/homogenous_4d_pts[3]
        
    
    prev_descriptor=descriptor_list
    prev_keypoint=keypoint_list
    prev_img=rgb_img.copy()
print(drone_position," ",estimated_3d_point)
p.disconnect()
cv2.destroyAllWindows()