import pybullet as p
import time
import pybullet_data as pd
import cv2
import numpy as np

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.setGravity(0, 0, 0)

quaternion = p.getQuaternionFromEuler([1.57, 0, 0])
forest = p.loadURDF("./forest/forest.urdf",basePosition=[0,0,0],baseOrientation=quaternion,useFixedBase=True)
drone=p.loadURDF("./cf2/cf2.urdf",globalScaling=100)
texture_id = p.loadTexture("./forest/forest.png")
p.changeVisualShape(forest, -1, textureUniqueId=texture_id)
# Camera setup
fov = 60  # Field of view in degrees
aspect = 1  # Aspect ratio
near = 0.02  # Near clipping plane
far = 1  # Far clipping plane
width, height = 320, 320  # Image size

camera_position = [0, 0, 1]
camera_orientation = [0, 0, 0, 1]  # Quaternion for no rotation

drone_position, drone_orientation = p.getBasePositionAndOrientation(drone)

view_matrix = p.computeViewMatrix(camera_position, [0, 0, 0], [0, 1, 0])

projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

images = p.getCameraImage(width, height, view_matrix, projection_matrix)

rgb_img = np.reshape(images[2], (height, width, 4))  # RGBA image
depth_img = np.reshape(images[3], (height, width))  # Depth image
seg_img = np.reshape(images[4], (height, width))  # Segmentation image

if seg_img.dtype != np.uint8:
    seg_img = cv2.convertScaleAbs(seg_img)


cv2.imshow("RGB Image", rgb_img[:, :, :3])
cv2.imshow("Depth Image", depth_img)
cv2.imshow("Segmentation Image", seg_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


p.disconnect()
