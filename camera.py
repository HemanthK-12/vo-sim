import matplotlib.pyplot as plt
import numpy as np
import pybullet as p
import pybullet_data as pd
import time
import imageio

p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())
p.loadURDF('./cf2/cf2.urdf', globalScaling=5)
quaternion = p.getQuaternionFromEuler([1.57, 0, 0])
forest = p.loadURDF("./forest/forest.urdf",basePosition=[0,0,0],baseOrientation=quaternion,useFixedBase=True)
texture_id = p.loadTexture("./forest/forest.png")
p.changeVisualShape(forest, -1, textureUniqueId=texture_id)

width = 1920
height = 1080

fov = 60
aspect = width / height
near = 0.02
far = 1

view_matrix = p.computeViewMatrix([0, 0, 1], [0, 0, 0], [0, 1, 0])
projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

# Get depth values using the OpenGL renderer
images = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
depth_buffer_opengl = np.reshape(images[3], [width, height])
depth_opengl = far * near / (far - (far - near) * depth_buffer_opengl)

rgb_img = np.reshape(images[2], (height, width, 4))[:, :, :3]
imageio.imwrite('camera_view.png', rgb_img)

for i in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()