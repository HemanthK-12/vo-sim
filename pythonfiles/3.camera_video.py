import numpy as np
import pybullet as p
import pybullet_data as pd
import time
import imageio
from datetime import datetime

# Connect to PyBullet in GUI mode
p.connect(p.GUI)
p.setAdditionalSearchPath(pd.getDataPath())

# Load the URDF model
drone=p.loadURDF('./cf2/cf2.urdf', globalScaling=5, basePosition=[0,0,2])
quaternion = p.getQuaternionFromEuler([1.57, 0, 0])
forest = p.loadURDF("./forest/forest.urdf", basePosition=[0, 0, 0], baseOrientation=quaternion, useFixedBase=True)
texture_id = p.loadTexture("./forest/forest.png")
p.changeVisualShape(forest, -1, textureUniqueId=texture_id)

width = 1920
height = 1088

fov = 60
aspect = width / height
near = 0.02
far = 1

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
video_writer = imageio.get_writer(f'./video/{timestamp}.mp4', fps=30)

for frame in range(3000):  

    view_matrix = p.computeViewMatrix([0, 0, 1], [0, 0, 0], [0, 1, 0])
    projection_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)

    images = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

    # Extract the RGB image
    rgb_img = np.reshape(images[2], (height, width, 4))[:, :, :3]

    # Write the image to the video file
    video_writer.append_data(rgb_img)

    p.stepSimulation()
    time.sleep(1./240.)

video_writer.close()