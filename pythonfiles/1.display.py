import pybullet as p
import pybullet_data as pd
import time

p.connect(p.GUI, options="--mouse_move_camera=1")
p.setAdditionalSearchPath(pd.getDataPath())
p.setGravity(0, 0, 0)

quaternion = p.getQuaternionFromEuler([1.57, 0, 0])
quaternion2 = p.getQuaternionFromEuler([0, -1.57, 0])

forest = p.loadURDF("./forest/forest.urdf",basePosition=[0,0,0],baseOrientation=quaternion,useFixedBase=True)
cf2=p.loadURDF("./cf2/cf2.urdf",basePosition=[0,0,3],globalScaling=50.0,baseOrientation=quaternion2)
propeller_joints = [0, 1, 2, 3]

texture_id = p.loadTexture("./forest/forest.png")
p.changeVisualShape(forest, -1, textureUniqueId=texture_id)

for _ in range(10000):
    p.stepSimulation()
    time.sleep(1./240.)

p.disconnect()