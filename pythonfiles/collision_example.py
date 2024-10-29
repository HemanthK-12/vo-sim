import pybullet as p
import pybullet_data
import time


p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())


box_id = p.loadURDF("cube.urdf", basePosition=[0, 0, 1])  # Replace with a box URDF if needed


drone_id = p.loadURDF('./cf2/cf2.urdf', basePosition=[0, 0, 5],baseOrientation=p.getQuaternionFromEuler([1.57,0,0]),globalScaling=3)


step = 0.1
yaw_step = 0.1
keys = {
    p.B3G_UP_ARROW: [-step, 0, 0],
    p.B3G_DOWN_ARROW: [step, 0, 0],
    p.B3G_LEFT_ARROW: [0, -step, 0],
    p.B3G_RIGHT_ARROW: [0, step, 0],
    ord('w'): [0, 0, step],
    ord('s'): [0, 0, -step],
    ord('a'): -yaw_step,
    ord('d'): yaw_step
}

yaw_angle = 0


while True:
    keys_pressed = p.getKeyboardEvents()
    if ord('x') in keys_pressed and keys_pressed[ord('x')] & p.KEY_IS_DOWN:
        break

    drone_position, drone_orientation = p.getBasePositionAndOrientation(drone_id)
    for key, movement in keys.items():
        if key in keys_pressed and keys_pressed[key] & p.KEY_IS_DOWN:
            if key in [ord('a'), ord('d')]:
                yaw_angle += movement
                drone_orientation = p.getQuaternionFromEuler([0, 0, yaw_angle])
            else:
                drone_position = [drone_position[i] + movement[i] for i in range(3)]
            p.resetBasePositionAndOrientation(drone_id, drone_position, drone_orientation)


    p.stepSimulation()

    contact_points = p.getContactPoints()
    if contact_points:
        print("Collision detected!")
        for contact in contact_points:
            print(f"Object A ID: {contact[1]}, Object B ID: {contact[2]}, Position: {contact[5]}")


    time.sleep(1. / 240.)

p.disconnect()