# Here, 
#     Agent : Iris drone
#     Environment : Black Gridroom env through pegasus in Isaac sim simulator
#     Action : Controlling the rpm of all 4 motors of the drone
#     State : Position, orientation, angular velocity of all the motors of the drone
#     Reward : current altitude-targeted altitude
#     Goal : Hover at targeted altitude

#     State Space : Position, Orientation, rpm of all 4 motors
#     Action Space : rpm of all 4 motors
#     Reward Function : current altitude-targeted altitude

# We'll use a normal Proximal Policy Optimization reinforcement learning algorithm
# and for this we'll use stable-baselines3 lib.

# PPO: 