from isaaclab.envs import make
env = make("Drone-Hover-v0")

print(env)                  # should show ManagerBasedRLEnv
print(env.cfg)              # should show HoverEnvCfg
print(env.scene["drone"])   # should not raise KeyError
