# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym  # noqa: F401
from isaaclab.envs import ManagerBasedRLEnv
from .auto_drone_rl.hover_env_cfg import HoverEnvCfg

gym.register(
    id="Drone-Hover-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={"env_cfg_entry_point":HoverEnvCfg},
)
