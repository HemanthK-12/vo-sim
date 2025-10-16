from isaaclab.envs import ManagerBasedRLEnvCfg #since everything is manager based here due to pipeline
from .auto_drone_rl_env_cfg import AutoDroneRlEnvCfg as BaseCfg #importin the base auto-generated env config while creating the external project
from . import mdp # importing all reward functions details folder
from isaaclab.utils import configclass

@configclass
class HoverRewards:
    alive=BaseCfg().rewards.alive
    altitude=mdp.AltitudeReward(weight=1.0)


#new env config class for new task, can include this in auto_drone_rl_env_cfg.py itself but this is better in terms of modularity
class HoverEnvCfg(BaseCfg):     
    def __post_init__(self): 
        '''
        constructor which executes after the __init__ of default env config, for hover task, hence post init
        '''
        super().__post_init__()
        self.episode_length=5 # for now
        self.rewards=HoverRewards()

