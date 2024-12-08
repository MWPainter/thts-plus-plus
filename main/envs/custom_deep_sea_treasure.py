"""
Want a custom fruit deep sea treasure environment, which is adapted from:
(paper) https://arxiv.org/pdf/2110.06742
(code) https://github.com/imec-idlab/deep-sea-treasure/tree/master

Code is in the folder main/envs/dst

Outline:
1. Edit DST code to include currents (to provide some stochasticity)
1.1. All of the changes are marked with comments including "THTS additions:", so can ctrl-f to see changes
1.2. Physics was a bit janky when velocity can be more than 1 in any direction, but I dont want to go into weeds 
    editing collision code. (I.e. I want minimal changes from a generalised)
2. (Possibly at time of writing) adapt code from MORL Glue Generalised Deep Sea Treasure that generates maps with a 
    variable size for Deep Sea Treasure environments
3. make a 'DstThtsEnv' class, which is basically a custom 'MoGymThtsEnv' wrapping around the above and imported code
"""

#####
# DstThtsEnv
#####

from deep_sea_treasure import DeepSeaTreasureV0, VamplewWrapper, FuelWrapper
from mo_gym_thts_env import MoGymThtsEnv

class ImprovedDeepSeaTreasureThtsEnv(MoGymThtsEnv):

    def __init__(self, swept_by_current_prob=0.0, is_vamplew=False):
        swept_by_current_prob = float(swept_by_current_prob)
        is_vamplew = bool(int(is_vamplew))

        self.fully_observable = True
        
        self.env = DeepSeaTreasureV0.new(
            max_steps=1000,
            swept_by_current_prob=swept_by_current_prob
        )

        self.is_vamplew = is_vamplew
        if is_vamplew:
            self.env = VamplewWrapper.new(self.env)
        else:
            self.env = FuelWrapper.new(self.env)

        _, _ = self.env.reset()
        action = 0 if is_vamplew else (0,0)
        _, reward, _, _, _ = self.env.step(action)
        self.init_gym_state, _ = self.env.reset()
        self.reward_dim = reward.shape[0]

        self.rollout_state_cache = {}
        self.rollout_action_cache = {}
        self.rollout_reward_cache = {}

    def get_valid_actions(self, state):
        if self.is_vamplew:
            return super().get_valid_actions(state)
        n = self.env.action_space.spaces[0].n
        m = self.env.action_space.spaces[1].n
        return [(i,j) for i in range(n) for j in range(m)]

