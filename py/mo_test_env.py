from mo_py_thts_env import MoPyThtsEnv
import numpy as np
import random

RIGHT = 0
DOWN = 1

class MoPyTestThtsEnv(MoPyThtsEnv):
    """
    """

    def __init__(self, 
        walk_len, 
        wrong_dir_prob=0.0, 
        add_extra_rewards=False, 
        new_dir_bonus=0.5, 
        same_dir_bonus=0.3, 
        gamma=0.5):
        
        self.reward_dim = 4 if add_extra_rewards else 2
        super().__init__(reward_dim=self.reward_dim, fully_observable=True)
        self.walk_len = walk_len
        self.wrong_dir_prob = wrong_dir_prob         
        self.add_extra_rewards = add_extra_rewards
        self.new_dir_bonus = new_dir_bonus
        self.same_dir_bonus = same_dir_bonus
        self.gamma = gamma

    def _x(self, state):
        return state[0]
    
    def _y(self, state):
        return state[1]
    
    def _get_last_direction(self, state):
        return state[2]

    def get_initial_state(self):
        return (0,0,-1)
    
    def is_sink_state(self, state):
        return (self._x(state) + self._y(state) == self.walk_len);

    def get_valid_actions(self, state):
        if (self.is_sink_state(state)):
            return []
        return [RIGHT, DOWN]
    
    def candidate_next_state(self, state, action, wrong_dir):
        x,y,_ = state
        direction = action
        if wrong_dir:
            direction = 1 - action
        
        if (direction == RIGHT):
            return (x+1,y,RIGHT)
        else:
            return (x,y+1,DOWN)
    
    def get_transition_distribution(self, state, action):
        cand_next_state = self.candidate_next_state(state, action, False)
        distr = {cand_next_state: 1.0 - self.stay_prob}
        if (self.wrong_dir_prob > 0.0):
            wrong_dir_state = self.candidate_next_state(state, action, True)
            distr[wrong_dir_state] = self.stay_prob
        return distr
    
    def sample_transition_distribution(self, state, action):
        if (self.stay_prob == 0.0):
            return self.candidate_next_state(state,action, False)
        
        r = random.random()
        return self.candidate_next_state(state, action, (r < self.wrong_dir_prob))
        
    def get_reward(self, state, action):
        r = np.zeros(4)
        r[RIGHT] = -1.0
        r[DOWN] = -1.0
        r[action] += self.same_dir_bonus if self._get_last_direction(state) == action else self.new_dir_bonus
        
        if not self.add_extra_rewards:
            return r
        
        if (action == RIGHT):
            r[2] = self.gamma ** self._x(state)
        else:
            r[3] = self.gamma ** self._y(state)
        return r