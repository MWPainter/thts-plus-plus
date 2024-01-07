from py_thts_env import PyThtsEnv

import random


class PyTestThtsEnv(PyThtsEnv):
    """
    A python implementation of our test env
    States are (x,y) tuples
    Actions are strings in {"left", "down", "right", "up"}
    """

    def __init__(self, grid_size, stay_prob=0.0):
        super().__init__(fully_observable=True)
        self.grid_size = grid_size
        self.stay_prob = stay_prob

    def get_initial_state(self):
        return (0,0)
    
    def is_sink_state(self, state):
        return (state[0] == self.grid_size and state[1] == self.grid_size)

    def get_valid_actions(self, state):
        if (self.is_sink_state(state)):
            return []
        
        x,y = state
        valid_actions = []
        if (x > 0):
            valid_actions.append("left")
        if (x < self.grid_size):
            valid_actions.append("right")
        if (y > 0):
            valid_actions.append("up")
        if (y < self.grid_size):
            valid_actions.append("down")
        return valid_actions
    
    def candidate_next_state(self, state, action):
        x,y = state
        if (action == "left"):
            return (x-1,   y)
        if (action == "right"):
            return (x+1,   y)
        if (action == "down"):
            return (  x, y+1)
        if (action == "up"):
            return (x  , y-1)
        raise Exception("Something went wrong in python test env")
    
    def get_transition_distribution(self, state, action):
        cand_next_state = self.candidate_next_state(state, action)
        distr = {cand_next_state: 1.0 - self.stay_prob}
        if (self.stay_prob > 0.0):
            distr[state] = self.stay_prob
        return distr
    
    def sample_transition_distribution(self, state, action):
        if (self.stay_prob == 0.0):
            return self.candidate_next_state(state,action)
        
        r = random.random()
        if (r > self.stay_prob):
            return self.candidate_next_state(state,action)
        else:
            return state
        
    def get_reward(self, state, action, observation=None):
        return -1.0
