from py_thts_env import PyThtsEnv

class MoPyThtsEnv(PyThtsEnv):
    """
    A ThtsEnv baseclass defined in python 
    """

    def __init__(self, reward_dim, fully_observable=True):
        self.reward_dim = reward_dim
        self.fully_observable=fully_observable

    def get_reward_dim(self):
        return self.reward_dim
    