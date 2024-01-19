from py_thts_env import PyThtsEnv
import mo_gymnasium as mo_gym

def convert_numpy_array_to_int_list(arr):
    try:
        return [float(a) for a in arr]
    except:
        return arr

def convert_numpy_array_to_float_list(arr):
    try:
        return [float(a) for a in arr]
    except:
        return arr

class MoGymThtsEnv(PyThtsEnv):

    def __init__(self, mo_gym_env_id):
        self.fully_observable = True # might not be, but we dont have partial obs stuff implemented yet TODO?
        self.gym_env_id=mo_gym_env_id
        self.env = mo_gym.make(mo_gym_env_id)
        _, _ = self.env.reset()
        _, reward, _, _, _ = self.env.step(0)
        init_gym_state, _ = self.env.reset()
        self.init_gym_state = convert_numpy_array_to_int_list(init_gym_state)
        self.reward_dim = reward.shape[0]


    def get_reward_dim(self):
        return self.reward_dim

    def reset(self):
        """
        Reset any per trial state held in this env here
        """
        init_gym_state, _ = self.env.reset()
        self.init_gym_state = convert_numpy_array_to_int_list(init_gym_state)
        

    def get_initial_state(self):
        """
        Returns the initial state of the environment
        """
        return (self.init_gym_state, False)

    def is_sink_state(self, state):
        """
        Returns if 'state' is a sink state
        """
        _gym_state, is_sink = state
        return is_sink
        

    def get_valid_actions(self, state):
        """
        Returns a list of valid action objects that can be taken from 'state'
        """
        return [i for i in range(self.env.action_space.n)]

    def get_transition_distribution(self, state, action):
        """
        Returns a dictionary mapping from next states to their transition probabilities
        This function is optional (but recommended)
            (-recommended as it lets us avoid having to repetatively call 'sample_transition_distribution' and use the 
                python interpreter, which is slow)
        """
        raise Exception("Trying get transition distribution from python thts env without defining "
                        "'get_transition_distribution'")

    def sample_transition_distribution(self, state, action):
        """
        Samples a 'next_state' object from Pr('next_state'|'state','action') and returns it
        """
        obs, reward, terminated, truncated, _info = self.env.step(action)
        self.last_reward = reward
        return (convert_numpy_array_to_int_list(obs), (terminated or truncated))
        

    def get_observation_distribution(self, state, action):
        """
        Returns a dictionary mapping from observations to their observation probabilities
        In fully observable environments observations are just the next states
        """
        if self.is_fully_observable:
            return self.get_transition_distribution(state=state, action=action)
        raise Exception("Trying get observation distribution from python thts env without defining "
                        "'get_transition_distribution'")

    def sample_observation_distribution(self, state, action):
        """
        Samples a 'obs' object from Pr('obs'|'state','action') and returns it
        """
        if self.is_fully_observable:
            return self.sample_transition_distribution(state=state, action=action)
        raise Exception("Trying to sample observation from python thts env without defining "
                        "'sample_observation_distribution'")

    def get_reward(self, state, action):
        """
        Cant return np.array for reward because numpy doesnt work in subinterpreters yet, so wouldn't work in the 
        main process worker threads when need to translate reward to C++
        So this converts it to a list
        """
        return convert_numpy_array_to_float_list(self.last_reward)
    