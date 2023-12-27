


class PyThtsEnv:
    """
    A ThtsEnv baseclass defined in python 
    """

    def __init__(self, fully_observable=True):
        self.fully_observable=fully_observable

    def is_fully_observable(self):
        return self.fully_observable

    def get_init_state(self):
        """
        Returns the initial state of the environment
        """
        pass

    def is_sink_state(self, state):
        """
        Returns if 'state' is a sink state
        """
        pass

    def get_valid_actions(self, state):
        """
        Returns a list of valid action objects that can be taken from 'state'
        """
        pass

    def get_transition_distribution(self, state, action):
        """
        Returns a dictionary mapping from next states to their transition probabilities
        This function is optional (but recommended)
            (-recommended as it lets us avoid having to repetatively call 'sample_transition_distribution' and use the 
                python interpreter, which is slow)
        """
        pass

    def sample_transition_distribution(self, state, action):
        """
        Samples a 'next_state' object from Pr('next_state'|'state','action') and returns it
        """
        pass

    def get_observation_distribution(self, state, action):
        """
        Returns a dictionary mapping from observations to their observation probabilities
        In fully observable environments observations are just the next states
        """
        if self.is_fully_observable:
            return self.get_transition_distribution(state, action)
        raise Exception("Trying get observation distribution from python thts env without defining "
                        "'get_transition_distribution'")

    def sample_observation_distribution(self, state, action):
        """
        Samples a 'obs' object from Pr('obs'|'state','action') and returns it
        """
        if self.is_fully_observable:
            return self.sample_transition_distribution(state, action)
        raise Exception("Trying to sample observation from python thts env without defining "
                        "'sample_observation_distribution'")

    def get_reward(self, state, action, observation=None):
        """
        Returns reward given a state, action, observation tuple
        """
        pass

    def sample_context(self, state):
        """
        Returns a thts context to be used throughout the trial
        Most of the time this will be unused, or a dictionary is sufficient
        """
        return {}
    